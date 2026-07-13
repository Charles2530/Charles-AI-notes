#!/usr/bin/env bash
set -euo pipefail

APP_PATH="${APP_PATH:-/Applications/ChatGPT.app}"
CODEX_SUPPORT_DIR="${CODEX_SUPPORT_DIR:-$HOME/Library/Application Support/Codex}"
ROOT_DB="$CODEX_SUPPORT_DIR/Local Storage/leveldb"
DEFAULT_DB="$CODEX_SUPPORT_DIR/Default/Local Storage/leveldb"
LOG_FILE="${LOG_FILE:-/tmp/codex-desktop-model-cache-fix.log}"
FUTURE_MS="${FUTURE_MS:-1893456000000}"

APPLY=false
YES=false
RESTART_APP=true
KEEP_TEMP=false
SELF_TEST=false
TEMP_ROOT=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/fix_codex_desktop_models_macos.sh --dry-run
  bash scripts/fix_codex_desktop_models_macos.sh --apply --yes

Options:
  --dry-run       Copy the LevelDB directories to /private/tmp, patch the copies,
                  and verify the result. This is the default.
  --apply         Patch the real Codex Desktop App Local Storage cache.
  --yes           Skip the confirmation prompt for --apply.
  --no-restart    Do not reopen ChatGPT.app after a successful --apply.
  --keep-temp     Keep temporary npm package/cache directories for inspection.
  --self-test     Run script sanity tests without touching app data.
  -h, --help      Show this help.

Environment overrides:
  APP_PATH=/Applications/ChatGPT.app
  CODEX_SUPPORT_DIR="$HOME/Library/Application Support/Codex"
  LOG_FILE=/tmp/codex-desktop-model-cache-fix.log
  FUTURE_MS=1893456000000
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

cleanup_temp() {
  if [[ -z "${TEMP_ROOT:-}" ]]; then
    return 0
  fi

  if [[ "$KEEP_TEMP" != true ]]; then
    rm -rf "$TEMP_ROOT"
  else
    log "Keeping temp directory: $TEMP_ROOT"
  fi
}

have() {
  command -v "$1" >/dev/null 2>&1
}

confirm_apply() {
  if [[ "$YES" == true ]]; then
    return 0
  fi

  printf '%s\n' "This will close ChatGPT.app/Codex Desktop App, back up LevelDB, patch Statsig cache, and reopen the app."
  printf '%s' "Type yes to continue: "
  read -r answer
  [[ "$answer" == "yes" ]] || die "User cancelled."
}

run_self_test() {
  bash -n "$0"
  node <<'NODE'
const assert = require('assert');

function patchConfig(cfg) {
  const newModels = ['gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-5.6-luna'];
  const oldModels = Array.isArray(cfg.available_models) ? cfg.available_models : [];
  cfg.available_models = [...newModels, ...oldModels.filter((m) => !newModels.includes(m))];
  cfg.default_model = 'gpt-5.6-sol';
  cfg.use_hidden_models = false;
  return cfg;
}

const cfg = patchConfig({
  available_models: ['gpt-5.5', 'gpt-5.4', 'gpt-5.4-mini', 'gpt-5.3-codex', 'gpt-5.2'],
  default_model: 'gpt-5.4',
  use_hidden_models: true,
});

assert.deepStrictEqual(cfg.available_models.slice(0, 3), ['gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-5.6-luna']);
assert.strictEqual(cfg.default_model, 'gpt-5.6-sol');
assert.strictEqual(cfg.use_hidden_models, false);
assert.strictEqual(new Set(cfg.available_models).size, cfg.available_models.length);

console.log('SELF_TEST_OK');
NODE
}

parse_args() {
  if [[ "$#" -eq 0 ]]; then
    return 0
  fi

  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --dry-run)
        APPLY=false
        ;;
      --apply)
        APPLY=true
        ;;
      --yes)
        YES=true
        ;;
      --no-restart)
        RESTART_APP=false
        ;;
      --keep-temp)
        KEEP_TEMP=true
        ;;
      --self-test)
        SELF_TEST=true
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        usage >&2
        die "Unknown option: $1"
        ;;
    esac
    shift
  done
}

db_specs() {
  local found=false
  if [[ -d "$ROOT_DB" ]]; then
    printf 'root:%s\n' "$ROOT_DB"
    found=true
  fi
  if [[ -d "$DEFAULT_DB" ]]; then
    printf 'default:%s\n' "$DEFAULT_DB"
    found=true
  fi
  [[ "$found" == true ]] || die "No Codex Local Storage LevelDB directories found under: $CODEX_SUPPORT_DIR"
}

install_leveldb_package() {
  local pkg_dir="$1"
  have npm || die "npm is required to install the temporary classic-level package."
  have node || die "node is required."
  log "Installing temporary LevelDB helper package under: $pkg_dir"
  npm install --prefix "$pkg_dir" classic-level >/dev/null 2>&1
}

copy_db_specs() {
  local work_dir="$1"
  shift
  local spec label path copy_path
  for spec in "$@"; do
    label="${spec%%:*}"
    path="${spec#*:}"
    copy_path="$work_dir/$label-leveldb"
    cp -R "$path" "$copy_path"
    printf '%s:%s\n' "$label" "$copy_path"
  done
}

backup_db_specs() {
  local backup_dir="$1"
  shift
  local spec label path backup_path
  mkdir -p "$backup_dir"
  for spec in "$@"; do
    label="${spec%%:*}"
    path="${spec#*:}"
    backup_path="$backup_dir/$label-leveldb"
    cp -R "$path" "$backup_path"
    log "Backed up $label LevelDB to: $backup_path"
  done
}

lock_pids() {
  local pids=()
  if [[ -f "$ROOT_DB/LOCK" ]]; then
    while IFS= read -r pid; do pids+=("$pid"); done < <(lsof -t "$ROOT_DB/LOCK" 2>/dev/null || true)
  fi
  if [[ -f "$DEFAULT_DB/LOCK" ]]; then
    while IFS= read -r pid; do pids+=("$pid"); done < <(lsof -t "$DEFAULT_DB/LOCK" 2>/dev/null || true)
  fi
  if [[ "${#pids[@]}" -gt 0 ]]; then
    printf '%s\n' "${pids[@]}" | sort -u
  fi
}

release_locks_for_apply() {
  have lsof || die "lsof is required to find the LevelDB lock owner."

  log "Asking ChatGPT.app to quit."
  osascript -e 'tell application id "com.openai.codex" to quit' >/dev/null 2>&1 || \
    osascript -e 'tell application "ChatGPT" to quit' >/dev/null 2>&1 || true

  local pids
  for _ in $(seq 1 30); do
    pids="$(lock_pids | tr '\n' ' ' || true)"
    [[ -z "$pids" ]] && return 0
    sleep 0.5
  done

  pids="$(lock_pids | tr '\n' ' ' || true)"
  if [[ -n "$pids" ]]; then
    log "Sending SIGTERM to LevelDB lock owners: $pids"
    kill -TERM $pids 2>/dev/null || true
  fi

  for _ in $(seq 1 20); do
    pids="$(lock_pids | tr '\n' ' ' || true)"
    [[ -z "$pids" ]] && return 0
    sleep 0.5
  done

  pids="$(lock_pids | tr '\n' ' ' || true)"
  if [[ -n "$pids" ]]; then
    log "Sending SIGKILL to remaining LevelDB lock owners: $pids"
    kill -KILL $pids 2>/dev/null || true
  fi

  for _ in $(seq 1 20); do
    pids="$(lock_pids | tr '\n' ' ' || true)"
    [[ -z "$pids" ]] && return 0
    sleep 0.5
  done

  log "Current lock owners:"
  if [[ -f "$DEFAULT_DB/LOCK" ]]; then
    lsof "$DEFAULT_DB/LOCK" || true
  fi
  die "Could not release Codex LevelDB lock. Close ChatGPT.app manually and rerun this script."
}

patch_and_verify() {
  local pkg_dir="$1"
  shift
  node - "$pkg_dir" "$FUTURE_MS" "$@" <<'NODE'
(async () => {
  const { ClassicLevel } = require(process.argv[2] + '/node_modules/classic-level');
  const futureMs = Number(process.argv[3]);
  const specs = process.argv.slice(4).map((spec) => {
    const sep = spec.indexOf(':');
    return { label: spec.slice(0, sep), dir: spec.slice(sep + 1) };
  });
  const newModels = ['gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-5.6-luna'];

  function decodeValue(buf) {
    if (buf[0] === 0x00 && buf[1] === 0x7b) {
      return { text: buf.subarray(1).toString('utf16le'), encoding: 'nul-utf16le' };
    }
    if (buf[0] === 0x00 && buf[2] === 0x7b) {
      return { text: buf.subarray(2).toString('utf16le'), encoding: 'nul2-utf16le' };
    }
    return { text: buf.toString('utf8').replace(/^\u0000+/, '').replace(/^\u0001+/, ''), encoding: 'utf8' };
  }

  function encodeValue(text, encoding) {
    if (encoding === 'nul-utf16le') return Buffer.concat([Buffer.from([0]), Buffer.from(text, 'utf16le')]);
    if (encoding === 'nul2-utf16le') return Buffer.concat([Buffer.from([0, 0]), Buffer.from(text, 'utf16le')]);
    return Buffer.from(text, 'utf8');
  }

  function keyText(buf) {
    return buf.toString('utf8').replace(/^\u0000+|^\u0001+/, '').replace(/[^\x20-\x7E]+/g, ' ');
  }

  function parseJson(text) {
    try {
      return JSON.parse(text);
    } catch {
      return null;
    }
  }

  const report = [];

  for (const { label, dir } of specs) {
    const db = new ClassicLevel(dir, { keyEncoding: 'buffer', valueEncoding: 'buffer' });
    await db.open();
    const modifiedSuffixes = [];

    for await (const [keyBuf, valueBuf] of db.iterator()) {
      const key = keyText(keyBuf);
      if (!key.includes('statsig.cached.evaluations.')) continue;

      const decoded = decodeValue(valueBuf);
      const outer = parseJson(decoded.text);
      if (!outer || typeof outer.data !== 'string') continue;

      const inner = parseJson(outer.data);
      const cfg = inner?.dynamic_configs?.['107580212']?.value;
      if (!cfg || typeof cfg !== 'object') continue;

      const oldModels = Array.isArray(cfg.available_models) ? cfg.available_models : [];
      cfg.available_models = [...newModels, ...oldModels.filter((model) => !newModels.includes(model))];
      cfg.default_model = 'gpt-5.6-sol';
      cfg.use_hidden_models = false;
      inner.time = futureMs;
      outer.receivedAt = futureMs;
      outer.data = JSON.stringify(inner);

      await db.put(keyBuf, encodeValue(JSON.stringify(outer), decoded.encoding));
      modifiedSuffixes.push(key.match(/statsig\.cached\.evaluations\.([0-9]+)/)?.[1] ?? key);
    }

    for await (const [keyBuf, valueBuf] of db.iterator()) {
      const key = keyText(keyBuf);
      if (!key.includes('statsig.last_modified_time.evaluations')) continue;

      const decoded = decodeValue(valueBuf);
      const obj = parseJson(decoded.text);
      if (!obj || typeof obj !== 'object') continue;

      for (const suffix of modifiedSuffixes) {
        obj[`statsig.cached.evaluations.${suffix}`] = futureMs;
      }
      await db.put(keyBuf, encodeValue(JSON.stringify(obj), decoded.encoding));
    }

    report.push({ label, dir, modifiedStatsigEntries: modifiedSuffixes.length, modifiedSuffixes });
    await db.close();
  }

  const summaries = [];

  for (const { label, dir } of specs) {
    const db = new ClassicLevel(dir, { keyEncoding: 'buffer', valueEncoding: 'buffer' });
    await db.open();
    for await (const [keyBuf, valueBuf] of db.iterator()) {
      const key = keyText(keyBuf);
      if (!key.includes('statsig.cached.evaluations.')) continue;

      const outer = parseJson(decodeValue(valueBuf).text);
      const inner = typeof outer?.data === 'string' ? parseJson(outer.data) : null;
      const cfg = inner?.dynamic_configs?.['107580212']?.value;
      if (!cfg) continue;

      summaries.push({
        label,
        suffix: key.match(/statsig\.cached\.evaluations\.([0-9]+)/)?.[1] ?? key,
        receivedAt: outer.receivedAt,
        innerTime: inner.time,
        default_model: cfg.default_model,
        use_hidden_models: cfg.use_hidden_models,
        hasSol: Array.isArray(cfg.available_models) && cfg.available_models.includes('gpt-5.6-sol'),
        hasTerra: Array.isArray(cfg.available_models) && cfg.available_models.includes('gpt-5.6-terra'),
        hasLuna: Array.isArray(cfg.available_models) && cfg.available_models.includes('gpt-5.6-luna'),
        firstModels: Array.isArray(cfg.available_models) ? cfg.available_models.slice(0, 8) : null,
      });
    }
    await db.close();
  }

  const badEntries = summaries.filter((entry) =>
    entry.default_model !== 'gpt-5.6-sol' ||
    entry.use_hidden_models !== false ||
    !entry.hasSol ||
    !entry.hasTerra ||
    !entry.hasLuna ||
    entry.innerTime < futureMs
  );

  const newestByLabel = {};
  for (const entry of summaries) {
    if (!newestByLabel[entry.label] || entry.receivedAt > newestByLabel[entry.label].receivedAt) {
      newestByLabel[entry.label] = entry;
    }
  }

  const result = {
    futureMs,
    futureIso: new Date(futureMs).toISOString(),
    report,
    checkedEntries: summaries.length,
    badEntries: badEntries.length,
    newestByLabel,
  };

  console.log(JSON.stringify(result, null, 2));

  const totalModified = report.reduce((sum, entry) => sum + entry.modifiedStatsigEntries, 0);
  if (totalModified === 0) {
    console.error('No Statsig dynamic config 107580212 entries were modified.');
    process.exit(10);
  }
  if (badEntries.length > 0) {
    console.error('Verification found entries that were not patched correctly.');
    process.exit(11);
  }
})().catch((error) => {
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
});
NODE
}

main() {
  parse_args "$@"

  if [[ "$SELF_TEST" == true ]]; then
    run_self_test
    return 0
  fi

  mkdir -p "$(dirname "$LOG_FILE")"
  exec > >(tee -a "$LOG_FILE") 2>&1

  local mode="dry-run"
  [[ "$APPLY" == true ]] && mode="apply"
  log "Starting Codex Desktop model cache fix in $mode mode."
  log "Log file: $LOG_FILE"

  [[ -d "$APP_PATH" ]] || die "ChatGPT.app not found at: $APP_PATH"
  [[ -d "$CODEX_SUPPORT_DIR" ]] || die "Codex support directory not found: $CODEX_SUPPORT_DIR"

  local specs=()
  while IFS= read -r spec; do
    specs+=("$spec")
  done < <(db_specs)

  local pkg_dir work_dir
  TEMP_ROOT="$(mktemp -d /private/tmp/codex-model-cache-fix.XXXXXX)"
  pkg_dir="$TEMP_ROOT/npm"
  work_dir="$TEMP_ROOT/work"
  mkdir -p "$pkg_dir" "$work_dir"

  trap cleanup_temp EXIT

  install_leveldb_package "$pkg_dir"

  if [[ "$APPLY" != true ]]; then
    log "Dry-run copies will be patched under: $work_dir"
    local dry_specs=()
    while IFS= read -r spec; do
      dry_specs+=("$spec")
    done < <(copy_db_specs "$work_dir" "${specs[@]}")
    patch_and_verify "$pkg_dir" "${dry_specs[@]}"
    log "Dry-run succeeded. No real Codex App data was changed."
    return 0
  fi

  confirm_apply
  release_locks_for_apply

  local backup_dir="$CODEX_SUPPORT_DIR/codex-model-cache-backups/$(date +%Y%m%d-%H%M%S)"
  backup_db_specs "$backup_dir" "${specs[@]}"

  patch_and_verify "$pkg_dir" "${specs[@]}"
  log "Patched and verified real Codex Desktop App model cache."
  log "Backup directory: $backup_dir"

  if [[ "$RESTART_APP" == true ]]; then
    log "Reopening ChatGPT.app."
    open "$APP_PATH"
  else
    log "Skipped restart because --no-restart was set."
  fi
}

main "$@"

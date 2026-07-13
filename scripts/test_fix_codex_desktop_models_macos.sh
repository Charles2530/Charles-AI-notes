#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
target_script="$script_dir/fix_codex_desktop_models_macos.sh"

if [[ ! -x "$target_script" ]]; then
  echo "Missing executable script: $target_script" >&2
  exit 1
fi

"$target_script" --self-test

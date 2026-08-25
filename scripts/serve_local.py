"""Run MkDocs together with the local-only AI reading assistant proxy."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from scripts.ai_reading_assistant import (
        AI_PROXY_HOST,
        AI_PROXY_PORT,
        create_proxy_server,
        load_config,
    )
except ModuleNotFoundError:  # Direct execution via ``python scripts/serve_local.py``.
    from ai_reading_assistant import (
        AI_PROXY_HOST,
        AI_PROXY_PORT,
        create_proxy_server,
        load_config,
    )


def build_mkdocs_command(
    *, mkdocs: str, dev_addr: str, dirty: bool, open_browser: bool
) -> list[str]:
    command = [mkdocs, "serve", "--livereload"]
    if dirty:
        command.append("--dirty")
    if open_browser:
        command.append("--open")
    command.append(f"--dev-addr={dev_addr}")
    return command


def run_services(
    *,
    mkdocs: str,
    dev_addr: str,
    dirty: bool,
    open_browser: bool,
    environ: Mapping[str, str],
    env_path: Path,
    server_factory: Callable[..., Any] = create_proxy_server,
    popen_factory: Callable[[list[str]], Any] = subprocess.Popen,
    logger: Callable[[str], None] = print,
) -> int:
    proxy = None

    try:
        config = load_config(environ=environ, env_path=env_path)
    except (UnicodeError, ValueError) as exc:
        logger(
            f"[atlas-ai] AI 服务暂时不可用：配置读取失败 category={type(exc).__name__}"
        )
    except OSError as exc:
        logger(
            f"[atlas-ai] AI 服务暂时不可用：配置读取失败 category={type(exc).__name__}"
        )
    else:
        try:
            proxy = server_factory(
                config,
                host=AI_PROXY_HOST,
                port=AI_PROXY_PORT,
                logger=logger,
            )
            threading.Thread(target=proxy.serve_forever, daemon=True).start()
            if config.api_key:
                logger(
                    f"[atlas-ai] 本地代理已启动：http://{AI_PROXY_HOST}:{proxy.server_port} "
                    f"model={config.model}"
                )
            else:
                logger("[atlas-ai] AI 服务暂时不可用：未配置 OPENAI_API_KEY")
        except OSError as exc:
            logger(
                f"[atlas-ai] AI 服务暂时不可用：代理启动失败 category={type(exc).__name__}"
            )

    command = build_mkdocs_command(
        mkdocs=mkdocs,
        dev_addr=dev_addr,
        dirty=dirty,
        open_browser=open_browser,
    )
    child = popen_factory(command)
    try:
        return int(child.wait())
    except KeyboardInterrupt:
        if hasattr(child, "send_signal"):
            child.send_signal(signal.SIGINT)
        return int(child.wait())
    finally:
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mkdocs", default="mkdocs")
    parser.add_argument("--dev-addr", default="127.0.0.1:8000")
    parser.add_argument("--dirty", action="store_true")
    parser.add_argument("--open", dest="open_browser", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_services(
        mkdocs=args.mkdocs,
        dev_addr=args.dev_addr,
        dirty=args.dirty,
        open_browser=args.open_browser,
        environ=os.environ,
        env_path=Path.cwd() / ".env.local",
        logger=lambda message: print(message, file=sys.stderr, flush=True),
    )


if __name__ == "__main__":
    raise SystemExit(main())

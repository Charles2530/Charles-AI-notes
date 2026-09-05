"""Local-only AI proxy primitives for the MkDocs reading assistant."""

from __future__ import annotations

import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlsplit


DEFAULT_BASE_URL = "https://47.95.249.76"
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_LONGCAT_BASE_URL = "https://api.longcat.chat/openai/v1"
DEFAULT_LONGCAT_MODEL = "LongCat-2.0"
DEFAULT_LONGCAT_MAX_TOKENS = 1000
DEFAULT_LONGCAT_TEMPERATURE = 0.7
AI_PROXY_HOST = "127.0.0.1"
AI_PROXY_PORT = 8765
GENERIC_CLIENT_ERROR = "AI 服务暂时不可用，请稍后重试。"
MAX_REQUEST_BYTES = 128 * 1024
SYSTEM_INSTRUCTIONS = """你是 Charles AI notes 的中文阅读助手。
优先依据用户提供的文章标题、正文上下文和可选划线原文回答。
如果使用文章之外的通用知识，明确说明这是补充解释，不要伪造原文结论。
回答应准确、清晰，并在有帮助时使用简短列表或代码块。"""


@dataclass(frozen=True)
class AIConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    longcat_api_key: str = ""
    longcat_base_url: str = DEFAULT_LONGCAT_BASE_URL
    longcat_model: str = DEFAULT_LONGCAT_MODEL

    @property
    def responses_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/responses"

    @property
    def longcat_url(self) -> str:
        return f"{self.longcat_base_url.rstrip('/')}/chat/completions"

    @property
    def ready(self) -> bool:
        return bool(self.api_key or self.longcat_api_key)

    @property
    def display_model(self) -> str:
        if self.api_key:
            return self.model
        if self.longcat_api_key:
            return self.longcat_model
        return ""


@dataclass(frozen=True)
class ChatRequest:
    title: str
    path: str
    selection: str
    context: str
    question: str
    history: tuple[dict[str, str], ...]


class ChatValidationError(ValueError):
    pass


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_config(*, environ: Mapping[str, str], env_path: Path) -> AIConfig:
    values = _load_env_file(env_path)
    values.update({key: value for key, value in environ.items() if value})
    return AIConfig(
        api_key=values.get("OPENAI_API_KEY", ""),
        base_url=values.get("ATLAS_AI_BASE_URL", DEFAULT_BASE_URL),
        model=values.get("ATLAS_AI_MODEL", DEFAULT_MODEL),
        reasoning_effort=values.get(
            "ATLAS_AI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT
        ),
        longcat_api_key=values.get("LONGCAT_API_KEY", ""),
        longcat_base_url=values.get("LONGCAT_BASE_URL", DEFAULT_LONGCAT_BASE_URL),
        longcat_model=values.get("LONGCAT_MODEL", DEFAULT_LONGCAT_MODEL),
    )


def is_allowed_origin(origin: str | None) -> bool:
    if not origin:
        return False
    parsed = urlsplit(origin)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname in {"127.0.0.1", "localhost"}
    )


def _required_text(payload: Mapping[str, Any], key: str, max_length: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ChatValidationError(f"{key} must be a non-empty string")
    value = value.strip()
    if len(value) > max_length:
        raise ChatValidationError(f"{key} is too long")
    _validate_utf8(value, key)
    return value


def _optional_text(payload: Mapping[str, Any], key: str, max_length: int) -> str:
    value = payload.get(key, "")
    if not isinstance(value, str):
        raise ChatValidationError(f"{key} must be a string")
    value = value.strip()
    if len(value) > max_length:
        raise ChatValidationError(f"{key} is too long")
    _validate_utf8(value, key)
    return value


def _validate_utf8(value: str, key: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ChatValidationError(f"{key} contains invalid Unicode") from error


def validate_chat_payload(payload: Mapping[str, Any]) -> ChatRequest:
    raw_history = payload.get("history", [])
    if not isinstance(raw_history, list):
        raise ChatValidationError("history must be a list")

    history: list[dict[str, str]] = []
    for item in raw_history[-12:]:
        if not isinstance(item, Mapping):
            raise ChatValidationError("history items must be objects")
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ChatValidationError("history item is invalid")
        if len(content) > 8_000:
            raise ChatValidationError("history item is too long")
        _validate_utf8(content, "history item")
        history.append({"role": role, "content": content.strip()})

    if sum(len(item["content"]) for item in history) > 48_000:
        raise ChatValidationError("history is too long")

    return ChatRequest(
        title=_required_text(payload, "title", 300),
        path=_required_text(payload, "path", 1_000),
        selection=_optional_text(payload, "selection", 4_000),
        context=_required_text(payload, "context", 100_000),
        question=_required_text(payload, "question", 4_000),
        history=tuple(history),
    )


def build_context_question(request: ChatRequest) -> str:
    if request.selection:
        context_lines = [
            "上下文模式：划线引用",
            f"文章标题：{request.title}",
            f"页面路径：{request.path}",
            f"划线原文：{request.selection}",
            f"所在段落：{request.context}",
        ]
    else:
        context_lines = [
            "上下文模式：整篇文章",
            f"文章标题：{request.title}",
            f"页面路径：{request.path}",
            f"文章正文：{request.context}",
        ]
    return "\n".join([*context_lines, f"用户问题：{request.question}"])


def build_upstream_payload(config: AIConfig, request: ChatRequest) -> dict[str, Any]:
    current_question = build_context_question(request)
    return {
        "model": config.model,
        "reasoning": {"effort": config.reasoning_effort},
        "store": False,
        "stream": True,
        "instructions": SYSTEM_INSTRUCTIONS,
        "input": [*request.history, {"role": "user", "content": current_question}],
    }


def build_longcat_payload(config: AIConfig, request: ChatRequest) -> dict[str, Any]:
    current_question = build_context_question(request)
    return {
        "model": config.longcat_model,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            *request.history,
            {"role": "user", "content": current_question},
        ],
        "stream": True,
        "max_tokens": DEFAULT_LONGCAT_MAX_TOKENS,
        "temperature": DEFAULT_LONGCAT_TEMPERATURE,
    }


def iter_browser_events(lines: Any):
    terminal_event_seen = False
    for raw_line in lines:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace").strip()
        else:
            line = str(raw_line).strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            event = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                yield {"type": "delta", "delta": delta}
        elif event_type == "response.completed":
            terminal_event_seen = True
            yield {"type": "done"}
            return
        elif event_type in {"response.failed", "response.incomplete", "error"}:
            terminal_event_seen = True
            yield {"type": "error"}
            return
    if not terminal_event_seen:
        yield {"type": "error"}


def iter_longcat_sse_events(lines: Any):
    terminal_event_seen = False
    for raw_line in lines:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace").strip()
        else:
            line = str(raw_line).strip()
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            terminal_event_seen = True
            yield {"type": "done"}
            return
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if event.get("error"):
            terminal_event_seen = True
            yield {"type": "error"}
            return
        for choice in event.get("choices", []):
            if not isinstance(choice, Mapping):
                continue
            delta = choice.get("delta")
            if isinstance(delta, Mapping):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    yield {"type": "delta", "delta": content}
            if choice.get("finish_reason"):
                terminal_event_seen = True
                yield {"type": "done"}
                return
    if not terminal_event_seen:
        yield {"type": "error"}


def iter_longcat_json_events(payload: Mapping[str, Any]):
    if payload.get("error"):
        yield {"type": "error"}
        return
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        yield {"type": "error"}
        return
    first = choices[0]
    if not isinstance(first, Mapping):
        yield {"type": "error"}
        return
    message = first.get("message")
    if not isinstance(message, Mapping):
        yield {"type": "error"}
        return
    content = message.get("content")
    if not isinstance(content, str) or not content:
        yield {"type": "error"}
        return
    yield {"type": "delta", "delta": content}
    yield {"type": "done"}


class AIProxyHandler(BaseHTTPRequestHandler):
    server_version = "AtlasAIProxy/1.0"

    @property
    def config(self) -> AIConfig:
        return self.server.ai_config  # type: ignore[attr-defined]

    @property
    def logger(self) -> Callable[[str], None]:
        return self.server.ai_logger  # type: ignore[attr-defined]

    def _allowed_origin(self) -> str | None:
        origin = self.headers.get("Origin")
        return origin if is_allowed_origin(origin) else None

    def _send_cors(self, origin: str | None) -> None:
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self._send_cors(self._allowed_origin())
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, event: Mapping[str, Any]) -> None:
        data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _send_model_event(self, model: str) -> None:
        self._send_sse({"type": "model", "model": model})

    def do_OPTIONS(self) -> None:
        origin = self._allowed_origin()
        if self.path != "/chat" or not origin:
            self._send_json(403, {"error": GENERIC_CLIENT_ERROR})
            return
        self.send_response(204)
        self._send_cors(origin)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/health":
            self._send_json(404, {"error": "not found"})
            return
        self._send_json(
            200,
            {
                "ready": self.config.ready,
                "model": self.config.display_model,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/chat":
            self._send_json(404, {"error": "not found"})
            return
        if not self._allowed_origin():
            self._send_json(403, {"error": GENERIC_CLIENT_ERROR})
            return
        if self.headers.get_content_type() != "application/json":
            self._send_json(415, {"error": GENERIC_CLIENT_ERROR})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": GENERIC_CLIENT_ERROR})
            return

        try:
            raw_payload = json.loads(self.rfile.read(content_length))
            if not isinstance(raw_payload, Mapping):
                raise ChatValidationError("payload must be an object")
            chat_request = validate_chat_payload(raw_payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ChatValidationError):
            self._send_json(400, {"error": GENERIC_CLIENT_ERROR})
            return

        if not self.config.ready:
            self.logger("[atlas-ai] chat unavailable category=missing_api_key")
            self._send_json(503, {"error": GENERIC_CLIENT_ERROR})
            return

        if self.config.api_key:
            try:
                upstream = self._open_primary_upstream(chat_request)
            except urllib_error.HTTPError as exc:
                if not self.config.longcat_api_key:
                    self.logger(
                        "[atlas-ai] chat unavailable provider=primary "
                        f"category=upstream_http status={exc.code}"
                    )
                    self._send_json(503, {"error": GENERIC_CLIENT_ERROR})
                    return
                self.logger(
                    "[atlas-ai] primary unavailable category=upstream_http "
                    f"status={exc.code} fallback=longcat"
                )
            except urllib_error.URLError as exc:
                if not self.config.longcat_api_key:
                    self.logger(
                        "[atlas-ai] chat unavailable provider=primary "
                        "category=upstream_network "
                        f"detail={type(exc.reason).__name__}"
                    )
                    self._send_json(503, {"error": GENERIC_CLIENT_ERROR})
                    return
                self.logger(
                    "[atlas-ai] primary unavailable category=upstream_network "
                    f"detail={type(exc.reason).__name__} fallback=longcat"
                )
            except TimeoutError:
                if not self.config.longcat_api_key:
                    self.logger(
                        "[atlas-ai] chat unavailable provider=primary "
                        "category=upstream_timeout"
                    )
                    self._send_json(503, {"error": GENERIC_CLIENT_ERROR})
                    return
                self.logger(
                    "[atlas-ai] primary unavailable category=upstream_timeout "
                    "fallback=longcat"
                )
            else:
                self._stream_primary_response_events(upstream, chat_request)
                return

        if not self.config.longcat_api_key:
            self.logger("[atlas-ai] chat unavailable category=missing_longcat_api_key")
            self._send_json(503, {"error": GENERIC_CLIENT_ERROR})
            return

        try:
            upstream = self._open_longcat_upstream(chat_request)
        except urllib_error.HTTPError as exc:
            self.logger(
                "[atlas-ai] chat unavailable provider=longcat "
                f"category=upstream_http status={exc.code}"
            )
            self._send_json(503, {"error": GENERIC_CLIENT_ERROR})
            return
        except urllib_error.URLError as exc:
            self.logger(
                "[atlas-ai] chat unavailable provider=longcat "
                f"category=upstream_network detail={type(exc.reason).__name__}"
            )
            self._send_json(503, {"error": GENERIC_CLIENT_ERROR})
            return
        except TimeoutError:
            self.logger(
                "[atlas-ai] chat unavailable provider=longcat "
                "category=upstream_timeout"
            )
            self._send_json(503, {"error": GENERIC_CLIENT_ERROR})
            return

        self._stream_longcat_response_events(upstream)

    def _open_primary_upstream(self, chat_request: ChatRequest) -> Any:
        upstream_request = urllib_request.Request(
            self.config.responses_url,
            data=json.dumps(build_upstream_payload(self.config, chat_request)).encode(
                "utf-8"
            ),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        return urllib_request.urlopen(upstream_request, timeout=90)

    def _open_longcat_upstream(self, chat_request: ChatRequest) -> Any:
        upstream_request = urllib_request.Request(
            self.config.longcat_url,
            data=json.dumps(build_longcat_payload(self.config, chat_request)).encode(
                "utf-8"
            ),
            headers={
                "Authorization": f"Bearer {self.config.longcat_api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream, application/json",
            },
            method="POST",
        )
        return urllib_request.urlopen(upstream_request, timeout=90)

    def _stream_primary_response_events(
        self,
        upstream: Any,
        chat_request: ChatRequest,
    ) -> None:
        emitted_text = False
        fallback_started = False
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self._send_cors(self._allowed_origin())
            self.end_headers()
            self._send_model_event(self.config.model)

            with upstream:
                for event in iter_browser_events(upstream):
                    if event.get("type") == "delta":
                        emitted_text = True
                        self._send_sse(event)
                        continue
                    if event.get("type") != "error":
                        self._send_sse(event)
                        continue
                    if emitted_text or not self.config.longcat_api_key:
                        self.logger(
                            "[atlas-ai] stream unavailable "
                            "category=upstream_incomplete"
                        )
                        self._send_sse(event)
                        continue

                    fallback_started = True
                    self._stream_longcat_fallback(chat_request, "primary_stream")
                    return
        except (BrokenPipeError, ConnectionResetError):
            upstream.close()
        except Exception as exc:
            if (
                not emitted_text
                and not fallback_started
                and self.config.longcat_api_key
            ):
                self._stream_longcat_fallback(
                    chat_request,
                    f"primary_stream_exception detail={type(exc).__name__}",
                )
                return
            self.logger(
                f"[atlas-ai] stream unavailable category={type(exc).__name__}"
            )
            try:
                self._send_sse({"type": "error", "error": GENERIC_CLIENT_ERROR})
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _stream_longcat_fallback(
        self,
        chat_request: ChatRequest,
        category: str,
    ) -> None:
        self.logger(
            "[atlas-ai] primary stream unavailable "
            f"category={category} fallback=longcat"
        )
        try:
            fallback = self._open_longcat_upstream(chat_request)
        except urllib_error.HTTPError as exc:
            self.logger(
                "[atlas-ai] chat unavailable provider=longcat "
                f"category=upstream_http status={exc.code}"
            )
            self._send_sse({"type": "error", "error": GENERIC_CLIENT_ERROR})
        except urllib_error.URLError as exc:
            self.logger(
                "[atlas-ai] chat unavailable provider=longcat "
                "category=upstream_network "
                f"detail={type(exc.reason).__name__}"
            )
            self._send_sse({"type": "error", "error": GENERIC_CLIENT_ERROR})
        except TimeoutError:
            self.logger(
                "[atlas-ai] chat unavailable provider=longcat "
                "category=upstream_timeout"
            )
            self._send_sse({"type": "error", "error": GENERIC_CLIENT_ERROR})
        else:
            self._send_model_event(self.config.longcat_model)
            self._send_longcat_events(fallback)

    def _stream_longcat_response_events(self, upstream: Any) -> None:
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self._send_cors(self._allowed_origin())
            self.end_headers()
            self._send_model_event(self.config.longcat_model)
            self._send_longcat_events(upstream)
        except (BrokenPipeError, ConnectionResetError):
            upstream.close()
        except Exception as exc:
            self.logger(
                f"[atlas-ai] stream unavailable provider=longcat "
                f"category={type(exc).__name__}"
            )
            try:
                self._send_sse({"type": "error", "error": GENERIC_CLIENT_ERROR})
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _send_longcat_events(self, upstream: Any) -> None:
        try:
            for event in self._iter_longcat_response_events(upstream):
                if event.get("type") == "error":
                    self.logger(
                        "[atlas-ai] stream unavailable provider=longcat "
                        "category=upstream_incomplete"
                    )
                self._send_sse(event)
        except (BrokenPipeError, ConnectionResetError):
            raise
        except Exception as exc:
            self.logger(
                "[atlas-ai] stream unavailable provider=longcat "
                f"category=stream_exception detail={type(exc).__name__}"
            )
            self._send_sse({"type": "error", "error": GENERIC_CLIENT_ERROR})

    def _iter_longcat_response_events(self, upstream: Any) -> Any:
        content_type = upstream.headers.get("Content-Type", "").lower()
        if "text/event-stream" in content_type:
            with upstream:
                yield from iter_longcat_sse_events(upstream)
            return

        try:
            with upstream:
                payload = json.loads(upstream.read().decode("utf-8"))
                if not isinstance(payload, Mapping):
                    raise json.JSONDecodeError("payload must be object", "", 0)
        except (UnicodeDecodeError, json.JSONDecodeError):
            upstream.close()
            self.logger(
                "[atlas-ai] stream unavailable provider=longcat "
                "category=invalid_json"
            )
            yield {"type": "error"}
            return

        yield from iter_longcat_json_events(payload)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def create_proxy_server(
    config: AIConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    logger: Callable[[str], None] = print,
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), AIProxyHandler)
    server.daemon_threads = True
    server.ai_config = config  # type: ignore[attr-defined]
    server.ai_logger = logger  # type: ignore[attr-defined]
    return server

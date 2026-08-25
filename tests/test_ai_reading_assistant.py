import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase


class AIReadingAssistantConfigTests(TestCase):
    def test_system_environment_overrides_private_env_file(self):
        from scripts.ai_reading_assistant import load_config

        with TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env.local"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=file-key",
                        "ATLAS_AI_MODEL=file-model",
                        "ATLAS_AI_REASONING_EFFORT=medium",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(
                environ={
                    "OPENAI_API_KEY": "system-key",
                    "ATLAS_AI_MODEL": "gpt-5.6-sol",
                    "ATLAS_AI_REASONING_EFFORT": "high",
                },
                env_path=env_path,
            )

        self.assertEqual(config.api_key, "system-key")
        self.assertEqual(config.model, "gpt-5.6-sol")
        self.assertEqual(config.reasoning_effort, "high")
        self.assertEqual(config.base_url, "https://47.95.249.76")
        self.assertEqual(config.responses_url, "https://47.95.249.76/responses")


class AIReadingAssistantRequestTests(TestCase):
    def test_only_loopback_blog_origins_are_allowed(self):
        from scripts.ai_reading_assistant import is_allowed_origin

        self.assertTrue(is_allowed_origin("http://127.0.0.1:8000"))
        self.assertTrue(is_allowed_origin("http://localhost:8010"))
        self.assertFalse(is_allowed_origin("https://example.com"))
        self.assertFalse(is_allowed_origin("http://127.0.0.1.evil.example"))
        self.assertFalse(is_allowed_origin("null"))

    def test_builds_verified_responses_api_payload(self):
        from scripts.ai_reading_assistant import (
            AIConfig,
            build_upstream_payload,
            validate_chat_payload,
        )

        request = validate_chat_payload(
            {
                "title": "KV Cache 与长上下文",
                "path": "/inference/kv-cache/",
                "selection": "容量随序列长度线性增长。",
                "context": "KV Cache 的容量随序列长度、层数与 KV head 数近似线性增长。",
                "question": "为什么是线性增长？",
                "history": [
                    {"role": "user", "content": "先解释一下 KV Cache。"},
                    {"role": "assistant", "content": "它缓存每层的 K/V 张量。"},
                ],
            }
        )

        payload = build_upstream_payload(
            AIConfig(
                api_key="test-key",
                base_url="https://47.95.249.76",
                model="gpt-5.6-sol",
                reasoning_effort="high",
            ),
            request,
        )

        self.assertEqual(payload["model"], "gpt-5.6-sol")
        self.assertEqual(payload["reasoning"], {"effort": "high"})
        self.assertIs(payload["store"], False)
        self.assertIs(payload["stream"], True)
        self.assertNotIn("service_tier", payload)
        self.assertEqual(payload["input"][0], request.history[0])
        current_question = payload["input"][-1]["content"]
        self.assertIn("上下文模式：划线引用", current_question)
        self.assertIn("KV Cache 与长上下文", current_question)
        self.assertIn("容量随序列长度线性增长。", current_question)
        self.assertIn("为什么是线性增长？", current_question)

    def test_accepts_whole_article_context_without_a_selection(self):
        from scripts.ai_reading_assistant import (
            AIConfig,
            build_upstream_payload,
            validate_chat_payload,
        )

        request = validate_chat_payload(
            {
                "title": "整篇文章",
                "path": "/article/",
                "selection": "",
                "context": "第一节\n第二节解释全文上下文。",
                "question": "这篇文章的主线是什么？",
                "history": [],
            }
        )
        payload = build_upstream_payload(AIConfig(api_key="test-key"), request)
        current_question = payload["input"][-1]["content"]

        self.assertEqual(request.selection, "")
        self.assertIn("上下文模式：整篇文章", current_question)
        self.assertIn("文章正文：第一节\n第二节解释全文上下文。", current_question)
        self.assertNotIn("划线原文：", current_question)

    def test_whole_article_context_is_required_and_bounded(self):
        from scripts.ai_reading_assistant import (
            ChatValidationError,
            validate_chat_payload,
        )

        base = {
            "title": "Article",
            "path": "/article/",
            "selection": "",
            "question": "question",
            "history": [],
        }
        with self.assertRaises(ChatValidationError):
            validate_chat_payload({**base, "context": ""})
        with self.assertRaises(ChatValidationError):
            validate_chat_payload({**base, "context": "文" * 100_001})

    def test_rejects_isolated_unicode_surrogates_before_upstream_encoding(self):
        from scripts.ai_reading_assistant import (
            ChatValidationError,
            validate_chat_payload,
        )

        payload = {
            "title": "Article",
            "path": "/article/",
            "selection": "\ud83d",
            "context": "context",
            "question": "question",
            "history": [],
        }
        with self.assertRaises(ChatValidationError):
            validate_chat_payload(payload)
        with self.assertRaises(ChatValidationError):
            validate_chat_payload(
                {
                    **payload,
                    "selection": "",
                    "history": [{"role": "assistant", "content": "\ud83d"}],
                }
            )

    def test_rejects_oversized_user_content(self):
        from scripts.ai_reading_assistant import (
            ChatValidationError,
            validate_chat_payload,
        )

        payload = {
            "title": "Article",
            "path": "/article/",
            "selection": "selected",
            "context": "context",
            "question": "q" * 4001,
            "history": [],
        }

        with self.assertRaises(ChatValidationError):
            validate_chat_payload(payload)


class AIReadingAssistantStreamingTests(TestCase):
    def test_translates_upstream_sse_to_safe_browser_events(self):
        from scripts.ai_reading_assistant import iter_browser_events

        upstream_lines = [
            b'data: {"type":"response.created"}\n',
            b'data: {"type":"response.output_text.delta","delta":"API"}\n',
            b'data: {"type":"response.output_text.delta","delta":"_OK"}\n',
            b'data: {"type":"response.completed"}\n',
            b"data: [DONE]\n",
        ]

        events = list(iter_browser_events(upstream_lines))

        self.assertEqual(
            events,
            [
                {"type": "delta", "delta": "API"},
                {"type": "delta", "delta": "_OK"},
                {"type": "done"},
            ],
        )

    def test_marks_failed_and_incomplete_upstream_streams_as_errors(self):
        from scripts.ai_reading_assistant import iter_browser_events

        failed = list(
            iter_browser_events(
                [b'data: {"type":"response.failed","response":{}}\n']
            )
        )
        truncated = list(
            iter_browser_events(
                [b'data: {"type":"response.output_text.delta","delta":"partial"}\n']
            )
        )

        self.assertEqual(failed, [{"type": "error"}])
        self.assertEqual(
            truncated,
            [{"type": "delta", "delta": "partial"}, {"type": "error"}],
        )

    def test_local_proxy_streams_text_from_a_responses_gateway(self):
        from scripts.ai_reading_assistant import AIConfig, create_proxy_server

        class GatewayHandler(BaseHTTPRequestHandler):
            request_payload = None

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                type(self).request_payload = json.loads(self.rfile.read(length))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for event in (
                    {"type": "response.output_text.delta", "delta": "API"},
                    {"type": "response.output_text.delta", "delta": "_OK"},
                    {"type": "response.completed"},
                ):
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()

            def log_message(self, _format, *_args):
                return

        gateway = ThreadingHTTPServer(("127.0.0.1", 0), GatewayHandler)
        gateway_thread = threading.Thread(target=gateway.serve_forever, daemon=True)
        gateway_thread.start()

        logs = []
        proxy = create_proxy_server(
            AIConfig(
                api_key="test-key",
                base_url=f"http://127.0.0.1:{gateway.server_port}",
            ),
            host="127.0.0.1",
            port=0,
            logger=logs.append,
        )
        proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        proxy_thread.start()

        payload = {
            "title": "Article",
            "path": "/article/",
            "selection": "selected",
            "context": "paragraph",
            "question": "question",
            "history": [],
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:{proxy.server_port}/chat",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Origin": "http://127.0.0.1:8000",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                browser_events = [
                    json.loads(line[6:])
                    for raw_line in response
                    if (line := raw_line.decode().strip()).startswith("data: ")
                ]
        finally:
            proxy.shutdown()
            proxy.server_close()
            gateway.shutdown()
            gateway.server_close()

        self.assertEqual(
            browser_events,
            [
                {"type": "delta", "delta": "API"},
                {"type": "delta", "delta": "_OK"},
                {"type": "done"},
            ],
        )
        self.assertEqual(GatewayHandler.request_payload["model"], "gpt-5.6-sol")
        self.assertIs(GatewayHandler.request_payload["store"], False)
        self.assertEqual(logs, [])


class AIReadingAssistantServeTests(TestCase):
    def test_builds_the_existing_mkdocs_serve_command(self):
        from scripts.serve_local import build_mkdocs_command

        self.assertEqual(
            build_mkdocs_command(
                mkdocs="./docs/bin/mkdocs",
                dev_addr="127.0.0.1:8010",
                dirty=True,
                open_browser=True,
            ),
            [
                "./docs/bin/mkdocs",
                "serve",
                "--livereload",
                "--dirty",
                "--open",
                "--dev-addr=127.0.0.1:8010",
            ],
        )

    def test_proxy_start_failure_does_not_block_mkdocs(self):
        from scripts.serve_local import run_services

        commands = []
        logs = []

        class FinishedProcess:
            def wait(self):
                return 0

        def unavailable_proxy(*_args, **_kwargs):
            raise OSError("port in use")

        def start_mkdocs(command):
            commands.append(command)
            return FinishedProcess()

        exit_code = run_services(
            mkdocs="mkdocs",
            dev_addr="127.0.0.1:8000",
            dirty=False,
            open_browser=False,
            environ={},
            env_path=Path("/nonexistent/.env.local"),
            server_factory=unavailable_proxy,
            popen_factory=start_mkdocs,
            logger=logs.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(commands[0][0:3], ["mkdocs", "serve", "--livereload"])
        self.assertTrue(any("AI 服务暂时不可用" in message for message in logs))

    def test_invalid_private_config_does_not_block_mkdocs(self):
        from scripts.serve_local import run_services

        commands = []
        logs = []

        class FinishedProcess:
            def wait(self):
                return 0

        with TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env.local"
            env_path.write_bytes(b"OPENAI_API_KEY=\xff")
            exit_code = run_services(
                mkdocs="mkdocs",
                dev_addr="127.0.0.1:8000",
                dirty=False,
                open_browser=False,
                environ={},
                env_path=env_path,
                popen_factory=lambda command: (
                    commands.append(command) or FinishedProcess()
                ),
                logger=logs.append,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(commands[0][0:3], ["mkdocs", "serve", "--livereload"])
        self.assertTrue(any("AI 服务暂时不可用" in message for message in logs))

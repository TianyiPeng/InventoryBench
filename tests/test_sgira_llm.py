from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sgira.llm import OpenAICompatibleLLM, _json_object


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **request):
        self.calls += 1
        self.request = request
        return SimpleNamespace(
            id="response-1",
            model=request["model"],
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content='```json\n{"route":"OR","note":"a } b"}\n```'),
            )],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
        )


class OpenAICompatibleLLMTests(unittest.TestCase):
    def test_extracts_fenced_balanced_object(self) -> None:
        self.assertEqual(
            _json_object('prefix {"value":"escaped \\\" }", "ok":true} suffix'),
            {"value": 'escaped " }', "ok": True},
        )

    def test_rejects_non_json_response(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not contain"):
            _json_object("plain text")

    def test_calls_gateway_then_uses_disk_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict("os.environ", {"MA_LLM_TRUST_ENV": "false"}):
                llm = OpenAICompatibleLLM(
                    model="company-gemini",
                    base_url="https://gateway.example/v1",
                    api_key="test-key",
                    max_retries=1,
                    cache_dir=root / "cache",
                    audit_path=root / "audit.jsonl",
                )
            completions = FakeCompletions()
            llm.client = SimpleNamespace(
                chat=SimpleNamespace(completions=completions)
            )
            request = {
                "role": "controller",
                "system_prompt": "Return JSON.",
                "payload": {"period": 1},
            }
            expected = {"route": "OR", "note": "a } b"}
            self.assertEqual(llm.complete_json(**request), expected)
            self.assertEqual(llm.complete_json(**request), expected)
            self.assertEqual(completions.calls, 1)
            audit = json.loads((root / "audit.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(audit["requested_model"], "company-gemini")
            self.assertNotIn("test-key", json.dumps(audit))


if __name__ == "__main__":
    unittest.main()

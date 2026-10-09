import asyncio
import json

import pytest

from app.services.ollama_service import OllamaService, OllamaServiceError


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(
            {"message": {"content": "Cơn đau bắt đầu từ khi nào?"}}
        ).encode("utf-8")


def test_ollama_rephrases_using_state_and_recent_conversation(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(
        "app.services.ollama_service.urlopen",
        fake_urlopen,
    )
    service = OllamaService(
        base_url="http://localhost:11434/",
        model="qwen2.5:3b",
    )

    question = asyncio.run(
        service.rephrase_question(
            question="Bạn bị đau đầu từ khi nào?",
            symptoms={"headache": "có"},
            history=[
                {"role": "user", "content": "Tôi bị đau đầu."},
                {"role": "assistant", "content": "Bạn đau ở đâu?"},
            ],
        )
    )

    assert question == "Cơn đau bắt đầu từ khi nào?"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["timeout"] == 45.0
    payload = captured["payload"]
    assert payload["model"] == "qwen2.5:3b"
    context = json.loads(payload["messages"][1]["content"])
    assert context["known_symptoms"] == {"headache": "có"}
    assert len(context["recent_conversation"]) == 2


def test_ollama_rejects_response_that_is_not_one_question(monkeypatch):
    class InvalidResponse(FakeResponse):
        def read(self):
            return json.dumps(
                {"message": {"content": "Bạn bị đau đầu. Tôi có thể giúp gì?"}}
            ).encode("utf-8")

    monkeypatch.setattr(
        "app.services.ollama_service.urlopen",
        lambda *_args, **_kwargs: InvalidResponse(),
    )

    with pytest.raises(OllamaServiceError, match="exactly one short question"):
        OllamaService()._request_question({})

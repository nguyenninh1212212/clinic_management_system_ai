import asyncio
import time
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.controller import chat_router
from app.main import app
from app.schemas.chat_schema import ChatRequest
from app.services.ollama_service import OllamaServiceError


class FakeEngine:
    def __init__(self):
        self.symptom_pipeline = SimpleNamespace(
            patient_state=SimpleNamespace(
                get_state=lambda: SimpleNamespace(
                    symptoms={
                        "headache": SimpleNamespace(value=True),
                    }
                )
            )
        )

    def process_message(self, _message):
        return {
            "status": "asking_question",
            "symptoms": ["headache"],
            "finished": False,
            "department_triage": {
                "suggested_department": "Thần kinh",
                "score": 0.72,
                "departments": [
                    {
                        "department": "Thần kinh",
                        "score": 0.72,
                        "diseases": ["example_disease"],
                    }
                ],
                "status": "suggested",
            },
            "next_question": {
                "question": "Bạn bị đau đầu từ khi nào?",
            },
        }


def test_chat_uses_ollama_and_reports_template_fallback(monkeypatch):
    conversation = chat_router.Conversation(engine=FakeEngine())
    monkeypatch.setattr(
        chat_router,
        "_get_or_create_conversation",
        lambda _conversation_id: conversation,
    )
    requests = []

    async def fake_rephrase_question(**kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            return "Cơn đau đầu bắt đầu từ khi nào?"
        raise OllamaServiceError("Ollama is offline")

    monkeypatch.setattr(
        chat_router._ollama,
        "rephrase_question",
        fake_rephrase_question,
    )

    async def run_conversation():
        first = await chat_router.chat(
            ChatRequest(message="Tôi bị đau đầu")
        )
        second = await chat_router.chat(
            ChatRequest(
                conversation_id=first.conversation_id,
                message="Bắt đầu từ sáng nay",
            )
        )
        return first, second

    first, second = asyncio.run(run_conversation())

    assert first.message == "Cơn đau đầu bắt đầu từ khi nào?"
    assert first.llm_used is True
    assert first.warning is None
    assert first.department_triage is None
    assert second.message == "Bạn bị đau đầu từ khi nào?"
    assert second.llm_used is False
    assert "Ollama is offline" in second.warning
    assert requests[1]["history"][0] == {
        "role": "user",
        "content": "Tôi bị đau đầu",
    }


def test_emergency_response_does_not_call_ollama(monkeypatch):
    conversation = chat_router.Conversation(engine=FakeEngine())
    monkeypatch.setattr(
        chat_router,
        "_get_or_create_conversation",
        lambda _conversation_id: conversation,
    )

    async def unexpected_rephrase(**_kwargs):
        raise AssertionError("Emergency responses must bypass Ollama")

    monkeypatch.setattr(
        chat_router._ollama,
        "rephrase_question",
        unexpected_rephrase,
    )
    conversation.engine.process_message = lambda _message: {
        "status": "emergency",
        "symptoms": ["chest_pain"],
        "finished": False,
        "message": "Vui lòng gọi cấp cứu ngay.",
        "next_question": None,
        "department_triage": {
            "suggested_department": "Thần kinh",
            "score": 0.7,
            "departments": [],
            "status": "suggested",
        },
    }

    response = asyncio.run(
        chat_router.chat(
            ChatRequest(message="Tôi đau ngực dữ dội")
        )
    )

    assert response.status == "emergency"
    assert response.message == "Vui lòng gọi cấp cứu ngay."
    assert response.symptoms == ["chest_pain"]
    assert response.llm_used is False
    assert response.department_triage is None


def test_chat_returns_department_after_question_limit(monkeypatch):
    conversation = chat_router.Conversation(engine=FakeEngine())
    monkeypatch.setattr(
        chat_router,
        "_get_or_create_conversation",
        lambda _conversation_id: conversation,
    )
    conversation.engine.process_message = lambda _message: {
        "status": "question_limit_reached",
        "symptoms": ["headache", "fever"],
        "finished": False,
        "message": "Đã đạt giới hạn câu hỏi tự động.",
        "next_question": None,
        "department_triage": {
            "suggested_department": "Thần kinh",
            "score": 0.72,
            "departments": [
                {
                    "department": "Thần kinh",
                    "score": 0.72,
                    "diseases": ["example_disease"],
                }
            ],
            "status": "suggested",
        },
    }

    response = asyncio.run(
        chat_router.chat(
            ChatRequest(message="Tôi vẫn còn triệu chứng")
        )
    )

    assert response.status == "question_limit_reached"
    assert response.department_triage.suggested_department == "Thần kinh"


def test_websocket_keeps_conversation_id_across_messages(monkeypatch):
    conversation = chat_router.Conversation(engine=FakeEngine())
    created_ids = []

    def get_or_create_conversation(conversation_id):
        created_ids.append(conversation_id)
        return conversation

    monkeypatch.setattr(
        chat_router,
        "_get_or_create_conversation",
        get_or_create_conversation,
    )

    async def fake_rephrase_question(**_kwargs):
        raise OllamaServiceError("Ollama is offline")

    monkeypatch.setattr(
        chat_router._ollama,
        "rephrase_question",
        fake_rephrase_question,
    )

    with TestClient(app).websocket_connect("/api/v1/chat/ws") as websocket:
        websocket.send_json({"message": "Tôi bị đau đầu"})
        first = websocket.receive_json()
        websocket.send_json({"message": "Bắt đầu từ sáng nay"})
        second = websocket.receive_json()

    assert first["conversation_id"]
    assert second["conversation_id"] == first["conversation_id"]
    assert created_ids == [first["conversation_id"]] * 2
    assert first["status"] == "asking_question"
    assert first["llm_used"] is False


def test_websocket_reports_invalid_payload_without_closing(monkeypatch):
    with TestClient(app).websocket_connect("/api/v1/chat/ws") as websocket:
        websocket.send_text("not-json")
        invalid_json_response = websocket.receive_json()
        websocket.send_json({"message": ""})
        invalid_payload_response = websocket.receive_json()

    assert invalid_json_response["status"] == "error"
    assert "valid JSON" in invalid_json_response["message"]
    assert invalid_payload_response["status"] == "error"
    assert invalid_payload_response["message"] == "Invalid chat message."


def test_expired_conversation_is_replaced(monkeypatch):
    conversations = {}
    monkeypatch.setattr(chat_router, "_conversations", conversations)
    monkeypatch.setattr(
        chat_router,
        "ClinicalQuestionService",
        lambda **_kwargs: FakeEngine(),
    )
    monkeypatch.setattr(chat_router, "CONVERSATION_TTL_SECONDS", 30)
    conversation_id = str(uuid.uuid4())
    expired = chat_router.Conversation(engine=FakeEngine())
    expired.last_activity = time.monotonic() - 31
    conversations[conversation_id] = expired

    replacement = chat_router._get_or_create_conversation(conversation_id)

    assert replacement is not expired
    assert conversation_id in conversations


def test_conversation_capacity_is_bounded(monkeypatch):
    conversations = {"active": chat_router.Conversation(engine=FakeEngine())}
    monkeypatch.setattr(chat_router, "_conversations", conversations)
    monkeypatch.setattr(chat_router, "MAX_ACTIVE_CONVERSATIONS", 1)

    with pytest.raises(RuntimeError, match="giới hạn phiên hội thoại"):
        chat_router._get_or_create_conversation(str(uuid.uuid4()))

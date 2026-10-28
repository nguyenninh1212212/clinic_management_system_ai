import asyncio
import time
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.controller import chat_router
from app.main import app
from app.schemas.chat_schema import ChatRequest


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


def test_chat_uses_template_questions(monkeypatch):
    conversation = chat_router.Conversation(engine=FakeEngine())
    monkeypatch.setattr(
        chat_router,
        "_get_or_create_conversation",
        lambda _conversation_id: conversation,
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

    assert first.message == "Bạn bị đau đầu từ khi nào?"
    assert first.department_triage is None
    assert second.message == "Bạn bị đau đầu từ khi nào?"


def test_unclear_symptom_scope_uses_template(monkeypatch):
    conversation = chat_router.Conversation(engine=FakeEngine())
    monkeypatch.setattr(
        chat_router,
        "_get_or_create_conversation",
        lambda _conversation_id: conversation,
    )
    expected_message = (
        "Mình hiện chỉ hỗ trợ ghi nhận triệu chứng, "
        "chưa trả lời câu hỏi kiến thức chung."
    )
    conversation.engine.process_message = lambda _message: {
        "status": "needs_clarification",
        "symptoms": [],
        "finished": False,
        "next_question": {
            "question_type": "unclear_symptom",
            "question": expected_message,
        },
    }

    response = asyncio.run(
        chat_router.chat(
            ChatRequest(message="Cảm cúm là gì?")
        )
    )

    assert response.message == expected_message


def test_emergency_response_uses_engine_message(monkeypatch):
    conversation = chat_router.Conversation(engine=FakeEngine())
    monkeypatch.setattr(
        chat_router,
        "_get_or_create_conversation",
        lambda _conversation_id: conversation,
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
    assert response.possible_diseases == []
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

    with TestClient(app).websocket_connect("/api/v1/chat/ws") as websocket:
        websocket.send_json({"message": "Tôi bị đau đầu"})
        first = websocket.receive_json()
        websocket.send_json({"message": "Bắt đầu từ sáng nay"})
        second = websocket.receive_json()

    assert first["conversation_id"]
    assert second["conversation_id"] == first["conversation_id"]
    assert created_ids == [first["conversation_id"]] * 2
    assert first["status"] == "asking_question"


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


def test_possible_diseases_returns_five_ranked_candidates():
    result = {
        "ranked_diseases": [
            {
                "disease": f"disease_{index}",
                "score": 1 - index / 10,
                "matched_symptoms": ["headache"],
            }
            for index in range(6)
        ]
    }

    diseases = chat_router._possible_diseases(result)

    assert len(diseases) == 5
    assert diseases[0] == {
        "disease": "disease_0",
        "score": 1.0,
        "matched_symptoms": ["headache"],
    }


def test_conversation_capacity_is_bounded(monkeypatch):
    conversations = {"active": chat_router.Conversation(engine=FakeEngine())}
    monkeypatch.setattr(chat_router, "_conversations", conversations)
    monkeypatch.setattr(chat_router, "MAX_ACTIVE_CONVERSATIONS", 1)

    with pytest.raises(RuntimeError, match="giới hạn phiên hội thoại"):
        chat_router._get_or_create_conversation(str(uuid.uuid4()))

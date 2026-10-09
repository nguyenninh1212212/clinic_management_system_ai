import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from AI.NER.inference import NERInference
from AI.question.question_service import QuestionService as ClinicalQuestionService
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.services.ollama_service import OllamaService, OllamaServiceError


router = APIRouter()
BASE_DIR = Path(__file__).resolve().parents[2]
CONVERSATION_TTL_SECONDS = 30 * 60
MAX_ACTIVE_CONVERSATIONS = 500
MODEL_PATH = BASE_DIR / "models" / "symptom_ner_model" / "final"
KNOWLEDGE_BASE_PATH = (
    BASE_DIR
    / "training"
    / "dataset"
    / "disease"
    / "disease_knowledge_base.csv"
)


@lru_cache(maxsize=1)
def _get_ner() -> NERInference:
    return NERInference(str(MODEL_PATH))


@dataclass
class Conversation:
    engine: ClinicalQuestionService
    history: list[dict[str, str]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_activity: float = field(default_factory=time.monotonic)


_conversations: dict[str, Conversation] = {}
_conversations_lock = threading.Lock()
_engine_lock = asyncio.Lock()
_ollama = OllamaService()


class ConversationCapacityError(RuntimeError):
    pass


def _get_or_create_conversation(conversation_id: str) -> Conversation:
    with _conversations_lock:
        now = time.monotonic()
        expired_ids = [
            session_id
            for session_id, item in _conversations.items()
            if now - item.last_activity >= CONVERSATION_TTL_SECONDS
        ]
        for session_id in expired_ids:
            _conversations.pop(session_id, None)

        conversation = _conversations.get(conversation_id)
        if conversation is None:
            if len(_conversations) >= MAX_ACTIVE_CONVERSATIONS:
                raise ConversationCapacityError(
                    "Máy chủ đang đạt giới hạn phiên hội thoại hoạt động. "
                    "Vui lòng thử lại sau."
                )
            conversation = Conversation(
                engine=ClinicalQuestionService(
                    model_path=str(MODEL_PATH),
                    knowledge_base_path=str(KNOWLEDGE_BASE_PATH),
                    ner=_get_ner(),
                )
            )
            _conversations[conversation_id] = conversation
        conversation.last_activity = now
        return conversation


def _known_symptoms(engine: ClinicalQuestionService) -> dict[str, str]:
    state = engine.symptom_pipeline.patient_state.get_state()
    return {
        code: (
            "có" if symptom.value is True
            else "không" if symptom.value is False
            else "chưa rõ"
        )
        for code, symptom in state.symptoms.items()
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    return await _process_chat(request)


@router.websocket("/chat/ws")
async def chat_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    conversation_id = None

    while True:
        try:
            raw_message = await websocket.receive_text()
        except WebSocketDisconnect:
            return
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            await websocket.send_json({
                "status": "error",
                "message": "Each WebSocket message must contain valid JSON.",
            })
            continue

        try:
            request = ChatRequest.model_validate(payload)
        except ValidationError as error:
            await websocket.send_json({
                "status": "error",
                "message": "Invalid chat message.",
                "details": error.errors(include_input=False),
            })
            continue

        if request.conversation_id is None:
            request = request.model_copy(
                update={"conversation_id": conversation_id}
            )

        try:
            response = await _process_chat(request)
        except HTTPException as error:
            await websocket.send_json({
                "status": "error",
                "message": str(error.detail),
                "code": error.status_code,
            })
            continue

        conversation_id = response.conversation_id
        await websocket.send_json(response.model_dump(mode="json"))


async def _process_chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(
            status_code=422,
            detail="message must not be blank",
        )

    conversation_id = request.conversation_id or str(uuid.uuid4())
    try:
        conversation = await run_in_threadpool(
            _get_or_create_conversation,
            conversation_id,
        )
    except ConversationCapacityError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error
    except (OSError, ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to initialize the clinical symptom model: {error}",
        ) from error

    async with conversation.lock:
        conversation.last_activity = time.monotonic()
        try:
            async with _engine_lock:
                result = await run_in_threadpool(
                    conversation.engine.process_message,
                    message,
                )
        except (ValueError, RuntimeError) as error:
            raise HTTPException(
                status_code=422,
                detail=str(error),
            ) from error

        next_question = result.get("next_question")
        llm_used = False
        warning = None
        if next_question and next_question.get("question"):
            target_question = next_question["question"]
            try:
                response_message = await _ollama.rephrase_question(
                    question=target_question,
                    symptoms=_known_symptoms(conversation.engine),
                    history=conversation.history,
                )
            except OllamaServiceError as error:
                response_message = target_question
                warning = (
                    f"LLM chưa sẵn sàng ({error}); "
                    "đang dùng câu hỏi mẫu."
                )
            else:
                llm_used = True
        elif result.get("status") == "ended":
            response_message = "Đã kết thúc cuộc trò chuyện."
        elif result.get("message"):
            response_message = result["message"]
        else:
            response_message = (
                "Cảm ơn bạn đã cung cấp thông tin. Đây chỉ là gợi ý "
                "định hướng, không thay thế việc thăm khám và chẩn đoán "
                "của nhân viên y tế."
            )

        conversation.history.extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response_message},
            ]
        )
        conversation.history = conversation.history[-20:]

        return ChatResponse(
            conversation_id=conversation_id,
            status=result["status"],
            message=response_message,
            symptoms=result.get("symptoms", []),
            finished=result.get("finished", False),
            llm_used=llm_used,
            warning=warning,
            department_triage=(
                result.get("department_triage")
                if result["status"] in {
                    "completed",
                    "question_limit_reached",
                    "ended",
                }
                else None
            ),
        )


@router.delete("/chat/{conversation_id}")
async def reset_chat(conversation_id: str) -> dict[str, str]:
    with _conversations_lock:
        _conversations.pop(conversation_id, None)
    return {
        "conversation_id": conversation_id,
        "status": "reset",
    }

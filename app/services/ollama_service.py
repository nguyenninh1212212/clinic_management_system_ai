import asyncio
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaServiceError(RuntimeError):
    pass


class OllamaService:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 45.0,
    ):
        self.base_url = (
            base_url or os.getenv(
                "OLLAMA_BASE_URL",
                "http://127.0.0.1:11434",
            )
        ).rstrip("/")
        self.model = model or os.getenv(
            "OLLAMA_MODEL",
            "qwen2.5:3b",
        )
        self.timeout = timeout

    async def rephrase_question(
        self,
        question: str,
        symptoms: dict[str, Any],
        history: list[dict[str, str]],
    ) -> str:
        context = json.dumps(
            {
                "known_symptoms": symptoms,
                "recent_conversation": history[-10:],
                "question_to_ask": question,
            },
            ensure_ascii=False,
        )
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý thu thập thông tin triệu chứng "
                        "bằng tiếng Việt. Chỉ viết lại câu hỏi được giao "
                        "cho tự nhiên, thân thiện và phù hợp với ngữ cảnh. "
                        "Giữ nguyên mục tiêu câu hỏi, không hỏi lại điều "
                        "đã được trả lời hoặc đã có trong triệu chứng đã "
                        "biết. Chỉ trả về đúng một câu hỏi ngắn. Không "
                        "chẩn đoán bệnh, không đưa lời khuyên điều trị, "
                        "không thêm câu hỏi khác."
                    ),
                },
                {
                    "role": "user",
                    "content": context,
                },
            ],
            "options": {
                "temperature": 0.3,
            },
        }

        return await asyncio.to_thread(
            self._request_question,
            payload,
        )

    def _request_question(
        self,
        payload: dict[str, Any],
    ) -> str:
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise OllamaServiceError(
                f"Ollama returned HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError) as error:
            raise OllamaServiceError(
                "Cannot connect to Ollama. Start Ollama and make sure "
                f"model '{self.model}' is available."
            ) from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise OllamaServiceError(
                "Ollama returned an invalid JSON response."
            ) from error

        if not isinstance(result, dict):
            raise OllamaServiceError(
                "Ollama response must be a JSON object."
            )

        message = result.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise OllamaServiceError(
                "Ollama response did not contain message.content."
            )

        question = content.strip().strip('"“”\' ')
        if (
            not question
            or len(question) > 400
            or not question.endswith("?")
            or question.count("?") != 1
            or ". " in question
            or "! " in question
            or "\n" in question
        ):
            raise OllamaServiceError(
                "Ollama did not return exactly one short question."
            )

        return question

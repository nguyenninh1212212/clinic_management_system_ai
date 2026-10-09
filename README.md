.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8002 --reload

## Local conversational chat with Ollama

The chat endpoint uses the existing symptom/question engine to interpret the
conversation and choose what information to ask next. Ollama only rewrites
that selected question in natural Vietnamese; it does not choose a diagnosis
or override the symptom state.

1. Install Ollama for Windows from <https://ollama.com/download>.
2. In PowerShell, download the default model:

   ```powershell
   ollama pull qwen2.5:3b
   ```

   Ollama must be running locally. To use another installed model, set
   `OLLAMA_MODEL` to its exact Ollama model name. `OLLAMA_BASE_URL` defaults
   to `http://127.0.0.1:11434`.

   For example, in PowerShell:

   ```powershell
   $env:OLLAMA_MODEL = "qwen2.5:3b"
   ```

   If Ollama is not already running, start it with `ollama serve` in a
   separate terminal.
3. Start the API:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8002 --reload
   ```

4. For interactive chat, connect to `ws://127.0.0.1:8002/api/v1/chat/ws`.
   Send one JSON object per user turn:

   ```json
   {
     "message": "Tôi bị đau đầu và tê tay"
   }
   ```

   The service returns the existing chat response as a JSON WebSocket message,
   including a `conversation_id`. The same WebSocket connection remembers that
   ID, so subsequent turns can omit it:

   ```json
   {
     "message": "Tê từ sáng nay"
   }
   ```

   Clients may include a `conversation_id` in a turn to resume a session. The
   REST `POST /api/v1/chat` and `DELETE /api/v1/chat/{conversation_id}` routes
   remain available temporarily for compatibility while the backend migrates.
   Keep the conversation ID private; it identifies the in-memory session.

Chat responses include `department_triage`, with `suggested_department`,
the heuristic `score`, per-department supporting disease candidates, and a
`status` (`suggested`, `ambiguous`, or `insufficient_evidence`). When a
department cannot be reliably selected from reviewed mappings, the suggested
department is `null`. These fields are triage guidance, not a diagnosis.

Department triage is part of the chat flow; there is no separate department
prediction endpoint. While the assistant is collecting symptoms, it asks
follow-up questions. Once collection completes or reaches its question limit,
the same chat response includes `department_triage` with a suggested
department, or explains that evidence is insufficient or ambiguous. Emergency
warning signs interrupt this flow and return the emergency guidance instead.

This setup is for local testing only. It allows anonymous requests and does
not authenticate users. Conversation state is held in server memory, inactive
sessions are evicted after 30 minutes when a later session is created, and at
most 500 active sessions are kept per process. State is lost on restart. Do
not expose this local API to the Internet or use real identifying or sensitive
health information. Public deployment is out of scope for this configuration
and requires a separate security, privacy, and clinical review.

If Ollama is unavailable, the endpoint returns the existing
question-template response and sets `llm_used` to `false` with a warning.
The clinical question engine limits non-final follow-up questions to
`max_questions` (default 8); once reached, it asks for confirmation rather than
continuing symptom or clarification questions. If the user indicates the
information is still incomplete without adding recognized details, the engine
returns `question_limit_reached` and retains the collected state.
Disease ranking scores are heuristic ordering scores, not calibrated
probabilities or a confirmed diagnosis. Known negative symptoms lower matching
disease scores, while unknown symptoms are not treated as positive or negative
evidence.
The department classifier's `confidence` and `top_predictions` are model
outputs for routing only; they are not clinically calibrated probabilities.
This prototype is for local information-gathering tests and triage guidance,
not a diagnosis or emergency-response service. A small backend phrase checker
interrupts routine questions for selected urgent warning signs and displays
an emergency-care message; it cannot recognize every emergency and is not
clinically validated. Department suggestions are withheld when no reviewed
mapping is available or the top department scores are tied.
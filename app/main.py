from fastapi import FastAPI
from app.controller import chat_router

app = FastAPI(title="AI Triage API", version="1.0")

app.include_router(
    chat_router.router,
    prefix="/api/v1",
    tags=["Chat"],
)
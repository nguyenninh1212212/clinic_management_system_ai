from fastapi import FastAPI
from app.controller import triage_router

app = FastAPI(title="AI Triage API", version="1.0")

app.include_router(
    triage_router.router, 
    prefix="/api/v1/triage", 
    tags=["Triage"]
)
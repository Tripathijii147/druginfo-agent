"""FastAPI production service for DrugInfo Agent.

Exposes:
- GET /health: Health check, index status, and available drugs
- POST /ask: Run query through LangGraph agent with citation verification, timing, and step trace
- GET /drugs: List all indexed FDA drugs and aliases
"""
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import rag
from agent import ask, make_graph
from drugs import available_drugs, DRUGS

app = FastAPI(
    title="DrugInfo Agent API",
    description="Agentic RAG assistant over FDA drug labels with LangGraph, citation verification, and safety guardrails.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory query cache for low-latency repeated queries
QUERY_CACHE = {}

# Cached agent graph instance
AGENT_GRAPH = None


def get_agent_graph():
    global AGENT_GRAPH
    if AGENT_GRAPH is None:
        AGENT_GRAPH = make_graph()
    return AGENT_GRAPH


class AskRequest(BaseModel):
    question: str = Field(..., example="Can I take ibuprofen with warfarin?")
    history: List[List[str]] = Field(default=[], description="Chat history as [[role, content], ...]")


class StepDetail(BaseModel):
    tool: str
    args: dict
    output: str


class AskResponse(BaseModel):
    question: str
    answer: str
    steps: List[StepDetail]
    latency_ms: float
    cached: bool


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "DrugInfo Agent",
        "indexed_drugs_count": len(DRUGS),
        "cache_size": len(QUERY_CACHE)
    }


@app.get("/drugs")
def list_drugs():
    return {
        "count": len(DRUGS),
        "drugs": available_drugs()
    }


def get_label_version():
    """Return mtime string of the labels file to ensure cache invalidation on data refresh."""
    import os
    from config import LABELS_PATH
    try:
        return str(int(os.path.getmtime(LABELS_PATH)))
    except Exception:
        return "v1"


def make_cache_key(question: str) -> str:
    return f"{get_label_version()}:{question.lower().strip()}"


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    cache_k = make_cache_key(q)
    # Check cache for identical query without prior conversational history
    if not req.history and cache_k in QUERY_CACHE:
        cached_result = QUERY_CACHE[cache_k]
        return AskResponse(
            question=q,
            answer=cached_result["answer"],
            steps=cached_result["steps"],
            latency_ms=cached_result.get("latency_ms", 0.0),
            cached=True
        )

    t0 = time.perf_counter()
    graph = get_agent_graph()
    # Format history tuples
    hist_tuples = [(item[0], item[1]) for item in req.history if len(item) == 2]

    try:
        final_text, steps = ask(graph, q, history=hist_tuples)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution error: {str(e)}")

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    step_details = [
        StepDetail(
            tool=s.get("tool", ""),
            args=s.get("args", {}),
            output=s.get("output", "")
        )
        for s in steps
    ]

    response_data = {
        "question": q,
        "answer": final_text,
        "steps": step_details,
        "latency_ms": elapsed_ms,
        "cached": False
    }

    if not req.history:
        QUERY_CACHE[cache_k] = response_data

    return AskResponse(**response_data)


@app.post("/cache/clear")
def clear_cache():
    """Clear query cache on data update or retrieval change."""
    global QUERY_CACHE
    size = len(QUERY_CACHE)
    QUERY_CACHE.clear()
    return {"status": "cleared", "previous_size": size}

class MedicationListRequest(BaseModel):
    medications: List[str] = Field(..., example=["warfarin", "aspirin", "metformin", "lisinopril"])


class MedicationListResponse(BaseModel):
    medications_analyzed: List[str]
    total_pairs: int
    report_markdown: str
    latency_ms: float


class FeedbackRequest(BaseModel):
    question: str = Field(..., example="Can I take ibuprofen with warfarin?")
    answer: str = Field(..., example="Warfarin label flags NSAIDs...")
    rating: str = Field(..., example="up", description="'up' for helpful, 'down' for unhelpful/incorrect")
    comments: Optional[str] = Field(default=None, example="Accurately flagged bleeding risk")


class FeedbackResponse(BaseModel):
    status: str
    message: str


@app.post("/check-list", response_model=MedicationListResponse)
def check_list_endpoint(req: MedicationListRequest):
    """Run a multi-drug polypharmacy interaction audit across all unique pairs."""
    if len(req.medications) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 medications to check.")

    t0 = time.perf_counter()
    import tools
    query_str = ", ".join(req.medications)
    report = tools.check_medication_list.invoke({"medications": query_str})
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return MedicationListResponse(
        medications_analyzed=req.medications,
        total_pairs=len(req.medications) * (len(req.medications) - 1) // 2,
        report_markdown=report,
        latency_ms=elapsed_ms
    )


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest):
    """Log user feedback (thumbs up/down) to data/feedback.jsonl for continuous evaluation."""
    import json
    import os
    from datetime import datetime

    if req.rating.lower() not in ("up", "down"):
        raise HTTPException(status_code=400, detail="Rating must be 'up' or 'down'.")

    os.makedirs("data", exist_ok=True)
    feedback_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": req.question,
        "answer": req.answer,
        "rating": req.rating.lower(),
        "comments": req.comments or "",
    }

    try:
        with open(os.path.join("data", "feedback.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_entry) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {e}")

    return FeedbackResponse(status="success", message="Feedback recorded. Thank you!")


@app.get("/brands")
def list_brands():
    """List supported Indian and international brand name mappings."""
    from drugs import ALIASES, BRAND_NOTES
    return {
        "total_brands_mapped": len(ALIASES),
        "brand_mappings": ALIASES,
        "clinical_notes": BRAND_NOTES
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


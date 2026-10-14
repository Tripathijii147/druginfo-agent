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


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Check cache for identical query without prior conversational history
    if not req.history and q.lower() in QUERY_CACHE:
        cached_result = QUERY_CACHE[q.lower()]
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
        QUERY_CACHE[q.lower()] = response_data

    return AskResponse(**response_data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


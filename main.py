import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
import json

from fastapi import Depends, FastAPI, Form, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import StreamingResponse

from config.config_loader import load_config
from graph.rag_graph import build_graph, run_pre_generate, generate_stream
from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader

load_dotenv(override=True)

# ── Rate limiter ──
limiter = Limiter(key_func=get_remote_address)

# ── API key auth ──
_API_KEY = os.environ.get("API_KEY", "")


async def verify_api_key(x_api_key: str = Header(default="")):
    """Require X-API-Key header when API_KEY env var is set."""
    if not _API_KEY:
        return  # auth disabled in dev (no API_KEY set)
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

# ── Globals (initialized once at startup) ──
rag_graph = None
config = load_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the LangGraph pipeline once at startup."""
    global rag_graph
    logging.info("Building RAG graph...")
    retriever = Retriever()
    model_loader = ModelLoader()
    max_retries = config.get("graph", {}).get("max_retries", 2)
    rag_graph = build_graph(retriever, model_loader, max_retries=max_retries)
    logging.info("RAG graph ready")
    yield
    logging.info("Shutting down")


app = FastAPI(
    title="CSSM — Product Assistant API",
    version="2.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Too many requests. Please wait a moment and try again."},
    )


# CORS
allowed_origins = os.environ.get(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:3001,http://10.0.0.52:3000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "CSSM Product Assistant API", "version": "2.0.0"}


@app.post("/retrieve", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def chat(request: Request, msg: str = Form(...), chat_history: str = Form("")):
    """Chat endpoint — runs the LangGraph RAG pipeline."""
    query = msg.strip()

    if not query:
        return JSONResponse(
            status_code=400,
            content={"error": "Query cannot be empty."},
        )
    if len(query) > 1000:
        return JSONResponse(
            status_code=400,
            content={"error": "Query too long. Maximum 1000 characters."},
        )

    try:
        result = rag_graph.invoke({
            "question": query,
            "rewritten_query": "",
            "documents": [],
            "sources": [],
            "grade": "",
            "answer": "",
            "retries": 0,
            "chat_history": chat_history or "No previous conversation.",
        })

        logging.info(f"Query: {query[:80]} | Answer length: {len(result.get('answer', ''))}")

        return JSONResponse(content={
            "response": result.get("answer", ""),
            "sources": result.get("sources", []),
        })

    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Sorry, I'm having trouble processing your request. Please try again.",
            },
        )


@app.post("/stream", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def chat_stream(request: Request, msg: str = Form(...), chat_history: str = Form("")):
    """Streaming chat endpoint — SSE for token-by-token response."""
    query = msg.strip()
    history = chat_history or "No previous conversation."

    if not query:
        return JSONResponse(status_code=400, content={"error": "Query cannot be empty."})
    if len(query) > 1000:
        return JSONResponse(status_code=400, content={"error": "Query too long. Maximum 1000 characters."})

    def event_stream():
        try:
            state = run_pre_generate({
                "question": query,
                "rewritten_query": "",
                "documents": [],
                "sources": [],
                "grade": "",
                "answer": "",
                "retries": 0,
                "chat_history": history,
            })

            # Send sources as first SSE event
            sources = state.get("sources", [])
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            # Stream the generate step token by token
            for token in generate_stream(state):
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            # Signal completion
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logging.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true":
        logging.info("LangSmith tracing enabled (project: %s)", os.environ.get("LANGCHAIN_PROJECT", "default"))
    port = int(os.environ.get("PORT", 8001))
    logging.info("Starting API server at http://0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)

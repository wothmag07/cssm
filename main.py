import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.config_loader import load_config
from graph.rag_graph import build_graph
from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader

load_dotenv(override=True)

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


@app.post("/retrieve")
async def chat(msg: str = Form(...)):
    """Chat endpoint — runs the LangGraph RAG pipeline."""
    query = msg.strip()

    # Input validation
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.info("Starting API server at http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)

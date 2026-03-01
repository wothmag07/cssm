"""
LangGraph RAG Pipeline — Self-correcting retrieval with document grading.

Flow:  retrieve → grade_docs → generate (if relevant)
                             → rewrite → retrieve (if irrelevant, max 2 retries)
"""
import logging
from typing import List, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from prompts.prompt import PROMPT_TEMPLATES
from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader


# ── State definition ──

class RAGState(TypedDict):
    question: str
    rewritten_query: str
    documents: List[Document]
    sources: list  # [{content, metadata, similarity}]
    grade: str     # "relevant" | "irrelevant"
    answer: str
    retries: int


# ── Node functions ──

def retrieve(state: RAGState) -> dict:
    """Retrieve documents from Supabase vector store."""
    query = state.get("rewritten_query") or state["question"]
    logging.info(f"[retrieve] query: {query[:80]}")

    results = _retriever_instance.retrieve(query)

    documents = []
    sources = []
    for doc, score in results:
        documents.append(doc)
        sources.append({
            "content": doc.page_content[:200],
            "metadata": doc.metadata,
            "similarity": round(float(score), 3),
        })

    logging.info(f"[retrieve] found {len(documents)} documents")
    return {"documents": documents, "sources": sources}


def grade_docs(state: RAGState) -> dict:
    """Grade whether retrieved documents are relevant to the question."""
    docs = state.get("documents", [])
    if not docs:
        logging.info("[grade_docs] no documents → irrelevant")
        return {"grade": "irrelevant"}

    docs_text = "\n\n".join(
        f"[{i+1}] {doc.page_content[:300]}" for i, doc in enumerate(docs[:5])
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATES["grade"])
    chain = prompt | _llm_instance | StrOutputParser()

    result = chain.invoke({
        "question": state["question"],
        "documents": docs_text,
    })

    grade = "relevant" if "relevant" in result.lower() and "irrelevant" not in result.lower() else "irrelevant"
    logging.info(f"[grade_docs] grade={grade} (raw: {result.strip()!r})")
    return {"grade": grade}


def generate(state: RAGState) -> dict:
    """Generate answer using retrieved context with citations."""
    docs = state.get("documents", [])
    context = "\n\n".join(
        f"[{i+1}] (Product: {doc.metadata.get('product_name', 'Unknown')}, "
        f"Rating: {doc.metadata.get('rating', 'N/A')}) {doc.page_content}"
        for i, doc in enumerate(docs)
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATES["generate"])
    chain = prompt | _llm_instance | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": state["question"],
    })

    logging.info(f"[generate] answer length: {len(answer)} chars")
    return {"answer": answer}


def rewrite(state: RAGState) -> dict:
    """Rewrite the query for better retrieval."""
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATES["rewrite"])
    chain = prompt | _llm_instance | StrOutputParser()

    rewritten = chain.invoke({"question": state["question"]})
    retries = state.get("retries", 0) + 1

    logging.info(f"[rewrite] retry={retries}, rewritten: {rewritten.strip()!r}")
    return {"rewritten_query": rewritten.strip(), "retries": retries}


# ── Routing logic ──

def route_after_grading(state: RAGState) -> str:
    """Route to generate if relevant, rewrite if irrelevant (with retry limit)."""
    max_retries = _max_retries
    if state.get("grade") == "relevant":
        return "generate"
    if state.get("retries", 0) >= max_retries:
        logging.info("[route] max retries reached → generating with available docs")
        return "generate"
    return "rewrite"


# ── Graph builder ──

# Module-level singletons (initialized once by build_graph)
_retriever_instance: Retriever = None  # type: ignore
_llm_instance = None
_max_retries: int = 2


def build_graph(retriever: Retriever, model_loader: ModelLoader, max_retries: int = 2):
    """Build and compile the LangGraph RAG pipeline."""
    global _retriever_instance, _llm_instance, _max_retries

    _retriever_instance = retriever
    _llm_instance = model_loader.load_llm()
    _max_retries = max_retries

    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_docs", grade_docs)
    workflow.add_node("generate", generate)
    workflow.add_node("rewrite", rewrite)

    # Set entry point
    workflow.set_entry_point("retrieve")

    # Edges
    workflow.add_edge("retrieve", "grade_docs")
    workflow.add_conditional_edges(
        "grade_docs",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        },
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    graph = workflow.compile()
    logging.info(
        f"RAG graph compiled: retrieve → grade_docs → generate/rewrite "
        f"(max_retries={max_retries})"
    )
    return graph

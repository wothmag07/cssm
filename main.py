import logging
from langchain_core.runnables import RunnablePassthrough
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader
from prompts.prompt import PROMPT_TEMPLATES
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI(title="Amazon Product Assistant API", version="1.0.0")

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
    ],  # Next.js default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever_obj = Retriever()
model = ModelLoader()


def invoke_chain(query: str):
    retriever = retriever_obj.load_retriever()
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATES["product_bot"])
    llm = model.load_llm()
    output_parser = StrOutputParser()

    # Create the RAG chain properly
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )

    output = chain.invoke(query)

    return output


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Amazon Product Assistant API"}


@app.post("/retrieve")
async def chat(msg: str = Form(...)):
    """Chat endpoint for product queries"""
    try:
        result = invoke_chain(msg)
        logging.info(f"Response: {result}")
        return JSONResponse(content={"response": result})
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}", exc_info=True)
        error_msg = "Sorry, I'm having trouble accessing the database right now. Please try again in a moment."
        return JSONResponse(content={"error": error_msg})


if __name__ == "__main__":
    logging.info("Starting API server at http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)

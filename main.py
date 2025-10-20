import logging
from langchain_core.runnables import RunnablePassthrough
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader
from prompts.prompt import PROMPT_TEMPLATES
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

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


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Get the index page"""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/retrieve", response_class=JSONResponse)
async def chat(msg: str = Form(...)):
    try:
        result = invoke_chain(msg)
        logging.info(f"Response: {result}")
        return JSONResponse(content={"response": result})
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}", exc_info=True)
        error_msg = "Sorry, I'm having trouble accessing the database right now. Please try again in a moment."
        return JSONResponse(content={"error": error_msg})


if __name__ == "__main__":
    logging.info("Starting server at http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

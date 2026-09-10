from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from rag.chain import chain

load_dotenv()

app = FastAPI(title="5G RCA RAG API")

class QueryRequest(BaseModel):
    question: str

@app.post("/diagnose")
async def diagnose(request: QueryRequest):
    from rag.retriever import build_retriever
    from rag.chain import format_context, prompt, llm
    from langchain_core.output_parsers import StrOutputParser
    import asyncio
    retriever = build_retriever()
    docs = await asyncio.to_thread(retriever.invoke, request.question)
    context = format_context(docs)

    generation_chain = prompt | llm | StrOutputParser()

    async def stream_token():
        async for token in generation_chain.astream({
            "context": context,
            "question": request.question,
        }):
            yield token

    return StreamingResponse(stream_token(), media_type="text/plain")

@app.get("/health")
async def health():
    return {"status": "ok"}
from datetime import datetime
import json
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from rag.chain import chain, format_context
from api.schemas import QueryRequest, TelemetryRequest, DiagnosisResponse
from rag.retriever import build_retriever
import asyncio

load_dotenv()

app = FastAPI(title="5G RCA RAG API")


@app.post("/diagnose/query")
async def diagnose(request: QueryRequest):
    retriever = build_retriever()
    docs = await asyncio.to_thread(retriever.invoke, request.question)
    context = format_context(docs)
    from rag.chain import prompt, llm
    from langchain_core.output_parsers import StrOutputParser
    generation_chain = prompt | llm | StrOutputParser()

    async def stream_token():
        async for token in generation_chain.astream({
            "context": context,
            "question": request.question,
        }):
            yield token

    return StreamingResponse(stream_token(), media_type="text/plain")

@app.post("/diagnose/telemetry", response_model=DiagnosisResponse)
async def diagnose_telemetry(request: TelemetryRequest):
    clean_query = (
        f"Engineering Parameters:\n{request.engineering_parameters}"
        f"\n\nDrive Test Data:\n{request.drive_test_data}"
    )
    retriever = build_retriever()
    docs = await asyncio.to_thread(retriever.invoke, clean_query)
    context = format_context(docs)

    from rag.prompts import TELEMETRY_PROMPT
    from langchain_core.output_parsers import StrOutputParser
    Telemetry_prompt = ChatPromptTemplate.from_messages([
        ("system", TELEMETRY_PROMPT),
        ("human", "{question}"),
    ])
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    telemetry_chain = Telemetry_prompt | llm | StrOutputParser()

    raw_output = await asyncio.to_thread(
        telemetry_chain.invoke,
        {"context":context, "question": clean_query}
    )

    #print("RAW OUTPUT: ---> ",raw_output)

    def extract_json(text):
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())

        raise ValueError(f"No JSON found in output: {text[:200]}")

    parsed = extract_json(raw_output)
    print(parsed)


    input_summary = {
        "engineering_parameters_preview": request.engineering_parameters[:100],
        "drive_test_preview": request.drive_test_data[:100]
    }

    return DiagnosisResponse(
        root_cause_code=parsed["root_cause_code"],
        description=parsed["description"],
        confidence=parsed["confidence"],
        timestamp=datetime.now().strftime("%I:%M%p on %B %d, %Y"),
        citations=parsed["citations"],
        input_summary=input_summary,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
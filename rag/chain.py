import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.cache import RedisSemanticCache
from langchain_core.globals import set_llm_cache
from dotenv import load_dotenv
from retriever import build_retriever
import os

load_dotenv()

# set_llm_cache(RedisSemanticCache(
#     redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
#     embedding=OpenAIEmbeddings(
#         model="text-embedding-3-large", 
#         dimensions=1024
#     ),
#     score_threshold=0.2,
# ))

SYSTEM_PROMPT = """You are a 5G network troubleshooting expert.
You will be given network engineering parameters and drive test measurements.
Analyze the data and identify the root cause of throughput degradation.
Use ONLY the provided context cases to support your diagnosis.
Cite sources as [Source 1 - Root Cause: C1].
State the most likely root cause code (C1-C8) clearly in your answer.
If you cannot determine the root cause, explain what additional data you need.

Context:
{context}"""

# Temporary debug in chain.py format_context function
# def format_context(docs):
#     print(f"DEBUG: {len(docs)} docs retrieved")
#     for doc in docs:
#         print(f"  {doc.metadata}")
#     return "\n\n".join(
#         f"[Source {i+1}] (Root Cause: {doc.metadata['root_cause_code']}):\n{doc.page_content}"
#         for i, doc in enumerate(docs)
#     )

def format_context(docs):
    return "\n\n".join(
        f'[Source {i+1}] ,Root Cause: {doc.metadata['root_cause_code']}:\n{doc.page_content}'
        for i, doc in enumerate(docs)
    )

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

llm = ChatOpenAI(model="gpt-4o", temperature=0)

retriever = build_retriever()

chain = (
    RunnableParallel({
        "context": retriever | format_context,
        "question": RunnablePassthrough(),
    })
    | prompt
    | llm
    | StrOutputParser()
)


#response = chain.invoke("Why does throughput drop when vehicle speed is high?")
#print(response)
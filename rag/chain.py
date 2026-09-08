from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
from retriever import build_retriever

load_dotenv()

SYSTEM_PROMPT = """ You are a 5G network troubleshooting expert. 
Answer the user's question using ONLY the context provided below.
Cite your scores using for example, [Source 1 - Root Cause: C7].
At the end of your answer, list all sources you cited with their root cause codes.
If the answer is not in the context, say "I don't have enough informationto answer that."
Context:
{context}
"""

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


response = chain.invoke("Why does throughput drop when vehicle speed is high?")
print(response)
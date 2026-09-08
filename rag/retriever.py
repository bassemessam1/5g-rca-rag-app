from langchain_pinecone import PineconeVectorStore
from langchain_classic.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_cohere import CohereRerank
from dotenv import load_dotenv
import os

load_dotenv()

def build_retriever():

    embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-large",
        dimensions=1024,
    )

    vectorstroe = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings,
        namespace=os.getenv("PINECONE_NAMESPACE"),
    )


    multi_retriever = MultiQueryRetriever.from_llm(
        retriever = vectorstroe.as_retriever(search_kwargs={"k": 20}),
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    )

    cohere_reranker = CohereRerank(
        model="rerank-english-v3.0", 
        top_n=5
    )

    rerank_retriever = ContextualCompressionRetriever(
        base_compressor=cohere_reranker,
        base_retriever=multi_retriever,
    )
    return rerank_retriever

#results = rerank_retriever.invoke("Why does throughput drop when vehicle speed is high?")

# for i, doc in enumerate(results):
#     print(f"\n--- Results {i+1} ---")
#     print(f"Metadata: {doc.metadata}")
#     print(f"Content preview: {doc.page_content[:200]}")

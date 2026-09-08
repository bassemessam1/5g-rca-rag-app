from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
from formatter import build_documents, telelogs_train

load_dotenv()
documents = build_documents(telelogs_train, split='train')
print(f'the length of the documents is: {len(documents)}')

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024
)

vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    namespace=os.getenv("PINECONE_NAMESPACE"),

    )
print(f"Successfully upserted {len(documents)} documents into Pinecone")
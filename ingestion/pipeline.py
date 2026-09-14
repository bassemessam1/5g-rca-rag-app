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


## TEST
# vectorstore = PineconeVectorStore(
#     embedding=embeddings,
#     index_name=os.getenv("PINECONE_INDEX_NAME"),
#     namespace=os.getenv("PINECONE_NAMESPACE"),

#     )
# test_queries = [ "downtilt antenna coverage weak far end", 
#                 "coverage distance exceeds 1km overshooting", 
#                 "neighboring cell higher throughput", 
#                 "co-frequency interference overlapping", 
#                 "frequent handovers degrade performance", 
#                 "PCI mod 30 same interference", 
#                 "vehicle speed 40 kmh throughput impact", 
#                 "scheduled resource blocks below 160", ] 
# for query in test_queries: 
#     results = vectorstore.similarity_search(query, k=3) 
#     codes = [r.metadata['root_cause_code'] for r in results] 
#     print(f"Query: {query[:40]} -> {codes}")

# query = """gNodeB ID|Cell ID|Longitude|Latitude|Mechanical Azimuth|Mechanical Downtilt|Digital Tilt|Digital Azimuth|Beam Scenario|Height|PCI|TxRx Mode|Max Transmit Power|Antenna Model
# 0000258|1|128.139529|32.623035|45|3|7|5|SCENARIO_7|9.0|737|32T32R|34.9|NR AAU 1
# 0000258|26|128.139529|32.623035|145|6|255|0|DEFAULT|9.0|291|32T32R|34.9|NR AAU 1
# 0000258|15|128.139529|32.623042|100|31|9|0|SCENARIO_1|15.0|919|32T32R|34.9|NR AAU 1"""

# results = vectorstore.similarity_search(query, k=5)
# codes = [r.metadata['root_cause_code'] for r in results]
# print(codes)


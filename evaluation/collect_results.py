import sys, os, json
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from rag.retriever import build_retriever
from rag.chain import chain, format_context

questions = [
    "How can the serving cell's downtilt angle affect weak coverage?",
    "What happens to cell edge users when downtilt is too steep?",
    "Why does increasing downtilt reduce coverage distance?",
    "What does cause overshooting of the serving cell?",
    "What happens when the cell's coverage distance exceeds 1 km?",
    "What are the relationship between the overshooting and the cell's coverage?",
    "Why does the neighbour cell provide higher throughput?",
    "What could be the reason for the neighbour cells offering higher throughput?",
    "Why is the serving cell offering lower throughput than the neighbour cell?",
    "What is the cause of the overlapping coverage?",
    "How can Non-colocated co-frequency neighbouring cells cause overlapping?",
    "How co-frequency neighbouring cells affect the overlapping?",
    "How do multiple handovers affect the performance?",
    "What causes the degraded performance?",
    "What happens with frequent handovers?",
    "What does cause the interference?",
    "What happens if the serving and neighbouring cells have the same PCI mod 30?",
    "How is the similar PCI mod affecting the interference?",
    "Why is the throughput drop when the vehicle is moving fast?",
    "What happens if the vehicle speed exceeds 40 km/h?",
    "Why is the vehicle affecting the throughput?",
    "How is the scheduled RBs affecting throughput?",
    "What is the threshold of the average RBs to affect the throughput?",
]

def collect():
    retriever = build_retriever()
    results = []

    for i, question in enumerate(questions):
        print(f"Processing {i+1}/{len(questions)}: {question[:50]}...")
        docs     = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        answer   = chain.invoke(question)

        results.append({
            "question": question,
            "answer":   answer,
            "contexts": contexts,
        })
        time.sleep(10)

    with open("evaluation/ragas_input.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to evaluation/ragas_input.json")

if __name__ == "__main__":
    collect()

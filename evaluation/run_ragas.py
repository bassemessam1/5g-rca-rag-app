import json
import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def measure_faithfulness(question, answer, contexts):
    context_str = "\n\n".join(contexts)
    prompt = f"""You are evaluating a RAG system.

Context provided to the system:
{context_str}

Generated answer:
{answer}

Task: Check if every claim in the answer is supported by the context.
Score from 0.0 to 1.0 where:
1.0 = all claims are fully supported by the context
0.0 = answer contains claims not found in the context at all

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 0.0

def measure_answer_relevancy(question, answer):
    prompt = f"""You are evaluating a RAG system.

Question asked:
{question}

Generated answer:
{answer}

Task: Score how well the answer addresses the question.
Score from 0.0 to 1.0 where:
1.0 = answer directly and completely addresses the question
0.0 = answer is completely irrelevant to the question

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 0.0

def evaluate():
    with open("evaluation/ragas_input.json") as f:
        data = json.load(f)

    faithfulness_scores = []
    relevancy_scores    = []

    print(f"Evaluating {len(data)} questions...\n")
    for i, item in enumerate(data):
        print(f"  Evaluating {i+1}/{len(data)}: {item['question'][:50]}...")

        f_score = measure_faithfulness(
            item["question"],
            item["answer"],
            item["contexts"],
        )
        r_score = measure_answer_relevancy(
            item["question"],
            item["answer"],
        )

        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)
        print(f"    Faithfulness: {f_score:.2f} | Relevancy: {r_score:.2f}")

    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_relevancy    = sum(relevancy_scores)    / len(relevancy_scores)

    print(f"\n=== Evaluation Results ({len(data)} questions) ===")
    print(f"  Faithfulness    : {avg_faithfulness:.3f}")
    print(f"  Answer Relevancy: {avg_relevancy:.3f}")

    # Save results
    with open("evaluation/ragas_results.json", "w") as f:
        json.dump({
            "faithfulness":     avg_faithfulness,
            "answer_relevancy": avg_relevancy,
            "per_question": [
                {
                    "question":     data[i]["question"],
                    "faithfulness": faithfulness_scores[i],
                    "relevancy":    relevancy_scores[i],
                }
                for i in range(len(data))
            ]
        }, f, indent=2)
    print("\nDetailed results saved to evaluation/ragas_results.json")

if __name__ == "__main__":
    evaluate()


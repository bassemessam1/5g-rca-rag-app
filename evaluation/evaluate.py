import json
import re
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.chain import chain
from rag.retriever import build_retriever
from ingestion.formatter import extract_section
import logging
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.DEBUG)

def build_query(question_text):
    lines = question_text.split('\n')
    engineering = extract_section(lines, "Engeneering parameters data as follows")
    drivetest = extract_section(lines, "User plane drive test data as follows：")
    return f"Engineering parameters: \n{engineering}\n\nDrive Test Data:\n{drivetest}"


def extract_predicted_code(rag_output):
    # Search for patterns like C1, C2, ...., C8 in the output.
    matches =  re.findall(r'\bC[1-8]\b', rag_output)
    if matches:
        return matches[0] # return first match
    return None


def evaluate(test_path, sample_size=50):
    with open(test_path) as f:
        test_data = json.load(f)

    # Sample to control cost — don't run all 864 at once
    test_data = test_data[:sample_size]

    correct = 0
    answered = 0
    total = len(test_data)
    results = []

    for i, record in enumerate(test_data):
        question = record['question']
        ground_truth = record['answer']

        print(f"Processing {i+1}/{total}...", end="\r")


        clean_query = build_query(question)
        #print(f"Clean Query: ---> {clean_query}")
        rag_output = chain.invoke(clean_query)
        print(rag_output)
        retriever = build_retriever()
        docs = retriever.invoke(question)
        retrieved_codes = [doc.metadata['root_cause_code'] for doc in docs]
        predicted = extract_predicted_code(rag_output)
        print(f"\nGround truth: {ground_truth} | Retrieved: {retrieved_codes} | predicted: {predicted}")

        if predicted is not None:
            answered += 1
            if predicted == ground_truth:
                correct += 1

        results.append({
            'ground_truth': ground_truth,
            'predicted': predicted,
            'correct': predicted == ground_truth,
        })
        time.sleep(7)

    accuracy    = correct / answered if answered > 0 else 0
    answer_rate = answered / total

    print("\nWrong predictions sample:")
    wrong = [r for r in results if not r['correct']]
    for r in wrong[:10]:
        print(f" Ground truth: {r['ground_truth']} | Predict: {r['predicted']}")

    print(f"\n\nResults on {total} test samples:")
    print(f"  Answer rate : {answer_rate:.1%}")
    print(f"  Accuracy    : {accuracy:.1%}")

    return results

if __name__ == "__main__":
    evaluate("data/raw/telelogs_test.json", sample_size=50)



import json
from ingestion.formatter import extract_section

with open('data/raw/telelogs_test.json') as f:
    test_data = json.load(f)

def build_query(question_text):
    lines = question_text.split('\n')
    engineering = extract_section(lines, "Engeneering parameters data as follows")
    drivetest   = extract_section(lines, "User plane drive test data as follows：")
    return f"Engineering Parameters:\n{engineering}\n\nDrive Test Data:\n{drivetest}"

# for i in range(3):
#     query = build_query(test_data[i]['question'])
#     print(f"\n=== Record {i} | \n Ground truth: {test_data[i]['answer']} ===")
#     print(query)
#     print("---")


for i in range(5):
    q = test_data[i]['question']
    lines = q.split('\n')
    # Find the timestamp line
    for line in lines:
        if '2025' in line and '|' in line:
            print(f"Record {i} | Answer: {test_data[i]['answer']} | First timestamp: {line[:50]}")
            break

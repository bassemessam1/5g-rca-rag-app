import json
from langchain_core.documents import Document


def extract_root_cause_description(question_text, answer_code):
    for line in question_text:
        parts = line.split(':', 1)
        if parts[0] == answer_code:
            return parts[1].strip()
    return ""

def extract_section(question_text, section_marker):
    lines = []
    capturing = False
    for line in question_text:
        if section_marker in line:
            capturing = True
            continue
        if capturing:
            if line.strip() == "" and len(lines) > 0:
                break
            lines.append(line)
    return "\n".join(lines)

def get_category(answer_code):

    if answer_code in ['C1','C2']:
        category = 'Coverage problems'
    if answer_code in ['C4','C6']:
        category = 'Interference problems'
    if answer_code in ['C5','C7']:
        category = 'Mobility problems'
    if answer_code in ['C3','C8']:
        category = 'Resource problems'    

    return category

def format_document(record):
    question_text = record['question'].split('\n')
    answer_code   = record['answer']

    description       = extract_root_cause_description(question_text, answer_code)
    engineering_table = extract_section(question_text, "Given")

    drivetest_table   = extract_section(question_text, "User plane drive test data as follows：")

    document = (
        f"ROOT CAUSE: {answer_code}\n"
        f"DESCRIPTION: {description}\n\n"
        f"ENGINEERING PARAMETERS:\n{engineering_table}\n\n"
        f"DRIVE TEST DATA:\n{drivetest_table}"
    )
    metadata = {
        "root_cause_code": answer_code,
        "category": get_category(answer_code),
        "split": "train"
    }
    return document, metadata

def build_documents(records, split="train"):
    documents = []
    for record in records:
        text,metadata = format_document(record)
        metadata["split"] = split
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)
    return documents


# ── Test on first record ──────────────────────────────────────────────────
with open('./data/raw/telelogs_train.json') as f:
    telelogs_train = json.load(f)

documents = build_documents(telelogs_train)

#doc = format_document(telelogs_train[0])
#print(doc)
#print(f"\nDocument length: {len(doc)} characters / ~{len(doc)//4} tokens")
print(f"Total documents: {len(documents)}")
#print(f"\nFirst Document page_content:\n{documents[0].page_content}")
#print(f"\nFirst document metadata:\n{documents[0].metadata}")


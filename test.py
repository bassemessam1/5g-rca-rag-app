import os
from dotenv import load_dotenv

load_dotenv()

keys = [
    "OPENAI_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "PINECONE_NAMESPACE",
    "COHERE_API_KEY",
]

print("Checking environment variables...")
all_good = True
for key in keys:
    val = os.getenv(key, "")
    if val:
        print(f"  OK  {key}")
    else:
        print(f"  MISSING  {key}")
        all_good = False

if all_good:
    print("\nAll keys set. Ready for Phase 2.")
else:
    print("\nFill in the missing keys in your .env file.")

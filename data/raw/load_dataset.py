import os
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
import json


load_dotenv()
HF_TOKEN = os.getenv('HF_KEY')

if HF_TOKEN:
    login(HF_TOKEN)
    print("Successfully logged in to Hugging Face!")
else:
    print("Token is not set. Please save the token first.")


ds = load_dataset("netop/TeleLogs")

for split in ds:
    records = [ds[split][i] for i in range(len(ds[split]))]
    with open(f"./data/raw/telelogs_{split}.json", "w") as f:
        json.dump(records, f, indent=2)



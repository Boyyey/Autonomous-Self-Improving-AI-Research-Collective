# hf_utils.py
from huggingface_hub import HfApi, upload_file
import json
import os

HF_TOKEN = os.getenv("HF_TOKEN")  # Set your token: huggingface-cli login
REPO_ID = "your-username/self-improving-ai"  # Change to your repo

def create_model_card(accuracy, generation, genome):
    card = f"""
---
license: mit
library_name: torch
tags:
- self-improving
- nas
- evolutionary-ai
---

# Self-Evolving AI Model

This model was **automatically generated** by a recursive self-learning agent.

- **Generation**: {generation}
- **Accuracy**: {accuracy:.4f}
- **Architecture**: {json.dumps(genome, indent=2)}
    """
    with open("checkpoints/README.md", "w") as f:
        f.write(card)

def upload_to_hf(generation, accuracy, genome):
    if not HF_TOKEN:
        print("HF_TOKEN not set. Skipping upload.")
        return

    api = HfApi()
    create_model_card(accuracy, generation, genome)

    try:
        api.create_repo(repo_id=REPO_ID, exist_ok=True, token=HF_TOKEN)
        upload_file(
            path_or_fileobj=f"checkpoints/gen_{generation}.pth",
            path_in_repo="pytorch_model.bin",
            repo_id=REPO_ID,
            token=HF_TOKEN
        )
        upload_file(
            path_or_fileobj="checkpoints/README.md",
            path_in_repo="README.md",
            repo_id=REPO_ID,
            token=HF_TOKEN
        )
        print(f"✅ Model gen {generation} uploaded to Hugging Face: {REPO_ID}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
# llm_mutator.py
from transformers import pipeline
import re
import random

# Use small local model for fast code generation
llm = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=0 if torch.cuda.is_available() else -1)

def llm_suggest_layer():
    prompt = """
    Generate one PyTorch layer config in JSON-like dict format.
    Only output the dict. Options: Conv2d, Linear, BatchNorm2d.
    Example: {'type': 'conv', 'filters': 64, 'kernel': 3}
    """
    try:
        response = llm(prompt, max_new_tokens=50, num_return_sequences=1)[0]['generated_text']
        # Extract dict-like string
        match = re.search(r"\{.*?\}", response)
        if match:
            return eval(match.group())
    except:
        pass
    # Fallback
    return random.choice([
        {'type': 'conv', 'filters': random.choice([64, 128]), 'kernel': 3},
        {'type': 'linear', 'units': 10}
    ])
import os

import torch
from dotenv import load_dotenv, find_dotenv
from llama_index.llms.openai import OpenAI

# ---------- Load environment variables
load_dotenv(find_dotenv())
OPENAI_MODEL = os.getenv("OPENAI_MODEL")


def llm():
    return OpenAI(model=OPENAI_MODEL)


def check_cuda():
    print(f"CUDA version that PyTorch is using: {torch.version.cuda}")
    print(f"GPU is available for PyTorch? {torch.cuda.is_available()}")
    print("GPU if available: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

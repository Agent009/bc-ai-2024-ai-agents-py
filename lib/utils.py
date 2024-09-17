import os

import torch
from dotenv import load_dotenv, find_dotenv
from llama_index.llms.openai import OpenAI

# ---------- Load environment variables
load_dotenv(find_dotenv())
OPENAI_MODEL = os.getenv("OPENAI_MODEL")


def llm(model=OPENAI_MODEL, temperature=0.7, max_tokens=1000):
    """
    Initialize and return an instance of the OpenAI LLM.

    Parameters:
    model (str): The OpenAI model to use. Default is the value of the OPENAI_MODEL environment variable.
    temperature (float): The randomness of the model's output. Higher values make output more random, while lower values make it more deterministic. Default is 0.7.
    max_tokens (int): The maximum number of tokens to generate. Default is 1000.

    Returns:
    OpenAI: An instance of the OpenAI LLM with the specified parameters.
    """
    return OpenAI(model=model, temperature=temperature, max_tokens=max_tokens)



def check_cuda():
    """
    This function checks the availability and details of CUDA and GPU for PyTorch.

    It prints the following information:
    1. The CUDA version that PyTorch is using.
    2. Whether a GPU is available for PyTorch.
    3. The name of the GPU if available, otherwise, it prints "No GPU detected".

    Parameters:
    None

    Returns:
    None
    """
    print(f"CUDA version that PyTorch is using: {torch.version.cuda}")
    print(f"GPU is available for PyTorch? {torch.cuda.is_available()}")
    print("GPU if available: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

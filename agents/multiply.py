import os
from dotenv import load_dotenv, find_dotenv
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent


# ---------- Load environment variables
load_dotenv(find_dotenv())
OPENAI_MODEL = os.getenv("OPENAI_MODEL")


# ---------- Setup function calls
def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


# ---------- Configure agent
def get_agent() -> ReActAgent:
    """Configure the AI agent"""
    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    llm = OpenAI(model=OPENAI_MODEL)
    agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)
    return agent


def run_agent():
    default_input = "What is 123 * (1 + 2 + 3)?"
    user_input = input(f"Give me a maths problem to solve (default: '{default_input}'): ")
    input_text = user_input if user_input else default_input
    agent = get_agent()
    agent.chat(input_text)

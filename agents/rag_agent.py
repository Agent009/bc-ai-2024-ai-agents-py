from llama_index.core import Settings
from llama_index.core.agent import ReActAgent

from agents.tools import query_tool
from lib.utils import llm


def run_agent():
    """
    This function sets the LLM settings, defines a ReActAgent using a query tool,
    and then tests the agent by asking a question.

    Parameters:
    None

    Returns:
    None

    Example:
    >>> run_agent()
    Ask me a question (default: 'How to play Zork I? or Explain a step by step on how to beat the first level of Zork I.'):
    """
    # Set the model settings
    Settings.llm = llm(temperature=0, max_tokens=100)

    # Define an agent to use the tool
    agent = ReActAgent.from_tools(
        [query_tool()], verbose=True
    )

    default_input = "How to play Zork I? or Explain a step by step on how to beat the first level of Zork I."
    user_input = input(f"Ask me a question (default: '{default_input}'): ")
    query = user_input if user_input else default_input

    # Test the RAG agent
    agent.chat(query)

from llama_index.core.agent import (
    ReActAgent,
    StructuredPlannerAgent,
    FunctionCallingAgentWorker, )

from agents.tools import multiply_tool
from lib.utils import llm


# ---------- Configure agent
def get_simple_agent() -> ReActAgent:
    """
    Configure a simple AI agent using the ReActAgent class.
    This function creates an instance of the ReActAgent class with a single tool.

    Parameters:
    None

    Returns:
    ReActAgent: An instance of the ReActAgent class configured with the specified tools and settings.
    """
    agent = ReActAgent.from_tools([multiply_tool()], llm=llm(), verbose=True)
    return agent


def get_worker_agent() -> StructuredPlannerAgent:
    """
    Configure the AI agent using a Structured Planner Agent with a Function Calling Agent Worker.
    This function creates a Structured Planner Agent that utilizes a Function Calling Agent Worker.

    Parameters:
    None

    Returns:
    StructuredPlannerAgent: An instance of StructuredPlannerAgent configured with the specified worker and tools.

    """
    worker = FunctionCallingAgentWorker.from_tools([multiply_tool()], llm=llm(), verbose=True)
    agent = StructuredPlannerAgent(worker, [multiply_tool()], verbose=True)

    return agent


def run_agent(agent_type="simple"):
    """
    This function runs a math problem-solving agent based on the provided agent type.

    Parameters:
    agent_type (str): The type of agent to run. It can be either "simple" or "worker".
                      Default value is "simple".

    Returns:
    None

    Raises:
    ValueError: If the provided agent type is not "simple" or "worker".
    """
    if agent_type not in ["simple", "worker"]:
        raise ValueError("Agent type should only allow the following values: simple, worker")

    default_input = "Solve the equation x = 123 * (x + 2y + 3)"
    user_input = input(f"Give me a maths problem to solve (default: '{default_input}'): ")
    input_text = user_input if user_input else default_input

    if agent_type == "simple":
        agent = get_simple_agent()
    else:
        agent = get_worker_agent()

    agent.chat(input_text)

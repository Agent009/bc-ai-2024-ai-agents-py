import os

from dotenv import load_dotenv, find_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool
from llama_parse import LlamaParse

from lib.multiply import multiply

# ---------- Load environment variables
load_dotenv(find_dotenv())
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")


# ---------- Configure tools

def multiply_tool() -> FunctionTool:
    """
    This function creates and returns a FunctionTool instance configured to use the 'multiply' function.

    Returns:
    FunctionTool: A configured FunctionTool instance that can be used to perform multiplication operations.
    """
    return FunctionTool.from_defaults(fn=multiply)


def query_tool(document="zork1.pdf", result_type="markdown", name="Zork",
               description="A RAG engine with the Game Manual for Zork I: The Great Underground Empire.") -> QueryEngineTool:
    """
    This function creates and configures a Query Engine Tool for a given document and result type.
    It uses LlamaParse to load data from a specified document, creates a VectorStoreIndex, and then
    defines a Query Engine from the index. The function also tests the Query Engine with a sample query.
    Finally, it creates and returns a Query Engine Tool with the specified name and description.

    Parameters:
    document (str): The name of the document to load data from. Default is "zork1.pdf".
    result_type (str): The type of result to be returned by LlamaParse. Default is "markdown".
    name (str): The name of the Query Engine Tool. Default is "Zork".
    description (str): The description of the Query Engine Tool. Default is "A RAG engine with the Game Manual for Zork I: The Great Underground Empire.".

    Returns:
    QueryEngineTool: A configured Query Engine Tool.
    """
    # Define the Query Engine
    documents = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type=result_type).load_data(
        "./data/" + document
    )
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    # Test the query engine:
    query = "What is the concept of interactive fiction?"
    response = query_engine.query(query)
    print(query)
    print(response)

    # Define and configure the Query Engine Tool
    return QueryEngineTool.from_defaults(
        query_engine,
        # Tool name should be the most probable word linked with the tool when the agent is asked to perform a task
        name=name,
        description=description,
    )

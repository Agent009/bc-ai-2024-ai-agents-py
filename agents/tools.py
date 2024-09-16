from llama_index.core.tools import FunctionTool

from lib.multiply import multiply


# ---------- Configure tools
def multiply_tool() -> FunctionTool:
    return FunctionTool.from_defaults(fn=multiply)

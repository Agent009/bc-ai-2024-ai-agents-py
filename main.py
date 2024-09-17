from agents.maths_agent import run_agent as run_maths_agent
from agents.rag_agent import run_agent as run_rag_agent

# ---------- Main script
if __name__ == '__main__':
    # ---------- Run multiply AI agent
    # run_maths_agent("simple")
    # run_maths_agent("worker")

    # ---------- Run RAG AI agent
    run_rag_agent()

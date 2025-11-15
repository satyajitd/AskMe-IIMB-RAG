from workflow.workflow import Workflow

# Create and expose the compiled graph once (for langgraph dev)
workflow = Workflow()
graph = workflow.graph

__all__ = ["workflow", "graph"]


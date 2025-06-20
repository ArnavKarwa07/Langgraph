from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_ollama.llms import OllamaLLM
from langgraph.graph import StateGraph, START, END
from IPython.display import display, Image

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = OllamaLLM(model="gemma3")

def process(state: AgentState) -> AgentState:
    """Process the state by invoking the LLM with the messages."""
    response = llm.invoke( state["messages"])
    print(f"\nAI: {response}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()

display(Image(app.get_graph().draw_mermaid_png()))  # Display the graph as an image

user_input = input("You: ")
while user_input.lower() != "exit":
    app.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("You: ")

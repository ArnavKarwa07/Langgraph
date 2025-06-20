from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages # Reducer function # Add messages to the state instead of replacing them
from langgraph.prebuilt import ToolNode
from IPython.display import display, Image

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def addition(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b
@tool
def subtraction(a: int, b: int) -> int:
    """Subtracts two numbers."""
    return a - b
@tool
def multiplication(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b
@tool
def division(a: int, b: int) -> float:
    """Divides two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

tools = [addition, subtraction, multiplication, division]

model = ChatOllama(model="mistral").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are a helpful assistant that can perform basic arithmetic operations.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> bool:
    """Check if the conversation should continue."""
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("model_call", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "model_call")
graph.add_conditional_edges(
    "model_call", 
    should_continue,
    {
        "continue": "tool_node",
        "end": END
    }
)
graph.add_edge("tool_node", "model_call")
app = graph.compile()

def print_stream(stream):
    """Prints the stream of messages."""
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [{"role": "user", "content": "What is 5 + 3?"}]}
print_stream(app.stream(inputs, stream_mode="values"))
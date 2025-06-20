from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama.llms import OllamaLLM
from langgraph.graph import StateGraph, START, END
from IPython.display import display, Image

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = OllamaLLM(model="gemma3", temperature=0.1)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])

    response_message = AIMessage(content=response)
    state["messages"].append(response_message)
    print(f"\nAI: {response_message.content}\n") 

    print(f"Current state: {state["messages"]}")

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()

# display(Image(app.get_graph().draw_mermaid_png()))  # Display the graph as an image

conversation_history = []

user_input = input("You: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = app.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("You: ")

print("Conversation ended.")

# Save the conversation history to a file
with open("conversation_history.txt", "w", encoding="utf-8") as f:
    f.write("Conversation History:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")
    f.write("\nEnd of conversation.\n")
print("Conversation history saved to 'conversation_history.txt'.")
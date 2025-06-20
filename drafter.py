from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import os

# Global variable
document_content = ""

class AgentState(TypedDict):
    """State of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """Update the document content."""
    global document_content
    document_content = content
    print(f"Document content updated: {content[:100]}...")  # Show first 100 chars
    return f"Document updated with content: {content}"

@tool
def save(filename: str) -> str:
    """Save the document content.
    Args:
        filename (str): The name of the text file to save the content to.
    """
    global document_content
    if not document_content:
        return "No content to save. Please update the document first."
    
    if not filename.endswith('.txt'):
        filename += '.txt'

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(document_content)
        
        print(f"✓ Document saved as {filename}")
        return f"Content successfully saved to file: {filename}"
    except Exception as e:
        error_msg = f"Error saving document: {str(e)}"
        print(f"✗ {error_msg}")
        return error_msg

@tool
def view() -> str:
    """View the current document content."""
    global document_content
    if not document_content:
        return "No document content available."
    return f"Current document content:\n{document_content}"

tools = [update, save, view]

# Initialize model
try:
    model = ChatOllama(model="mistral").bind_tools(tools)
    print("✓ Model initialized successfully")
except Exception as e:
    print(f"✗ Error initializing model: {e}")
    raise

def our_agent(state: AgentState) -> AgentState:
    """Agent function to process messages and interact with the model."""
    system_prompt = SystemMessage(content="""You are a drafter, a helpful assistant that helps users update and modify documents. 

Available tools:
- update(content): Update the document with new content
- save(filename): Save the current document to a file
- view(): View the current document content

Always use the appropriate tool when the user asks to update, save, or view the document. 
Be clear about what actions you're taking and their results.""")
    
    # Get user input
    if not state["messages"]:
        user_input = input("Hello! I am your drafter. What would you like to do? (update/save/view/quit): ")
    else:
        user_input = input("\nWhat would you like to do? (update/save/view/quit): ")
    
    # Check for quit command
    if user_input.lower() in ['quit', 'exit', 'q']:
        return {"messages": list(state["messages"]) + [HumanMessage(content="quit")]}
    
    user_message = HumanMessage(content=user_input)
    
    # Prepare messages for the model
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    
    try:
        response = model.invoke(all_messages)
        print(f"\nAI: {response.content}")
        
        # Debug: Show if there are tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"Debug - Tool calls detected: {len(response.tool_calls)}")
            for i, tool_call in enumerate(response.tool_calls):
                print(f"  Tool {i+1}: {tool_call.get('name', 'unknown')} with args: {tool_call.get('args', {})}")
        
        return {"messages": list(state["messages"]) + [user_message, response]}
    
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        print(f"✗ {error_msg}")
        error_response = AIMessage(content=error_msg)
        return {"messages": list(state["messages"]) + [user_message, error_response]}

def should_continue(state: AgentState) -> str:
    """Check if the conversation should continue."""
    messages = state["messages"]
    if not messages:
        return "continue"
    
    # Check if user wants to quit
    last_message = messages[-1] if messages else None
    if isinstance(last_message, HumanMessage) and last_message.content.lower() in ['quit', 'exit', 'q']:
        return "end"
    
    # Check if document was successfully saved
    for message in reversed(messages[-5:]):  # Check last 5 messages
        if isinstance(message, ToolMessage):
            if "successfully saved" in message.content.lower():
                save_response = input("\nDocument saved! Do you want to continue working? (y/n): ")
                if save_response.lower() in ['n', 'no']:
                    return "end"
                break
    
    return "continue"

def print_messages(messages):
    """Prints the messages in a readable format."""
    if not messages:
        return
    
    # Only print tool messages to avoid spam
    for message in messages[-2:]:  # Show last 2 messages
        if isinstance(message, ToolMessage):
            print(f"Tool Result: {message.content}")

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("our_agent", our_agent)
graph.add_node("tool_node", ToolNode(tools=tools))

graph.add_edge(START, "our_agent")
graph.add_conditional_edges(
    "our_agent",
    lambda state: "tool_node" if (
        state["messages"] and 
        hasattr(state["messages"][-1], 'tool_calls') and 
        state["messages"][-1].tool_calls
    ) else "continue",
    {
        "tool_node": "tool_node",
        "continue": "our_agent"
    }
)

graph.add_conditional_edges(
    "tool_node", 
    should_continue,
    {
        "continue": "our_agent",
        "end": END
    }
)

app = graph.compile()

def run_agent():
    """Run the agent and print the messages."""
    print("=" * 50)
    print("    Welcome to the Drafter Agent!")
    print("=" * 50)
    print("Commands:")
    print("  - Type content to update the document")
    print("  - Ask to 'save filename' to save document")
    print("  - Ask to 'view' to see current content")
    print("  - Type 'quit' to exit")
    print("=" * 50)
    
    state = {"messages": []}
    
    try:
        for step in app.stream(state, stream_mode="values"):
            if "messages" in step:
                print_messages(step["messages"])
                # Update state for next iteration
                state = step
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
    
    print("\n" + "=" * 50)
    print("    Thank you for using the Drafter Agent!")
    print("=" * 50)

if __name__ == "__main__":
    run_agent()
# Expose python function to model as callable tool
# Before running
# 1. Install the required packages:
#    pip install -r requirements.txt
# 2. Create environment variable OPENAI_API_KEY (setx or $env:OPENAI_API_KEY = "YOUR_API_KEY")

# Notes
# Initially, the LLM assumed that get_current_balance woudl return the balance and that's all it called
# I had to use annotations to clarify that get_pending_transactions was also needed to calculate the final balance.

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from typing import Annotated
from langchain_core.tools import tool

llm = ChatOpenAI(model="gpt-4o")

@tool
def get_current_balance(
    account_id: Annotated[str, "The customer's account number"]
) -> float:
    """Get the current posted balance of an account.
    Does NOT include pending transactions."""
    return 1000.00

@tool
def get_pending_transactions(
    account_id: Annotated[str, "The customer's account number"]
) -> list[float]:
    """Get pending (not yet posted) transaction amounts for an account.
    Negative = debit, positive = credit. This is NOT a balance."""
    return [-200.00, -50.00]

@tool
def calculate_final_balance(
    balance: Annotated[float, "Current posted balance from get_current_balance"],
    pending: Annotated[list[float], "Amounts from get_pending_transactions"],
) -> float:
    """Calculate the pending balance (also called available or projected balance):
    current posted balance plus all pending transactions.
    Use this whenever the user asks for a pending balance."""
    return balance + sum(pending)

# Tools receive only their arguments from the tool call, not the full state.
tools = [get_current_balance, get_pending_transactions, calculate_final_balance]
llm_with_tools = llm.bind_tools(tools)

# Every node receives the FULL state.
def llm_node(state: MessagesState):
    # The ENTIRE message history is sent to the LLM here, so it grows every loop.
    # To limit it: state["messages"][-10:] or trim_messages.
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)

# add_node only REGISTERS a node. It isn't connected until edges are added.
builder.add_node("llm", llm_node)

# If one AIMessage requests several tools, ToolNode runs all of them in one pass,
# followed by ONE LLM call (not one per tool).
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)

# After the tools run, the LLM is ALWAYS called again.
# Alternative: tools → END to return raw tool results with no extra LLM call.
builder.add_edge("tools", "llm")

# compile() catches edges to missing nodes and a missing entry point.
# It may NOT flag unconnected nodes (<90% confident), so check with:
#   print(graph.get_graph().draw_mermaid())
graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage("What is my pending balance for account 123?")]})

for m in result["messages"]:
    m.pretty_print()

# sample response
# What is my pending balance for account 123?
# ================================== Ai Message ==================================
# Tool Calls:
#   get_current_balance (call_l7fEHE8jgkSdwKe6FjIeU0jS)
#  Call ID: call_l7fEHE8jgkSdwKe6FjIeU0jS
#   Args:
#     account_id: 123
#   get_pending_transactions (call_XRzXapO6Qt5xGTNJgZCAH9eD)
#  Call ID: call_XRzXapO6Qt5xGTNJgZCAH9eD
#   Args:
#     account_id: 123
# ================================= Tool Message =================================
# Name: get_current_balance

# 1000.0
# ================================= Tool Message =================================
# Name: get_pending_transactions

# [-200.0, -50.0]
# ================================== Ai Message ==================================
# Tool Calls:
#   calculate_final_balance (call_G8RqeouWovjx1bNcxkEHDOtA)
#  Call ID: call_G8RqeouWovjx1bNcxkEHDOtA
#   Args:
#     balance: 1000.0
#     pending: [-200.0, -50.0]
# ================================= Tool Message =================================
# Name: calculate_final_balance

# 750.0
# ================================== Ai Message ==================================

# Your pending balance for account 123 is $750.00.
#
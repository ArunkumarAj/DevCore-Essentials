import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

# Working With Tools
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun

# Langgraph Application
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition

def initialize_tools():
    # Initialize Arxiv and Wikipedia tools - (third party tools)
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

    return arxiv_tool, wiki_tool

def query_tools(wiki_tool, arxiv_tool):
    # query Wikipedia and Arxiv tools
    wiki_result = wiki_tool.invoke("who is Elon musk?")  # Quantum Computing
    print(">> " + wiki_result + " <<") 
    # # it is a tool to get all scientific released artical papers
    arxiv_result = arxiv_tool.invoke("Attention is all you need") # Vision Transformers(ViT), ImageNet Classification, Yolo
    print(">> " + arxiv_result + " <<") 

def setup_graph(tools):
    #  Langgraph Application, setup the state graph with tools
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    graph_builder = StateGraph(State)
    groq_api_key = os.getenv("groq_api_key")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

    # Giving the LLM for the additional functionality
    llm_with_tools = llm.bind_tools(tools=tools)

    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    return graph_builder.compile()

def main():
    arxiv_tool, wiki_tool = initialize_tools()
    # query_tools(wiki_tool, arxiv_tool)

    # tools that to combain in a list
    tools = [wiki_tool, arxiv_tool]
    graph = setup_graph(tools)

    conversation_history = []

    print("Start chatting! Type 'exit' to end the conversation.")
    print(" ===== > Chatbot-using-LangGraph-Tools < ===== ")

    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            print("System: Goodbye!")
            break

        # Add user input to conversation history
        conversation_history.append(("user", user_input))

        events = graph.stream({"messages": conversation_history}, stream_mode="values")

        for event in events:
            response = event["messages"][-1]
            response.pretty_print()
            # Add system response to conversation history
            conversation_history.append(("system", response.content))

if __name__ == "__main__":
    main()

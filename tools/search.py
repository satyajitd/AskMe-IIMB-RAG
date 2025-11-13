import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

# Load environment variables
load_dotenv()

# Initialize the Tavily search tool
web_search_tool = TavilySearch(max_results=5, search_depth="basic")

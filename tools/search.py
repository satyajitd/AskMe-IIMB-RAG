from langchain_tavily import TavilySearch

# Initialize the Tavily search tool
web_search_tool = TavilySearch(max_results=5, search_depth="basic")

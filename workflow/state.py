from typing_extensions import TypedDict
from typing import List

class State(TypedDict):
    question : str
    generation : str
    web_search : str
    documents : List[str]
    web_search_docs : List[str]
    
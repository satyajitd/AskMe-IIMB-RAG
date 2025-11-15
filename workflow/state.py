from typing import List, Optional

from langchain_core.documents import Document
from typing_extensions import NotRequired, TypedDict


class State(TypedDict):
    question: str
    generation: NotRequired[str]
    web_search: NotRequired[str]
    documents: NotRequired[List[Document]]
    web_search_docs: NotRequired[List[Document]]
    
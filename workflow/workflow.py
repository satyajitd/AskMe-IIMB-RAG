from typing import Optional
import asyncio

from langchain_core.documents import Document
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from store.vectorstore import vector_store
from tools.search import web_search_tool

import utils.constants as constants
from workflow.state import State

from agents.router import RouterChain
from agents.generator import GeneratorChain
from agents.answer_grader import AnswerGraderChain

from utils.logger import configure_logger

class Workflow:

    def __init__(
        self,
        router: Optional[RouterChain] = None,
        generator: Optional[GeneratorChain] = None,
        answer_grader: Optional[AnswerGraderChain] = None,
        vector_store_instance=None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        # Initialize RAG components (allow dependency injection for testing)
        self.router = router or RouterChain()
        self.generator = generator or GeneratorChain()
        self.answer_grader = answer_grader or AnswerGraderChain()

        # Initialize vector store
        self.vector_store = vector_store_instance or vector_store

        # Configure logging
        self.configure_logging()

        # Compile LangGraph once for reuse
        self.graph = self._build_graph(checkpointer)

    def configure_logging(self):
        self.logger = configure_logger(self.__class__.__name__)

    # Methods representing graph nodes
    def retrieve(self, state: State) -> dict:
        """
        Retrieve documents from vector store based on the question.
        Args:
            state (dict): The current graph state.
        Returns:
            state (dict): Retrieved documents added to state.
        """
        try:
            question = state[constants.QUESTION]
            self.logger.info("Retrieving documents for question: %s", question)
            documents = self.vector_store.retrieve(question)
            return {constants.DOCUMENTS: documents, constants.QUESTION: question}
        except Exception as e:
            self.logger.error("Error retrieving documents: %s", e)
            return {constants.DOCUMENTS: [], constants.QUESTION: question}
        
    def generate(self, state: State) -> dict:
        """
        Generate answer using LLM based on retrieved documents.
        Args:
            state (dict): The current graph state.
        Returns:
            state (dict): New key added to state, generation, that contains the generated answer.
        """
        try:
            question = state[constants.QUESTION]
            documents = state[constants.DOCUMENTS]
            web_documents = state.get(constants.WEB_SEARCH_DOCS, [])

            # If web documents exist but aren't already included, append them (with simple de-dup by content)
            combined_docs = list(documents)
            if web_documents:
                seen = {getattr(d, "page_content", "") for d in combined_docs if isinstance(d, Document)}
                to_add = [d for d in web_documents if isinstance(d, Document) and getattr(d, "page_content", "") not in seen]
                if to_add:
                    combined_docs.extend(to_add)
                    self.logger.info("Appended %d web docs to context (total=%d)", len(to_add), len(combined_docs))
            
            attempts = int(state.get(constants.ATTEMPTS, 0)) + 1
            self.logger.info("Generating answer for question: %s", question)
            generation = self.generator.invoke({constants.CONTEXT: combined_docs, constants.QUESTION: question})
            return {constants.DOCUMENTS: combined_docs, constants.QUESTION: question, constants.GENERATION: generation, constants.ATTEMPTS: attempts}
        except Exception as e:
            self.logger.error("Error generating answer: %s", e)
            attempts = int(state.get(constants.ATTEMPTS, 0)) + 1
            return {constants.DOCUMENTS: state.get(constants.DOCUMENTS, []), constants.QUESTION: question, constants.GENERATION: "", constants.ATTEMPTS: attempts}

    # Methods representing decision nodes
    def route_question(self, state: State) -> str:
        """
        Route question to either off-topic handler or vector store based on router decision.
        Args:
            state (dict): The current graph state.
        Returns:
            str: Next node to route to, either "off_topic" or "vector_store"
        """
        question = state[constants.QUESTION]
        decision = (self.router.invoke({constants.QUESTION: question}) or constants.VECTOR_STORE).strip().lower()

        if decision == constants.OFF_TOPIC:
            self.logger.info("Router marked question as off-topic.")
            return constants.OFF_TOPIC

        if decision != constants.VECTOR_STORE:
            self.logger.warning("Unexpected router decision '%s', defaulting to vector_store", decision)
        else:
            self.logger.info("Routing question '%s' to vector store", question)

        return constants.VECTOR_STORE

    def decide_to_generate(self, state: State) -> str:
        """
        Decide whether to generate an answer or perform a web search.
        Only perform web search if no documents were retrieved.
        Args:
            state (dict): The current graph state.
        Returns:
            str: Next node to route to, either "generate" or "websearch"
        """
        question = state[constants.QUESTION]
        documents = state[constants.DOCUMENTS]
        self.logger.info("Deciding next step for question '%s' with %d documents.", question, len(documents))

        if len(documents) == 0:
            # No documents retrieved, perform web search as fallback
            self.logger.info("Decision: No documents retrieved, perform web search.")
            return constants.WEB_SEARCH
        else:
            # We have documents, generate answer (hallucination grader will verify quality)
            self.logger.info("Decision: Generate answer with %d documents.", len(documents))
            return constants.GENERATE

    def grade_answer(self, state: State) -> str:
        """
        Grade the generated answer for sufficiency/grounding and decide next step.
        If the answer is insufficient and we haven't done a web search yet,
        route to web search to augment context; otherwise request regeneration.
        """
        question = state[constants.QUESTION]
        documents = state[constants.DOCUMENTS]
        generation = state[constants.GENERATION]
        attempts = int(state.get(constants.ATTEMPTS, 0))
        self.logger.info("Grading answer sufficiency for question: %s", question)

        # Check answer sufficiency
        try:
            grade = self.answer_grader.invoke({constants.QUESTION: question, constants.DOCUMENTS: documents, constants.GENERATION: generation})
        except Exception:
            self.logger.warning("Answer grader failed, assuming supported.")
            return constants.SUPPORTED

        if not grade or grade.strip().lower() != constants.NO:
            self.logger.info("Answer is sufficient and grounded.")
            return constants.SUPPORTED

        max_attempts = constants.MAX_ATTEMPTS
        # Not grounded – first try web search if not done yet
        if state.get(constants.WEB_SEARCH) != constants.YES:
            self.logger.info("Answer insufficient (attempt %d); routing to web search.", attempts)
            return constants.WEB_SEARCH

        # Web search already done. If attempts < max_attempts, regenerate.
        if attempts < max_attempts:
            self.logger.info("Insufficient after web search (attempt %d < %d); regenerating.", attempts, max_attempts)
            return constants.NOT_SUPPORTED

        # Exceeded attempts – finalize.
        self.logger.info("Insufficient after %d attempts; stopping regeneration.", attempts)
        return constants.SUPPORTED

    def handle_off_topic(self, state: State) -> dict:
        """
        Handle off-topic queries that are not related to IIM Bangalore.
        Args:
            state (dict): The current graph state.
        Returns:
            state (dict): Updated state with polite off-topic message.
        """
        question = state[constants.QUESTION]
        self.logger.info("Handling off-topic question: %s", question)
        
        off_topic_response = (
            "I apologize, but I can only assist with queries related to IIM Bangalore. "
            "This includes information about faculties, student policies, course outlines, "
            "library resources, campus facilities, admissions, programs, and research activities. "
            "Please feel free to ask me anything about IIM Bangalore!"
        )
        
        return {
            constants.QUESTION: question,
            constants.GENERATION: off_topic_response,
            constants.DOCUMENTS: []
        }

    # Methods representing external tools
    async def web_search(self, state: State) -> dict:
        """
        Perform a web search based on the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended web results to documents and web_search_docs
        """
        question = state[constants.QUESTION]
        documents = state.get(constants.DOCUMENTS, [])
        documents_copy = list(documents) if documents else []
        self.logger.info("Performing web search for question: %s", question)

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, web_search_tool.invoke, question)
            self.logger.info("Web search response type: %s", type(response))
            
            web_search_docs = []
            # Tavily typically returns a list of dicts with content/title/url
            if isinstance(response, list):
                for item in response:
                    if isinstance(item, dict):
                        text = item.get(constants.CONTENT) or item.get("snippet") or ""
                        if not text:
                            continue
                        metadata = {
                            "source": "web",
                            "source_title": item.get("title"),
                            "source_url": item.get("url"),
                            "query": question,
                            "provider": "tavily",
                        }
                        web_search_docs.append(Document(page_content=text, metadata=metadata))
            elif isinstance(response, str):
                web_search_docs.append(
                    Document(page_content=response, metadata={"source": "web", "query": question, "provider": "tavily"})
                )
            else:
                web_search_docs.append(
                    Document(page_content=str(response), metadata={"source": "web", "query": question, "provider": "tavily"})
                )

            # Append web docs to current working context
            documents_copy.extend(web_search_docs)

            # Persist to vector store for continuous learning
            try:
                ingested = await loop.run_in_executor(None, self.vector_store.upsert_documents, web_search_docs)
                self.logger.info("Persisted %d web results to vector store", ingested)
            except Exception as persist_exc:
                self.logger.error("Failed to persist web results: %s", persist_exc)
                
        except Exception as e:
            self.logger.error("Error during web search: %s", e)
            # Return empty documents if web search fails
            web_search_docs = []
            
        return {
            constants.DOCUMENTS: documents_copy, 
            constants.QUESTION: question, 
            constants.WEB_SEARCH: constants.YES,
            constants.WEB_SEARCH_DOCS: web_search_docs
        }
    
    def _build_graph(self, checkpointer: Optional[BaseCheckpointSaver]) -> StateGraph:
        self.logger.info("Compiling graph workflow.")
        graph = StateGraph(State)
        graph.add_node(constants.OFF_TOPIC, self.handle_off_topic)
        graph.add_node(constants.WEB_SEARCH, self.web_search)
        graph.add_node(constants.RETRIEVE, self.retrieve)
        graph.add_node(constants.GENERATE, self.generate)

        graph.set_conditional_entry_point(
            self.route_question,
            {
                constants.OFF_TOPIC: constants.OFF_TOPIC,
                constants.VECTOR_STORE: constants.RETRIEVE
            },
        )

        graph.add_edge(constants.OFF_TOPIC, END)

        # Go directly from retrieve to decide_to_generate (skip grading)
        graph.add_conditional_edges(
            constants.RETRIEVE,
            self.decide_to_generate,
            {
                constants.WEB_SEARCH: constants.WEB_SEARCH,
                constants.GENERATE: constants.GENERATE
            },
        )
    
        graph.add_edge(constants.WEB_SEARCH, constants.GENERATE)
        graph.add_conditional_edges(
            constants.GENERATE,
            self.grade_answer,
            {
                constants.WEB_SEARCH: constants.WEB_SEARCH,
                constants.NOT_SUPPORTED: constants.GENERATE,
                constants.SUPPORTED: END,
            },
        )
        self.logger.info("Graph workflow defined successfully.")

        if checkpointer is not None:
            return graph.compile(checkpointer=checkpointer)
        return graph.compile()


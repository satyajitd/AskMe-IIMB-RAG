from typing import Optional

from langchain_core.documents import Document
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from store.vectorstore import vector_store
from tools.search import web_search_tool

import utils.constants as constants
from workflow.state import State

from agents.router import RouterChain
from agents.generator import GeneratorChain
from agents.hallucination_grader import HallucinationGraderChain

from utils.logger import configure_logger

class Workflow:

    def __init__(
        self,
        router: Optional[RouterChain] = None,
        generator: Optional[GeneratorChain] = None,
        hallucination_grader: Optional[HallucinationGraderChain] = None,
        vector_store_instance=None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        # Initialize RAG components (allow dependency injection for testing)
        self.router = router or RouterChain()
        self.generator = generator or GeneratorChain()
        self.hallucination_grader = hallucination_grader or HallucinationGraderChain()

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
            self.logger.info("Generating answer for question: %s", question)
            generation = self.generator.invoke({constants.CONTEXT: documents, constants.QUESTION: question})
            return {constants.DOCUMENTS: documents, constants.QUESTION: question, constants.GENERATION: generation}
        except Exception as e:
            self.logger.error("Error generating answer: %s", e)
            return {constants.DOCUMENTS: documents, constants.QUESTION: question, constants.GENERATION: ""}

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

    def grade_generation(self, state: State) -> str:
        """
        Grade the generated answer for hallucinations.
        Args:
            state (dict): The current graph state.
        Returns:
            str: "supported" if generation is grounded in documents, "not_supported" otherwise.
        """
        question = state[constants.QUESTION]
        documents = state[constants.DOCUMENTS]
        generation = state[constants.GENERATION]
        self.logger.info("Grading generation for hallucinations for question: %s", question)

        # Check hallucination
        try:
            grade = self.hallucination_grader.invoke({constants.DOCUMENTS: documents, constants.GENERATION: generation})
        except Exception:
            self.logger.warning("Hallucination grader failed, assuming supported.")
            return constants.SUPPORTED

        if not grade or grade.strip().lower() != constants.NO:
            self.logger.info("Generation is grounded in the documents.")
            return constants.SUPPORTED

        self.logger.info("Generation is not grounded in the documents, regenerating.")
        return constants.NOT_SUPPORTED

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
    def web_search(self, state: State) -> dict:
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
            response = web_search_tool.invoke(question)
            self.logger.info("Web search response type: %s", type(response))
            
            # TavilySearch returns a string with search results
            if isinstance(response, str):
                web_results = Document(page_content=response)
            elif isinstance(response, list):
                # If it returns a list, join the content
                web_results_text = "\n".join([
                    res.get(constants.CONTENT, str(res)) if isinstance(res, dict) else str(res) 
                    for res in response
                ])
                web_results = Document(page_content=web_results_text)
            else:
                web_results = Document(page_content=str(response))
            
            # Store web search results separately
            web_search_docs = [web_results]

            # Also append to documents for generation
            documents_copy.append(web_results)
                
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
            self.grade_generation,
            {
                constants.NOT_SUPPORTED: constants.GENERATE,
                constants.SUPPORTED: END,
            },
        )
        self.logger.info("Graph workflow defined successfully.")

        if checkpointer is not None:
            return graph.compile(checkpointer=checkpointer)
        return graph.compile()


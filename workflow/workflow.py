import json
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from store.vectorstore import vector_store
from tools.search import web_search_tool

import utils.constants as constants
from workflow.state import State

from agents.router import RouterChain
from agents.generator import GeneratorChain
from agents.answer_grader import AnswerGraderChain
from agents.retrieval_grader import RetrievalGraderChain
from agents.hallucination_grader import HallucinationGraderChain

import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

class Workflow:
    def __init__(self):

        # Initialize RAG components
        self.router = RouterChain()
        self.generator = GeneratorChain()
        self.answer_grader = AnswerGraderChain()
        self.retriever_grader = RetrievalGraderChain()
        self.hallucination_grader = HallucinationGraderChain()
        
        # Initialize vector store
        self.vector_store = vector_store

        # Initialize workflow
        self.workflow = StateGraph(State)

        # Configure logging
        self.configure_logging()

    def configure_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.log_dir = Path.cwd() / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"graph_{datetime.now().timestamp()}.log"

        # Configure handlers only if not already present to avoid duplicate logs
        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            # Stream handler (console)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
            # Optional file handler
            try:
                file_handler = RotatingFileHandler(self.log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
                file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.DEBUG)
                self.logger.addHandler(file_handler)
            except Exception:
                # If file handler can't be created, log a warning to console
                stream_handler.setLevel(logging.WARNING)
                self.logger.warning("Could not create log file handler at %s", self.log_file)
        self.logger.setLevel(level=logging.INFO)

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

    def grade_documents(self, state: State) -> dict:
        """
        Grade retrieved documents for relevance to the question.
        Args:
            state (dict): The current graph state.
        Returns:
            state (dict): Filtered documents and web search flag.
        """
        question = state[constants.QUESTION]
        documents = state[constants.DOCUMENTS]
        self.logger.info("Grading %d documents for question: %s", len(documents), question)

        # Score each retrieved document
        filtered_docs = []
        web_search = constants.NO
        for document in documents: 
            llm_response = self.retriever_grader.invoke({constants.QUESTION: question, constants.DOCUMENT: document})
            response = json.loads(llm_response)
            grade = response[constants.SCORE]
            if grade == constants.YES: # Relevant document
                self.logger.info("Document %s graded as relevant.", document)
                filtered_docs.append(document)
            else: # Not relevant document
                self.logger.info("Document %s graded as not relevant.", document)
                web_search = constants.YES
        
        return {constants.DOCUMENTS: filtered_docs, constants.QUESTION: question, constants.WEB_SEARCH: web_search}
    
    # Methods representing decision nodes
    def route_question(self, state: State) -> str:
        """
        Route question to either web search or vector store based on router decision.
        Args:
            state (dict): The current graph state.
        Returns:
            str: Next node to route to, either "web_search" or "vector_store"
        """
        question = state[constants.QUESTION]
        llm_response = self.router.invoke({constants.QUESTION: question})
        self.logger.info("Raw router response: %s", llm_response)
        
        try:
            source = json.loads(llm_response)
        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse router response as JSON: %s. Response: %s", e, llm_response)
            # Default to vectorstore on parse error
            return constants.VECTOR_STORE
        
        self.logger.info("Routing decision for question '%s': %s", question, source)
        
        datasource = source.get(constants.DATASOURCE, "").lower()
        if datasource == constants.WEB_SEARCH:
            self.logger.info("Routing to web search.")
            return constants.WEB_SEARCH
        elif datasource == constants.VECTOR_STORE or datasource == "vectorstore":
            self.logger.info("Routing to RAG.")
            return constants.VECTOR_STORE
        else:
            # Default to vectorstore if response is unexpected
            self.logger.warning("Unexpected datasource '%s', defaulting to vectorstore.", datasource)
            return constants.VECTOR_STORE

    def decide_to_generate(self, state: State) -> str:
        """
        Decide whether to generate an answer or perform a web search based on graded documents.
        Args:
            state (dict): The current graph state.
        Returns:
            str: Next node to route to, either "generate" or "websearch"
        """
        question = state[constants.QUESTION]
        web_search = state[constants.WEB_SEARCH]
        documents = state[constants.DOCUMENTS]
        self.logger.info("Deciding next step for question '%s' with %d relevant documents.", question, len(documents))

        if len(documents) == 0:
            # No relevant documents found, perform web search
            self.logger.info("Decision: No documents available, perform web search.")
            return constants.WEB_SEARCH
        elif web_search == constants.YES:
            # Documents were filtered out as irrelevant
            self.logger.info("Decision: Documents filtered as irrelevant, perform web search.")
            return constants.WEB_SEARCH
        else:
            # We have relevant documents, so generate answer
            self.logger.info("Decision: Generate answer with %d documents.", len(documents))
            return constants.GENERATE

    def grade_generation(self, state: State) -> str:
        """
        Grade the generated answer for hallucinations and relevance to the question.
        Args:
            state (dict): The current graph state.
        Returns:
            str: "useful" if generation is grounded and addresses the question, "not useful" otherwise.
        """
        question = state[constants.QUESTION]
        documents = state[constants.DOCUMENTS]
        generation = state[constants.GENERATION]
        self.logger.info("Grading generation for question: %s", question)

        # Check hallucination
        llm_response = self.hallucination_grader.invoke({constants.DOCUMENTS: documents, constants.GENERATION: generation})
        response = json.loads(llm_response)
        grade = response[constants.SCORE]
        if grade == constants.YES:
            self.logger.info("Generation is grounded in the documents.")
            # Check question-answering
            self.logger.info("Grading generation against question: %s", question)
            llm_response = self.answer_grader.invoke({constants.QUESTION: question, constants.GENERATION: generation})
            response = json.loads(llm_response)
            grade = response[constants.SCORE]
            if grade == constants.YES:
                self.logger.info("Generation addresses the question.")
                return constants.USEFUL
            else:
                self.logger.info("Generation does not address the question.")
                return constants.NOT_USEFUL
        else:
            self.logger.info("Generation is not grounded in the documents.")
        return constants.NOT_SUPPORTED

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
            if documents is not None and len(documents) > 0:
                documents.append(web_results)
            else:
                documents = [web_results]
                
        except Exception as e:
            self.logger.error("Error during web search: %s", e)
            # Return empty documents if web search fails
            documents = documents if documents else []
            web_search_docs = []
            
        return {
            constants.DOCUMENTS: documents, 
            constants.QUESTION: question, 
            constants.WEB_SEARCH: constants.YES,
            constants.WEB_SEARCH_DOCS: web_search_docs
        }
    
    def define_workflow(self) -> StateGraph:
        self.logger.info("Defining graph workflow.")
        self.workflow.add_node(constants.WEB_SEARCH, self.web_search)
        self.workflow.add_node(constants.RETRIEVE, self.retrieve)
        self.workflow.add_node(constants.GRADE_DOCUMENTS, self.grade_documents)
        self.workflow.add_node(constants.GENERATE, self.generate)

        self.workflow.set_conditional_entry_point(
            self.route_question,
            {
                constants.WEB_SEARCH: constants.WEB_SEARCH,
                constants.VECTOR_STORE: constants.RETRIEVE
            },
        )

        self.workflow.add_edge(constants.RETRIEVE, constants.GRADE_DOCUMENTS)
        self.workflow.add_conditional_edges(
            constants.GRADE_DOCUMENTS,
            self.decide_to_generate,
            {
                constants.WEB_SEARCH: constants.WEB_SEARCH,
                constants.GENERATE: constants.GENERATE
            },
        )
    
        self.workflow.add_edge(constants.WEB_SEARCH, constants.GENERATE)
        self.workflow.add_conditional_edges(
            constants.GENERATE,
            self.grade_generation,
            {
                constants.NOT_SUPPORTED: constants.GENERATE,
                constants.USEFUL: END,
                constants.NOT_USEFUL: constants.WEB_SEARCH,
            },
        )
        self.logger.info("Graph workflow defined successfully.")

        return self.workflow.compile()

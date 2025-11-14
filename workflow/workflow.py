import os
import json
import re
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from store.vectorstore import vector_store
from tools.search import web_search_tool

import utils.constants as constants
from workflow.state import State
from utils import env

from agents.router import RouterChain
from agents.generator import GeneratorChain
from agents.hallucination_grader import HallucinationGraderChain

import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

class Workflow:
    @staticmethod
    def extract_json_from_response(response: str) -> str:
        """
        Extract JSON from LLM response that may be wrapped in markdown code blocks.
        Handles cases where LLM adds extra text after the JSON.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Cleaned JSON string
        """
        if not response:
            return response
        
        # Try to match complete markdown code blocks: ```json ... ```
        # Use non-greedy match and stop at first closing backticks
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Handle incomplete code blocks (missing closing backticks): ```json ... (no closing)
        # This happens when LLM output is truncated or malformed
        incomplete_pattern = r'```(?:json)?\s*\n(.*?)$'
        incomplete_match = re.search(incomplete_pattern, response, re.DOTALL)
        if incomplete_match:
            # Extract content after opening backticks
            content = incomplete_match.group(1).strip()
            # Remove any trailing backticks that might be partial
            content = re.sub(r'`+$', '', content).strip()
            return content
        
        # Try to extract just the JSON object/array using braces
        # This handles cases where JSON is embedded in other text
        json_pattern = r'(\{.*?\}|\[.*?\])'
        json_match = re.search(json_pattern, response, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # Return original if no code blocks or JSON found
        return response.strip()
    
    @staticmethod
    def normalize_score(score_value) -> str:
        """
        Normalize score value to lowercase string 'yes' or 'no'.
        Handles int (0/1), string ('yes'/'no'), and other variations.
        
        Args:
            score_value: Score from LLM (can be int, str, or other)
            
        Returns:
            Normalized score as 'yes' or 'no'
        """
        if isinstance(score_value, int):
            return constants.YES if score_value == 1 else constants.NO
        elif isinstance(score_value, str):
            score_lower = score_value.lower().strip()
            # Handle variations: 'yes', 'no', '1', '0', 'true', 'false'
            if score_lower in ['yes', '1', 'true']:
                return constants.YES
            elif score_lower in ['no', '0', 'false']:
                return constants.NO
            return score_lower
        else:
            # Default to yes for unexpected types
            return constants.YES

    def __init__(self):
        # Initialize RAG components
        self.router = RouterChain()
        self.generator = GeneratorChain()
        self.hallucination_grader = HallucinationGraderChain()
        
        # Initialize vector store
        self.vector_store = vector_store

        # Initialize workflow
        self.workflow = StateGraph(State)

        # Configure logging
        self.configure_logging()

    def configure_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.log_file = os.getenv(env.APP_LOG)

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
        llm_response = self.router.invoke({constants.QUESTION: question})
        self.logger.info("Raw router response: '%s'", llm_response)
        
        try:
            # Handle empty or whitespace-only responses
            if not llm_response or not llm_response.strip():
                self.logger.warning("Empty response from router, defaulting to vectorstore.")
                return constants.VECTOR_STORE
            
            # Extract JSON from markdown code blocks if present
            cleaned_response = self.extract_json_from_response(llm_response)
            source = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse router response as JSON: %s. Response: '%s'", e, llm_response)
            # Default to vectorstore on parse error
            return constants.VECTOR_STORE
        
        self.logger.info("Routing decision for question '%s': %s", question, source)
        
        datasource = source.get(constants.DATASOURCE, "").lower()
        if datasource == constants.OFF_TOPIC or datasource == "off_topic":
            self.logger.info("Question is off-topic, not related to IIM Bangalore.")
            return constants.OFF_TOPIC
        elif datasource == constants.VECTOR_STORE or datasource == "vectorstore":
            self.logger.info("Routing to RAG.")
            return constants.VECTOR_STORE
        else:
            # Default to vectorstore if response is unexpected
            self.logger.warning("Unexpected datasource '%s', defaulting to vectorstore.", datasource)
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
            llm_response = self.hallucination_grader.invoke({constants.DOCUMENTS: documents, constants.GENERATION: generation})
            self.logger.info("Raw hallucination grader response: '%s'", llm_response)
            
            # Handle empty or whitespace-only responses
            if not llm_response or not llm_response.strip():
                self.logger.warning("Empty response from hallucination grader, assuming supported.")
                return constants.SUPPORTED
            
            # Extract JSON from markdown code blocks if present
            cleaned_response = self.extract_json_from_response(llm_response)
            response = json.loads(cleaned_response)
            grade = self.normalize_score(response.get(constants.SCORE, constants.YES))
            
            if grade == constants.YES:
                self.logger.info("Generation is grounded in the documents.")
                return constants.SUPPORTED
            else:
                self.logger.info("Generation is not grounded in the documents, regenerating.")
                return constants.NOT_SUPPORTED
        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse JSON from hallucination grader: %s. Response was: '%s'. Assuming supported.", e, llm_response)
            # Default to supported if we can't parse
            return constants.SUPPORTED
        except Exception as e:
            self.logger.error("Error grading generation: %s. Assuming supported.", e)
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
        self.workflow.add_node(constants.OFF_TOPIC, self.handle_off_topic)
        self.workflow.add_node(constants.WEB_SEARCH, self.web_search)
        self.workflow.add_node(constants.RETRIEVE, self.retrieve)
        self.workflow.add_node(constants.GENERATE, self.generate)

        self.workflow.set_conditional_entry_point(
            self.route_question,
            {
                constants.OFF_TOPIC: constants.OFF_TOPIC,
                constants.VECTOR_STORE: constants.RETRIEVE
            },
        )

        self.workflow.add_edge(constants.OFF_TOPIC, END)

        # Go directly from retrieve to decide_to_generate (skip grading)
        self.workflow.add_conditional_edges(
            constants.RETRIEVE,
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
                constants.SUPPORTED: END,
            },
        )
        self.logger.info("Graph workflow defined successfully.")

        return self.workflow.compile()

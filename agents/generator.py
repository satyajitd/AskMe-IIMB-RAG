import time
from langchain_core.prompts import PromptTemplate

from model.llm import LLM
from utils import constants

import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

class GeneratorChain:
    prompt = PromptTemplate(
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a knowledgeable assistant specializing in IIM Bangalore information. 
                Your role is to provide accurate, comprehensive, and well-structured answers about IIM Bangalore, including faculties, student policies, 
                course outlines, library resources, campus facilities, admissions, programs, and research activities.
                
                Instructions:
                - Use the retrieved context to provide detailed and informative answers
                - Structure your response clearly with relevant headings or bullet points when appropriate
                - For faculty queries: include their name, department, areas of expertise, and any relevant details
                - For policy queries: provide comprehensive policy details with specific rules and guidelines
                - For course queries: include course details, structure, duration, and key information
                - For library/facility queries: provide location, timings, resources, and access information
                - Be precise with dates, numbers, names, and specific requirements
                - If the context doesn't contain enough information to fully answer the question, acknowledge what you know and what information is missing
                - Maintain a professional and helpful tone appropriate for an educational institution
                
                Use the following pieces of retrieved context to answer the question accurately.
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                Question: {question} 
                Context: {context} 
                Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                input_variables=[constants.QUESTION, constants.CONTEXT]
            )
    
    def __init__(self):
        self.llm = LLM()
        self.configure_logging()

    def invoke(self, inputs: dict) -> str:
        try:
            start_time = time.time()
            prompt_text = GeneratorChain.prompt.format(**inputs)
            response = self.llm.invoke(prompt_text)
            end_time = time.time()
            self.logger.info(f"GeneratorChain invoked in {end_time - start_time:.2f} seconds.")
            self.logger.info(f"Generated answer for question '{inputs.get(constants.QUESTION)}' with context {inputs.get(constants.CONTEXT)}: {response.content}")
            return response.content
        except Exception as e:
            self.logger.error(f"Error during GeneratorChain invocation: {e}")
            raise
    
    def configure_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.log_dir = Path.cwd() / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"generator.log"

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

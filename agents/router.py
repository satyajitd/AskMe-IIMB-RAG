import os
import time
from langchain_core.prompts import PromptTemplate

from model.llm import LLM
from utils import constants, env

import logging
from logging.handlers import RotatingFileHandler

class RouterChain:
    prompt = PromptTemplate(
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
                user question to a vectorstore or marking it as off-topic. Use the vectorstore ONLY for questions related to IIM Bangalore, 
                including faculties, student policies, course outlines, library resources, campus facilities, admissions, 
                programs, research, and any other IIM Bangalore-specific information. 
                
                For questions NOT related to IIM Bangalore (such as general knowledge, other institutions, technical concepts like 
                prompt engineering, LLMs, or any other topics), return 'off_topic'. 
                
                Give a binary choice 'vectorstore' or 'off_topic' based on the question. Return a JSON with a single key 'datasource' and 
                no preamble or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                input_variables=[constants.QUESTION]
            )
    
    def __init__(self):
        self.llm = LLM()
        self.configure_logging()

    def invoke(self, inputs: dict) -> str:
        try:
            start_time = time.time()
            prompt_text = RouterChain.prompt.format(**inputs)
            response = self.llm.invoke(prompt_text)
            end_time = time.time()
            self.logger.info(f"RouterChain invoked in {end_time - start_time:.2f} seconds.")
            self.logger.info(f"Routing decision for {inputs.get(constants.QUESTION)}: {response.content}")
            return response.content
        except Exception as e:
            self.logger.error(f"Error during RouterChain invocation: {e}")
            raise
    
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

import os
import time
from langchain_core.prompts import PromptTemplate

from model.llm import LLM
from utils import constants, env

import logging
from logging.handlers import RotatingFileHandler

class HallucinationGraderChain:
    prompt = PromptTemplate(
                template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
                an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
                whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
                single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
                Here are the facts:
                \n ------- \n
                {documents} 
                \n ------- \n
                Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                input_variables=[constants.GENERATION, constants.DOCUMENTS]
            )
    
    def __init__(self):
        self.llm = LLM()
        self.configure_logging()

    def invoke(self, inputs: dict) -> str:
        try:
            start_time = time.time()
            prompt_text = HallucinationGraderChain.prompt.format(**inputs)
            response = self.llm.invoke(prompt_text)
            end_time = time.time()
            self.logger.info(f"HallucinationGraderChain invoked in {end_time - start_time:.2f} seconds.")
            self.logger.info(f"Grounding decision for generation '{inputs.get(constants.GENERATION)}' with documents {inputs.get(constants.DOCUMENTS)}: {response.content}")
            return response.content
        except Exception as e:
            self.logger.error(f"Error during HallucinationGraderChain invocation: {e}")
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

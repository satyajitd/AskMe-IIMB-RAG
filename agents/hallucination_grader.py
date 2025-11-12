import time
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm import LLM

import logging
from pathlib import Path
from datetime import datetime
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
                input_variables=["generation", "documents"]
            )
    
    def __init__(self):
        self.llm = LLM()
        self.output_parser = StrOutputParser()
        self.configure_logging()

    def invoke(self, inputs: dict) -> dict:
        try:
            start_time = time.time()
            prompt_text = HallucinationGraderChain.prompt.format(**inputs)
            llm_response = self.llm.generate(prompt_text)
            parsed_output = self.output_parser.parse(llm_response)
            end_time = time.time()
            self.logger.info(f"HallucinationGraderChain invoked in {end_time - start_time:.2f} seconds.")
            return parsed_output
        except Exception as e:
            self.logger.error(f"Error during HallucinationGraderChain invocation: {e}")
            return {"error": str(e)}
    
    def configure_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.log_dir = Path.cwd() / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"hallucination_grader_{datetime.now().timestamp()}.log"

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

__all__ = ["HallucinationGraderChain"]
import time
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from llm import LLM

import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

class RetrievalGraderChain:
    prompt = PromptTemplate(
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
                of a retrieved document to a user question. If the document contains keywords related to the user question, 
                grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
                Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                Here is the retrieved document: \n\n {document} \n\n
                Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """,
                input_variables=["question", "document"]
    )
    
    def __init__(self):
        self.llm = LLM()
        self.output_parser = JsonOutputParser()
        self.configure_logging()

    def invoke(self, inputs: dict) -> dict:
        try:
            start_time = time.time()
            prompt_text = RetrievalGraderChain.prompt.format(**inputs)
            llm_response = self.llm.generate(prompt_text)
            parsed_output = self.output_parser.parse(llm_response)
            end_time = time.time()
            self.logger.info(f"RetrievalGraderChain invoked in {end_time - start_time:.2f} seconds.")
            return parsed_output
        except Exception as e:
            self.logger.error(f"Error during RetrievalGraderChain invocation: {e}")
            return {"error": str(e)}
    
    def configure_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.log_dir = Path.cwd() / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"retrieval_grader_{datetime.now().timestamp()}.log"

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

__all__ = ["RetrievalGraderChain"]
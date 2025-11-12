import time
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm import LLM

import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

class GeneratorChain:
    prompt = PromptTemplate(
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
                Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
                Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
                Question: {question} 
                Context: {context} 
                Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                input_variables=["question", "document"]
            )
    
    def __init__(self):
        self.llm = LLM()
        self.output_parser = StrOutputParser()
        self.configure_logging()

    def invoke(self, inputs: dict) -> dict:
        try:
            start_time = time.time()
            prompt_text = GeneratorChain.prompt.format(**inputs)
            llm_response = self.llm.generate(prompt_text)
            parsed_output = self.output_parser.parse(llm_response)
            end_time = time.time()
            self.logger.info(f"GeneratorChain invoked in {end_time - start_time:.2f} seconds.")
            return parsed_output
        except Exception as e:
            self.logger.error(f"Error during GeneratorChain invocation: {e}")
            return {"error": str(e)}
    
    def configure_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.log_dir = Path.cwd() / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"generator_{datetime.now().timestamp()}.log"

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

__all__ = ["GeneratorChain"]
import time
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

from model.llm import LLM
from utils import constants

import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

class AnswerGraderChain:
    prompt = PromptTemplate(
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
                answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
                useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
                <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
                \n ------- \n
                {generation} 
                \n ------- \n
                Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                input_variables=[constants.GENERATION, constants.QUESTION]
            )
                
    def __init__(self):
        self.llm = LLM()
        self.configure_logging()

    def invoke(self, inputs: dict) -> dict:
        try:
            start_time = time.time()
            prompt_text = AnswerGraderChain.prompt.format(**inputs)
            response = self.llm.invoke(HumanMessage(content=prompt_text))
            end_time = time.time()
            self.logger.info(f"AnswerGraderChain invoked in {end_time - start_time:.2f} seconds.")
            return response.content
        except Exception as e:
            self.logger.error(f"Error during AnswerGraderChain invocation: {e}")
            return {"error": str(e)}
    
    def configure_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.log_dir = Path.cwd() / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"answer_grader_{datetime.now().timestamp()}.log"

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

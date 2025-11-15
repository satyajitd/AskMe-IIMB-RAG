import time
from typing import Dict

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from utils import constants


class RouteDecision(BaseModel):
    """Structured router response."""

    datasource: str = Field(description="Either 'vector_store' or 'off_topic'")


class RouterChain(BaseAgent):
    """Routes questions to vectorstore or off-topic sink using structured output."""

    _DEFAULT_DESTINATION = constants.VECTOR_STORE
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You route user queries to either the vectorstore (IIM Bangalore related)
                or mark them off-topic. Only treat questions about IIM Bangalore faculties, policies,
                courses, facilities, admissions, programs, or research as vectorstore candidates.
                Return a JSON with key 'datasource' and value 'vector_store' or 'off_topic'.""",
            ),
            (
                "human",
                "Question to route: {question}",
            ),
        ]
    )

    def __init__(self):
        super().__init__()

    def invoke(self, inputs: Dict[str, str]) -> str:
        start_time = time.perf_counter()
        llm_with_schema = self.llm.with_structured_output(RouteDecision)

        try:
            prompt_value = RouterChain.prompt.invoke(inputs)
            decision = llm_with_schema.invoke(prompt_value)
            elapsed = time.perf_counter() - start_time
            datasource = decision.datasource.strip().lower()

            allowed = {constants.VECTOR_STORE, constants.OFF_TOPIC}
            if datasource not in allowed:
                self.logger.warning("Unexpected datasource '%s', defaulting to %s", datasource, self._DEFAULT_DESTINATION)
                datasource = self._DEFAULT_DESTINATION

            self.logger.info(
                "Router decision",
                extra={
                    "question": inputs.get(constants.QUESTION),
                    "datasource": datasource,
                    "latency": round(elapsed, 3),
                },
            )
            return datasource
        except Exception as exc:
            self.logger.error("RouterChain failed: %s", exc, exc_info=True)
            return self._DEFAULT_DESTINATION

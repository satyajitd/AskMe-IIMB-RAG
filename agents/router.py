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
                """You are a routing assistant for IIM Bangalore queries. Route questions to the vectorstore if they are about:
                - IIM Bangalore faculty (directors, professors, staff, their expertise, contact info)
                - Student policies, rules, guidelines, code of conduct
                - Courses, programs, curriculum, MBA, Executive programs
                - Campus facilities (library, hostels, sports, dining, IT services)
                - Admissions, eligibility, application process, entrance exams
                - Research activities, centers, publications, projects
                - Any other IIM Bangalore institutional information
                
                Mark as off-topic ONLY if the question is clearly unrelated to IIM Bangalore (e.g., general knowledge, other universities, unrelated topics).
                
                Examples:
                - "Who is the director of IIM Bangalore?" -> vector_store
                - "What are the hostel facilities?" -> vector_store
                - "Tell me about faculty research" -> vector_store
                - "What is machine learning?" -> off_topic
                - "How to cook pasta?" -> off_topic
                
                Return JSON: {{"datasource": "vector_store"}} or {{"datasource": "off_topic"}}""",
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

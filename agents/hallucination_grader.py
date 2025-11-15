import time
from typing import Any, Dict, Iterable

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from utils import constants


class HallucinationScore(BaseModel):
    score: str = Field(description="'yes' if grounded, 'no' otherwise")


class HallucinationGraderChain(BaseAgent):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You check whether the provided answer is grounded in the supplied facts.
                Respond with JSON containing key 'score' and value 'yes' if the answer is supported,
                otherwise 'no'.""",
            ),
            (
                "human",
                "Facts:\n{documents}\n\nAnswer:\n{generation}",
            ),
        ]
    )

    def __init__(self) -> None:
        super().__init__()

    def invoke(self, inputs: Dict[str, Any]) -> str:
        start_time = time.perf_counter()
        documents = self._format_documents(inputs.get(constants.DOCUMENTS))
        generation = inputs.get(constants.GENERATION, "")

        llm_with_schema = self.llm.with_structured_output(HallucinationScore)

        try:
            prompt_value = HallucinationGraderChain.prompt.invoke(
                {constants.DOCUMENTS: documents, constants.GENERATION: generation}
            )
            result = llm_with_schema.invoke(prompt_value)
            latency = time.perf_counter() - start_time

            score = result.score.strip().lower()
            if score not in {constants.YES, constants.NO}:
                self.logger.warning("Unexpected hallucination score '%s', defaulting to yes", score)
                score = constants.YES

            self.logger.info(
                "Hallucination check",
                extra={"latency": round(latency, 3), "score": score},
            )
            return score
        except Exception as exc:
            self.logger.error("HallucinationGraderChain failed: %s", exc, exc_info=True)
            return constants.YES

    @staticmethod
    def _format_documents(documents: Any) -> str:
        if documents is None:
            return "No supporting documents were retrieved."
        if isinstance(documents, str):
            return documents
        if isinstance(documents, Document):
            return documents.page_content
        if isinstance(documents, Iterable):
            parts = []
            for doc in documents:
                if isinstance(doc, Document):
                    parts.append(doc.page_content)
                else:
                    parts.append(str(doc))
            return "\n---\n".join(parts) if parts else "No supporting documents were retrieved."
        return str(documents)

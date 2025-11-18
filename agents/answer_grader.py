import time
from typing import Any, Dict, Iterable

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from utils import constants


class AnswerScore(BaseModel):
    score: str = Field(description="'yes' if the answer sufficiently addresses the question based on the provided facts, 'no' otherwise")
    reason: str = Field(description="Brief justification for the decision")


class AnswerGraderChain(BaseAgent):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a strict answer quality grader.
                Determine whether the ANSWER sufficiently and directly addresses the QUESTION using the provided FACTS.

                Decision rules:
                - Return JSON only with keys 'score' and 'reason'.
                - 'score' must be 'yes' if the answer clearly answers the question using the facts (paraphrasing allowed) OR after web results were added; otherwise 'no'.
                - If the answer says it "doesn't have enough information", "not mentioned", "cannot determine", or otherwise avoids answering, respond 'no'.
                - If facts are empty or irrelevant to the question, respond 'no'.
                - Keep 'reason' concise (one sentence).

                Output strictly as a JSON object, with no extra text or code fences.""",
            ),
            (
                "human",
                "Question: {question}\nFacts (numbered):\n{documents}\n\nAnswer:\n{generation}",
            ),
        ]
    )

    def __init__(self) -> None:
        super().__init__()

    def invoke(self, inputs: Dict[str, Any]) -> str:
        start_time = time.perf_counter()
        question = inputs.get(constants.QUESTION, "")
        documents = self._format_documents(inputs.get(constants.DOCUMENTS))
        generation = inputs.get(constants.GENERATION, "")

        llm_with_schema = self.llm.with_structured_output(AnswerScore)

        try:
            prompt_value = AnswerGraderChain.prompt.invoke(
                {constants.QUESTION: question, constants.DOCUMENTS: documents, constants.GENERATION: generation}
            )
            result = llm_with_schema.invoke(prompt_value)
            latency = time.perf_counter() - start_time

            score = result.score.strip().lower()
            if score not in {constants.YES, constants.NO}:
                self.logger.warning("Unexpected answer score '%s', defaulting to no", score)
                score = constants.NO

            self.logger.info(
                "Answer quality check",
                extra={"latency": round(latency, 3), "score": score},
            )
            return score
        except Exception as exc:
            self.logger.error("AnswerGraderChain failed: %s", exc, exc_info=True)
            return constants.NO

    @staticmethod
    def _format_documents(documents: Any) -> str:
        if documents is None:
            return "[0] No supporting documents were retrieved."
        if isinstance(documents, str):
            return documents
        if isinstance(documents, Document):
            src = documents.metadata.get("source") if isinstance(documents.metadata, dict) else None
            prefix = "[1] "
            if src:
                prefix = f"[1] (source: {src}) "
            return f"{prefix}{documents.page_content}"
        if isinstance(documents, Iterable):
            parts = []
            idx = 1
            for doc in documents:
                if isinstance(doc, Document):
                    src = doc.metadata.get("source") if isinstance(doc.metadata, dict) else None
                    header = f"[{idx}] " if not src else f"[{idx}] (source: {src}) "
                    parts.append(f"{header}{doc.page_content}")
                else:
                    parts.append(f"[{idx}] {str(doc)}")
                idx += 1
            return "\n---\n".join(parts) if parts else "[0] No supporting documents were retrieved."
        return str(documents)

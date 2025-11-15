import time
from typing import Any, Dict, Iterable

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from utils import constants


class HallucinationScore(BaseModel):
    score: str = Field(description="'yes' if grounded, 'no' otherwise")
    reason: str = Field(description="Brief justification for the decision")


class HallucinationGraderChain(BaseAgent):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a rigorous grounding checker.
                Decide whether the ANSWER is fully supported by the provided FACTS.

                Rules:
                - Return JSON only with keys 'score' and 'reason'.
                - 'score' must be 'yes' if the answer is directly supported by the facts; otherwise 'no'.
                - If any part of the answer introduces information not present in the facts, respond 'no'.
                - Do not infer beyond the facts. Paraphrasing is fine if the meaning is preserved.
                - If facts are empty or irrelevant to the answer, respond 'no'.
                - Keep 'reason' concise (one sentence).

                Output strictly as a JSON object, with no extra text or code fences.""",
            ),
            (
                "human",
                "Facts (numbered):\n{documents}\n\nAnswer:\n{generation}",
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

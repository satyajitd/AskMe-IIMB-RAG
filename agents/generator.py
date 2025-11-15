import time
from typing import Any, Dict, Iterable

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from agents.base_agent import BaseAgent
from utils import constants


class GeneratorChain(BaseAgent):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a knowledgeable assistant specializing in IIM Bangalore.
                Follow these rules when answering:
                - Use only the provided Context; do not invent facts.
                - If the Context is insufficient or unrelated, say you don't have enough information.
                - Start with a brief Summary, then provide Details with bullets.
                - Cite using bracketed indices [n] that match the numbered Context items.
                - Add a final 'Sources' section listing the referenced [n] with any available source metadata.
                - Keep the tone concise, neutral, and helpful.""",
            ),
            (
                "human",
                "Question: {question}\nContext (numbered):\n{context}\n\nAnswer:",
            ),
        ]
    )

    def __init__(self) -> None:
        super().__init__()

    def invoke(self, inputs: Dict[str, Any]) -> str:
        start_time = time.perf_counter()
        question = inputs.get(constants.QUESTION, "")
        context = self._format_context(inputs.get(constants.CONTEXT))

        try:
            prompt_value = GeneratorChain.prompt.invoke(
                {constants.QUESTION: question, constants.CONTEXT: context}
            )
            response = self.llm.invoke(prompt_value)
            latency = time.perf_counter() - start_time

            content = getattr(response, "content", response)
            self.logger.info(
                "Generated response",
                extra={
                    "question": question,
                    "latency": round(latency, 3),
                    "context_preview": context[:200],
                },
            )
            return content
        except Exception as exc:
            self.logger.error("GeneratorChain failed: %s", exc, exc_info=True)
            raise

    @staticmethod
    def _format_context(context: Any) -> str:
        """Normalize various context payloads to a string."""

        if context is None:
            return "[0] No retrieved documents."
        if isinstance(context, str):
            return context
        if isinstance(context, Document):
            src = context.metadata.get("source") if isinstance(context.metadata, dict) else None
            prefix = "[1] "
            if src:
                prefix = f"[1] (source: {src}) "
            return f"{prefix}{context.page_content}"
        if isinstance(context, Iterable):
            parts = []
            idx = 1
            for chunk in context:
                if isinstance(chunk, Document):
                    src = chunk.metadata.get("source") if isinstance(chunk.metadata, dict) else None
                    header = f"[{idx}] " if not src else f"[{idx}] (source: {src}) "
                    parts.append(f"{header}{chunk.page_content}")
                else:
                    parts.append(f"[{idx}] {str(chunk)}")
                idx += 1
            return "\n\n".join(parts) if parts else "[0] No retrieved documents."
        return str(context)

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
                """You are a knowledgeable assistant specializing in IIM Bangalore. Provide accurate,
                comprehensive, well-structured answers based on the supplied context. Use headings/bullets
                when useful, cite concrete facts (names, dates, policies) and acknowledge gaps if context
                is insufficient.""",
            ),
            (
                "human",
                "Question: {question}\nContext:\n{context}\nAnswer:",
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
            return "No retrieved documents."
        if isinstance(context, str):
            return context
        if isinstance(context, Document):
            return context.page_content
        if isinstance(context, Iterable):
            parts = []
            for chunk in context:
                if isinstance(chunk, Document):
                    parts.append(chunk.page_content)
                else:
                    parts.append(str(chunk))
            return "\n\n".join(parts) if parts else "No retrieved documents."
        return str(context)

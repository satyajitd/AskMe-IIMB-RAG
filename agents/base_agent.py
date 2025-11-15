"""Shared functionality for all agent chains."""

from __future__ import annotations

import logging

from model.llm import LLM
from utils.logger import configure_logger


class BaseAgent:
    """Provides a configured LLM instance and consistent logging."""

    def __init__(self) -> None:
        self.llm = LLM()
        self.logger = self._configure_logger()

    def _configure_logger(self) -> logging.Logger:
        """Configure a class-specific logger if needed."""

        return configure_logger(self.__class__.__name__)

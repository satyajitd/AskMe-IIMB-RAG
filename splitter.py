import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Splitter(RecursiveCharacterTextSplitter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.configure_logging()

    def split_documents(self, documents) -> list[Document]:
        self.logger.info(f"Splitting {len(documents)} documents into smaller chunks.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=0,
            length_function=len,
        )
        split_documents = text_splitter.split_documents(documents)
        self.logger.info(f"Split into {len(split_documents)} chunks.")
        return split_documents

    def configure_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.log_dir = Path.cwd() / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"embeddings_{datetime.now().timestamp()}.log"

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


__all__ = ["Splitter"]

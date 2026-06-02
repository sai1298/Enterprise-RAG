import os
from typing import Dict, Any, Optional, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    CSVLoader,
)
from Models.Embeddings.embedding_model import EmbeddingModels


class DocumentProcessor:
    def __init__(
        self,
        chunking_strategy: str = "recursive",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.chunking_strategy = chunking_strategy
        self.config = config or {
            "chunk_size": 500,
            "chunk_overlap": 100,
        }

        self.loader_map = {
            ".pdf":  PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".xls":  UnstructuredExcelLoader,
            ".csv":  CSVLoader,
        }

    def _get_splitter(self):
        size = self.config.get("chunk_size", 500)
        overlap = self.config.get("chunk_overlap", 100)

        if self.chunking_strategy == "fixed":
            return CharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        elif self.chunking_strategy == "semantic":
            embeddings = EmbeddingModels.gemini_embedding_2_preview()
            return SemanticChunker(embeddings)
        else:
            return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)

    def load(self, file_path: str):
        """Load a file based on its extension."""
        ext = os.path.splitext(file_path)[-1].lower()
        loader_cls = self.loader_map.get(ext)

        if not loader_cls:
            raise ValueError(f"Unsupported file type: {ext}")

        loader = loader_cls(file_path)
        return loader.load()

    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """Load and chunk a file."""
        documents = self.load(file_path)
        splitter = self._get_splitter()

        chunks = splitter.split_documents(documents)

        return [
            {
                "text": chunk.page_content,
                "metadata": {
                    "chunk_index": i,
                    "chunk_strategy": self.chunking_strategy,
                    "source": file_path,
                    **chunk.metadata,
                },
            }
            for i, chunk in enumerate(chunks)
        ]


# if __name__ == "__main__":
#     processor = DocumentProcessor(chunking_strategy="semantic")
#     process = processor.process("RAG_files/SAIRAM_PURIMETLA_Resume.pdf")
#     print(process)
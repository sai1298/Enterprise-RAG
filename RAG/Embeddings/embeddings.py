from Models.Embeddings.embedding_model import EmbeddingModels
from typing import Dict, Any, Optional, List

class EmbeddingPipeline:
    def __init__(self):
        self.model = EmbeddingModels.gemini_embedding_2_preview()

    def embed(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        texts = [c["text"] for c in chunks]
        vectors = self.model.embed_documents(texts)  # batch call

        for chunk, vector in zip(chunks, vectors):
            chunk["embedding"] = vector

        return chunks
import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from Document_processing.document_processing import DocumentProcessor
from Models.Embeddings.embedding_model import EmbeddingModels
from VectorStore.store import VectorStore

class VectorIndex:
    """
    Orchestrates the full ingest pipeline:
        file → DocumentProcessor → embed → VectorStore.upsert

    Also handles:
        - Duplicate detection (SHA-256 of file content)
        - Manifest persistence  (.index_manifest.json)
        - Querying with auto-embed
    """

    MANIFEST_PATH = ".index_manifest.json"

    def __init__(
        self,
        index_name: str,
        chunking_strategy: str = "recursive",
        chunking_config: Optional[Dict[str, Any]] = None,
        dimension: int = 3072,
        namespace: str = "",
        api_key: Optional[str] = None,
    ):
        self.store = VectorStore(
            index_name=index_name,
            dimension=dimension,
            namespace=namespace,
            api_key=api_key,
        )

        self.chunking_strategy = chunking_strategy
        self.chunking_config   = chunking_config or {"chunk_size": 500, "chunk_overlap": 100}
        self.embedding_model   = EmbeddingModels.gemini_embedding_2_preview()
        self.manifest          = self._load_manifest()

    # ------------------------------------------------------------------ #
    #  Manifest                                                            #
    # ------------------------------------------------------------------ #

    def _load_manifest(self) -> Dict[str, str]:
        """Load persisted {file_path: sha256} map."""
        if os.path.exists(self.MANIFEST_PATH):
            with open(self.MANIFEST_PATH, "r") as f:
                return json.load(f)
        return {}

    def _save_manifest(self) -> None:
        with open(self.MANIFEST_PATH, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def _file_hash(self, file_path: str) -> str:
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha.update(block)
        return sha.hexdigest()

    def _is_already_indexed(self, file_path: str) -> bool:
        current_hash = self._file_hash(file_path)
        return self.manifest.get(file_path) == current_hash

    def _mark_indexed(self, file_path: str) -> None:
        self.manifest[file_path] = self._file_hash(file_path)
        self._save_manifest()

    # ------------------------------------------------------------------ #
    #  Embed                                                               #
    # ------------------------------------------------------------------ #

    def _embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        texts   = [c["text"] for c in chunks]
        vectors = self.embedding_model.embed_documents(texts)
        for chunk, vec in zip(chunks, vectors):
            chunk["embedding"] = vec
        return chunks

    # ------------------------------------------------------------------ #
    #  Build (Ingest)                                                      #
    # ------------------------------------------------------------------ #

    def add_file(self, file_path: str, force: bool = False) -> None:
        """
        Ingest a single file into the vector index.

        Args:
            file_path : Path to the document.
            force     : Re-index even if the file hash hasn't changed.
        """
        if not force and self._is_already_indexed(file_path):
            print(f"[VectorIndex] Skipping '{file_path}' — already indexed (hash match)")
            return

        print(f"[VectorIndex] Processing '{file_path}' ...")

        # 1. Load + chunk
        processor = DocumentProcessor(
            chunking_strategy=self.chunking_strategy,
            config=self.chunking_config,
        )
        chunks = processor.process(file_path)
        print(f"[VectorIndex]   → {len(chunks)} chunks")

        # 2. Embed
        chunks = self._embed_chunks(chunks)
        print(f"[VectorIndex]   → embeddings done")

        # 3. Upsert
        self.store.upsert(chunks)

        # 4. Update manifest
        self._mark_indexed(file_path)

    def add_files(self, file_paths: List[str], force: bool = False) -> None:
        """Ingest multiple files."""
        for path in file_paths:
            try:
                self.add_file(path, force=force)
            except Exception as e:
                print(f"[VectorIndex] ERROR on '{path}': {e}")

    def remove_file(self, file_path: str) -> None:
        """Delete all vectors + manifest entry for a file."""
        self.store.delete_by_source(file_path)
        self.manifest.pop(file_path, None)
        self._save_manifest()
        print(f"[VectorIndex] Removed '{file_path}' from index")

    # ------------------------------------------------------------------ #
    #  Query                                                               #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed a query string and return top-k matching chunks.

        Args:
            query         : Natural language question.
            top_k         : Number of results to return.
            source_filter : Restrict results to a specific file path.
        """
        query_vec = self.embedding_model.embed_query(query)

        filter_dict = {"source": source_filter} if source_filter else None

        results = self.store.query(
            query_vector=query_vec,
            top_k=top_k,
            filter=filter_dict,
        )

        return results

    def stats(self) -> Dict[str, Any]:
        return self.store.stats()

    def list_indexed_files(self) -> List[str]:
        return list(self.manifest.keys())


# ── Quick test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    idx = VectorIndex(
        index_name="rag-index",
        chunking_strategy="semantic",
    )

    # Ingest
    idx.add_files([
        "RAG_files/SAIRAM_PURIMETLA_Resume.pdf",
    ])

    # Query
    results = idx.search("What ML frameworks does Sairam know?", top_k=3)
    for r in results:
        print(f"[{r['score']}] {r['text'][:150]}\n")

    # Stats
    print(idx.stats())
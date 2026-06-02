import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec


class VectorStore:
    def __init__(
        self,
        index_name: str,
        dimension: int = 3072,
        metric: str = "cosine",
        namespace: str = "",
        api_key: Optional[str] = None,
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        self.index_name = index_name
        self.namespace = namespace
        self.dimension = dimension

        self._pc = Pinecone(api_key=api_key or os.environ["PINECONE_API_KEY"])

        existing = [i.name for i in self._pc.list_indexes()]
        if index_name not in existing:
            self._pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            print(f"[VectorStore] Created new index '{index_name}'")
        else:
            print(f"[VectorStore] Connected to existing index '{index_name}'")

        self.index = self._pc.Index(index_name)

    # ------------------------------------------------------------------ #
    #  Write                                                               #
    # ------------------------------------------------------------------ #

    def upsert(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """
        Upsert embedded chunks into Pinecone.

        Each chunk must have:
            chunk["text"]               : str
            chunk["embedding"]          : List[float]
            chunk["metadata"]           : dict  (source, chunk_index, chunk_strategy, ...)
        """
        if not chunks:
            print("[VectorStore] No chunks to upsert.")
            return

        records = []
        for c in chunks:
            meta = c["metadata"]

            # Pinecone metadata: only str / int / float / bool / list[str]
            pinecone_meta = {
                "text":           c["text"],
                "source":         meta.get("source", ""),
                "chunk_index":    meta.get("chunk_index", 0),
                "chunk_strategy": meta.get("chunk_strategy", ""),
            }

            # Forward any extra scalar metadata from the loader (page, row, etc.)
            for k, v in meta.items():
                if k not in pinecone_meta and isinstance(v, (str, int, float, bool)):
                    pinecone_meta[k] = v

            records.append(
                {
                    "id":       f"{meta.get('source', 'doc')}::{meta.get('chunk_index', 0)}",
                    "values":   c["embedding"],
                    "metadata": pinecone_meta,
                }
            )

        for i in range(0, len(records), batch_size):
            self.index.upsert(
                vectors=records[i : i + batch_size],
                namespace=self.namespace,
            )

        print(f"[VectorStore] Upserted {len(records)} vectors → '{self.index_name}'")

    # ------------------------------------------------------------------ #
    #  Read                                                                #
    # ------------------------------------------------------------------ #

    def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return top-k semantically similar chunks.

        filter examples:
            {"source": "resume.pdf"}
            {"source": {"$in": ["a.pdf", "b.pdf"]}}
        """
        response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
            filter=filter,
        )

        return [
            {
                "id":       m["id"],
                "score":    round(m["score"], 4),
                "text":     m["metadata"].pop("text", ""),
                "metadata": m["metadata"],
            }
            for m in response.get("matches", [])
        ]

    # ------------------------------------------------------------------ #
    #  Delete                                                              #
    # ------------------------------------------------------------------ #

    def delete_by_source(self, file_path: str) -> None:
        """Remove all vectors from a specific source file."""
        self.index.delete(
            filter={"source": file_path},
            namespace=self.namespace,
        )
        print(f"[VectorStore] Deleted vectors for source '{file_path}'")

    def delete_by_ids(self, ids: List[str]) -> None:
        self.index.delete(ids=ids, namespace=self.namespace)
        print(f"[VectorStore] Deleted {len(ids)} vectors by ID")

    def clear_namespace(self) -> None:
        """Wipe everything in the current namespace."""
        self.index.delete(delete_all=True, namespace=self.namespace)
        print(f"[VectorStore] Cleared namespace '{self.namespace}'")

    # ------------------------------------------------------------------ #
    #  Utils                                                               #
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict[str, Any]:
        return self.index.describe_index_stats()

    def exists(self, vector_id: str) -> bool:
        result = self.index.fetch(ids=[vector_id], namespace=self.namespace)
        return vector_id in result.get("vectors", {})
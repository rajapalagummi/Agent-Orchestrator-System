import json
import logging
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "agent_long_term_memory"

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_chroma():
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


class WorkingMemory:
    """Short-term memory scoped to a single task. In-memory dict, no external dependency."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    def init_task(self, task_id: str, question: str):
        self._store[task_id] = {
            "question": question,
            "plan": None,
            "completed_outputs": {},
            "errors": [],
            "started_at": time.time(),
        }

    def update(self, task_id: str, key: str, value):
        if task_id in self._store:
            self._store[task_id][key] = value

    def get(self, task_id: str) -> dict:
        return self._store.get(task_id, {})

    def clear(self, task_id: str):
        self._store.pop(task_id, None)


class LongTermMemory:
    """Semantic memory across tasks. Stores what worked, tool usage, and outcomes in ChromaDB."""

    def __init__(self):
        self.embedder = get_embedder()
        self.client = get_chroma()
        try:
            self.collection = self.client.get_collection(COLLECTION_NAME)
        except Exception:
            self.collection = self.client.create_collection(
                COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )

    def store_outcome(self, question: str, plan_summary: str, outcome: str, success: bool):
        emb = self.embedder.encode([question]).tolist()
        doc_id = f"outcome_{int(time.time() * 1000)}"
        self.collection.add(
            ids=[doc_id],
            embeddings=emb,
            documents=[plan_summary],
            metadatas=[{
                "question": question,
                "outcome": outcome[:500],
                "success": success,
                "timestamp": time.time(),
            }],
        )

    def retrieve_similar(self, question: str, n: int = 3) -> list[dict]:
        try:
            emb = self.embedder.encode([question]).tolist()
            count = self.collection.count()
            if count == 0:
                return []
            res = self.collection.query(
                query_embeddings=emb,
                n_results=min(n, count),
                include=["documents", "metadatas", "distances"],
            )
            results = []
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            ):
                results.append({
                    "plan_summary": doc,
                    "question": meta.get("question", ""),
                    "outcome": meta.get("outcome", ""),
                    "success": meta.get("success", False),
                    "similarity": round(max(0, 1 - dist), 3),
                })
            return results
        except Exception as e:
            logger.warning(f"Long-term memory retrieval failed: {e}")
            return []

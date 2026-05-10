"""
╔══════════════════════════════════════════════════════════════════╗
║  TYPE 5: SEMANTIC MEMORY — The Knowledge Encyclopedia           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  WHAT: Stores factual knowledge, relationships, and concepts    ║
║        with vector embeddings for semantic retrieval.            ║
║                                                                  ║
║  WHY:  "Python is a programming language" → agent knows this    ║
║        even if the user never explicitly said it. Knowledge     ║
║        that goes beyond the user's conversation.                 ║
║                                                                  ║
║  ANALOGY: Encyclopedia/knowledge graph — structured facts.      ║
║                                                                  ║
║  LIFESPAN: Persistent. Grows as the agent learns.               ║
║                                                                  ║
║  DIFFERENCE FROM EPISODIC:                                       ║
║  - Episodic: "On May 5, we used pandas to clean data"           ║
║  - Semantic: "pandas is a Python data manipulation library"     ║
║  Episodic = WHEN/WHERE it happened. Semantic = WHAT is true.    ║
║                                                                  ║
║  KEY FEATURE: VECTOR SIMILARITY SEARCH                          ║
║  Instead of exact keyword matching, semantic memory finds       ║
║  knowledge by MEANING. "happy" matches "joyful" because         ║
║  their embeddings are close in vector space.                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import json
import time
import numpy as np
from pathlib import Path


@dataclass
class KnowledgeNode:
    """A single piece of semantic knowledge."""
    id: str
    content: str                      # The factual content
    category: str = "general"         # domain, user, system, tool
    embedding: list[float] | None = None  # Vector representation
    confidence: float = 1.0           # How confident we are in this fact
    source: str = "inferred"          # conversation, document, tool, inferred
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    relations: list[str] = field(default_factory=list)  # IDs of related nodes


class SemanticMemory:
    """
    Semantic Memory: Vector-based knowledge store with similarity search.
    
    This is the most technically interesting memory type for your internship.
    It uses embeddings to store knowledge as vectors, enabling retrieval
    by MEANING rather than exact keywords.
    
    ARCHITECTURE:
    1. Knowledge comes in (from conversations, documents, tools)
    2. Text is converted to a vector embedding
    3. Stored in a vector index (FAISS, Pinecone, etc.)
    4. At retrieval time: query is embedded, nearest vectors found
    5. Most relevant knowledge is injected into the context window
    
    IMPLEMENTATIONS:
    - Simple: TF-IDF vectors + cosine similarity (this demo)
    - Production: Sentence transformers + FAISS/Pinecone/Weaviate
    - Advanced: Knowledge graphs (Neo4j) + embeddings
    
    🔑 OPTIMIZATION OPPORTUNITIES:
    - Embedding model choice (trade-off: quality vs speed vs cost)
    - Index structure (flat vs HNSW vs IVF for large-scale)
    - Chunking strategy (how to split documents into knowledge units)
    - Relevance scoring (combine semantic + recency + importance)
    - Knowledge graph relationships (not just similarity)
    """

    def __init__(self, embedding_dim: int = 128, storage_path: str = "semantic_store.json"):
        self._nodes: dict[str, KnowledgeNode] = {}
        self._embedding_dim = embedding_dim
        self._storage_path = Path(storage_path)
        self._vocab: dict[str, int] = {}  # For simple TF-IDF-like embedding
        self._next_id = 0

    # ── Embedding Generation ─────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """
        Generate a vector embedding for text.
        
        🔑 THIS IS A SIMPLIFIED VERSION for demonstration.
        
        In production, you would use:
        - sentence-transformers: model.encode("text") → 384/768-dim vector
        - OpenAI embeddings API: text-embedding-3-small → 1536-dim vector
        - Cohere embed: cohere.embed(texts=["text"]) → 1024-dim vector
        
        The quality of your embeddings directly determines the quality
        of your semantic retrieval. This is a KEY optimization area.
        
        For your internship project, consider comparing:
        1. Bag-of-words (this demo)
        2. TF-IDF 
        3. Sentence transformers (all-MiniLM-L6-v2)
        4. Domain-specific fine-tuned embeddings
        """
        # Simple bag-of-words embedding (for demo purposes)
        words = text.lower().split()
        
        # Build vocabulary on the fly
        for word in words:
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab)

        # Create sparse vector, then hash to fixed dimension
        vec = np.zeros(self._embedding_dim)
        for word in words:
            idx = self._vocab[word]
            # Use hashing to map vocab indices to fixed-dim vector
            hash_idx = idx % self._embedding_dim
            vec[hash_idx] += 1.0

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    # ── Core Operations ──────────────────────────────────────────────

    def add_knowledge(
        self,
        content: str,
        category: str = "general",
        confidence: float = 1.0,
        source: str = "inferred",
        relations: list[str] | None = None,
    ) -> KnowledgeNode:
        """
        Add a piece of knowledge to semantic memory.
        
        Knowledge can come from:
        - Conversations: "I work at TCS" → fact about the user
        - Documents: Extracted from uploaded PDFs/docs
        - Tools: API responses containing factual data
        - Inference: Agent deduces from multiple sources
        - Episodic consolidation: Patterns from past experiences
        """
        node_id = f"k_{self._next_id:04d}"
        self._next_id += 1

        embedding = self._embed(content)

        node = KnowledgeNode(
            id=node_id,
            content=content,
            category=category,
            embedding=embedding.tolist(),
            confidence=confidence,
            source=source,
            relations=relations or [],
        )
        self._nodes[node_id] = node
        return node

    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.1) -> list[tuple[KnowledgeNode, float]]:
        """
        Semantic search: find knowledge most relevant to the query.
        
        This is the CORE VALUE of semantic memory — retrieval by meaning.
        
        "What programming language does the user prefer?"
        → finds "User likes Python for ML projects" even though
          the query doesn't contain "Python" or "ML"
        
        🔑 OPTIMIZATION: In production, you'd use:
        - FAISS for fast approximate nearest neighbor search
        - HNSW index for million-scale knowledge bases
        - Hybrid search: combine vector similarity with keyword matching
        """
        query_embedding = self._embed(query)

        scored = []
        for node in self._nodes.values():
            if node.embedding is None:
                continue
            node_embedding = np.array(node.embedding)
            similarity = self._cosine_similarity(query_embedding, node_embedding)

            # Boost by confidence
            adjusted_score = similarity * node.confidence

            if adjusted_score >= min_similarity:
                scored.append((node, adjusted_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_related(self, node_id: str, depth: int = 1) -> list[KnowledgeNode]:
        """
        Traverse knowledge graph relationships.
        
        Example: "Python" → related to → ["FastAPI", "pandas", "ML"]
        
        This enables graph-based reasoning:
        "What tools work with Python?" → follow relations from Python node
        """
        if node_id not in self._nodes:
            return []

        visited = set()
        to_visit = [node_id]
        result = []

        for _ in range(depth):
            next_visit = []
            for nid in to_visit:
                if nid in visited:
                    continue
                visited.add(nid)
                node = self._nodes.get(nid)
                if node:
                    result.append(node)
                    next_visit.extend(node.relations)
            to_visit = next_visit

        return result[1:]  # Exclude the starting node

    def update_confidence(self, node_id: str, delta: float) -> None:
        """
        Adjust confidence in a piece of knowledge.
        
        Confidence increases when knowledge is confirmed.
        Confidence decreases when knowledge is contradicted.
        
        🔑 This is important for handling conflicting information:
        "User prefers Python" (confidence=0.9)
        Then user says "Actually I'm switching to Rust"
        → decrease Python confidence, add Rust with high confidence
        """
        if node_id in self._nodes:
            node = self._nodes[node_id]
            node.confidence = max(0.0, min(1.0, node.confidence + delta))
            node.updated_at = time.time()

    # ── Context Injection ────────────────────────────────────────────

    def get_context_for_prompt(self, query: str, max_items: int = 5) -> str:
        """
        Build semantic context for injection into the LLM prompt.
        
        Only the most relevant knowledge gets included — this is how
        you keep the context window efficient.
        """
        results = self.search(query, top_k=max_items)
        if not results:
            return ""

        lines = ["[Semantic Memory — Relevant Knowledge]:"]
        for node, score in results:
            lines.append(f"  • [{node.category}] {node.content} (relevance={score:.2f})")
        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────────

    def save(self) -> None:
        """Save to disk."""
        data = {}
        for nid, node in self._nodes.items():
            data[nid] = {
                "id": node.id, "content": node.content, "category": node.category,
                "embedding": node.embedding, "confidence": node.confidence,
                "source": node.source, "created_at": node.created_at,
                "updated_at": node.updated_at, "relations": node.relations,
            }
        extra = {"_vocab": self._vocab, "_next_id": self._next_id, "nodes": data}
        self._storage_path.write_text(json.dumps(extra, indent=2))

    def load(self) -> None:
        """Load from disk."""
        if self._storage_path.exists():
            extra = json.loads(self._storage_path.read_text())
            self._vocab = extra.get("_vocab", {})
            self._next_id = extra.get("_next_id", 0)
            for nid, ndata in extra.get("nodes", {}).items():
                self._nodes[nid] = KnowledgeNode(**ndata)

    @property
    def stats(self) -> dict:
        categories = {}
        for n in self._nodes.values():
            categories[n.category] = categories.get(n.category, 0) + 1
        return {
            "total_knowledge": len(self._nodes),
            "categories": categories,
            "vocab_size": len(self._vocab),
            "avg_confidence": (
                sum(n.confidence for n in self._nodes.values()) / len(self._nodes)
                if self._nodes else 0
            ),
        }


# ── Demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TYPE 5: SEMANTIC MEMORY DEMO")
    print("=" * 60)

    sm = SemanticMemory(embedding_dim=64)

    # Build a knowledge base
    print("\n📚 Building knowledge base:\n")

    knowledge = [
        ("Python is a high-level programming language used for ML and web development", "domain", 1.0),
        ("FastAPI is a modern Python web framework for building APIs", "domain", 0.95),
        ("PostgreSQL is a relational database with strong ACID compliance", "domain", 0.9),
        ("The user prefers Python for machine learning projects", "user", 0.85),
        ("The user is a summer intern at TCS Research", "user", 0.9),
        ("The project is about optimizing memory for agentic AI", "user", 0.95),
        ("Vector databases store data as high-dimensional embeddings", "domain", 0.9),
        ("FAISS is a library for efficient similarity search by Facebook", "domain", 0.85),
        ("Sentence transformers convert text to dense vector representations", "domain", 0.9),
        ("Redis is an in-memory key-value store used for caching", "domain", 0.8),
        ("LangChain provides memory modules for LLM-based agents", "domain", 0.85),
        ("The context window is the token limit for LLM input", "domain", 0.95),
    ]

    nodes = []
    for content, category, confidence in knowledge:
        node = sm.add_knowledge(content, category=category, confidence=confidence)
        nodes.append(node)
        print(f"  ✅ [{category:>6}] {content[:60]}...")

    # Add some relationships
    nodes[0].relations = [nodes[1].id, nodes[3].id]   # Python → FastAPI, user preference
    nodes[1].relations = [nodes[0].id]                  # FastAPI → Python
    nodes[6].relations = [nodes[7].id, nodes[8].id]    # Vector DBs → FAISS, Sentence transformers

    print(f"\n📊 Stats: {sm.stats}")

    # Semantic search demonstrations
    queries = [
        "What database should I use?",
        "How do embeddings work?",
        "What does the user work on?",
        "web framework for Python APIs",
    ]

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        results = sm.search(query, top_k=3)
        for node, score in results:
            print(f"  → [{score:.3f}] {node.content[:70]}...")

    # Graph traversal
    print(f"\n🕸️ Knowledge related to 'Vector databases':")
    related = sm.get_related(nodes[6].id, depth=1)
    for r in related:
        print(f"  → {r.content[:70]}...")

    # Context injection
    print("\n🪟 Context for prompt about 'optimize search performance':")
    print("-" * 50)
    print(sm.get_context_for_prompt("optimize search performance"))

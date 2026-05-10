"""
╔══════════════════════════════════════════════════════════════════╗
║  TYPE 2: LONG-TERM MEMORY (LTM) — Persistent Cross-Session     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  WHAT: Stores user preferences, facts, and knowledge that       ║
║        persist ACROSS sessions (conversations).                  ║
║                                                                  ║
║  WHY:  So the agent remembers "I prefer dark mode" or           ║
║        "My project uses FastAPI" even after you close the chat.  ║
║                                                                  ║
║  ANALOGY: Hard disk — slow to access, but persistent.           ║
║                                                                  ║
║  LIFESPAN: Across sessions. Survives restarts.                  ║
║                                                                  ║
║  KEY CHALLENGE: What to store vs what to forget? How to         ║
║  retrieve the RIGHT memories at the RIGHT time?                  ║
║                                                                  ║
║  STORAGE BACKENDS (production):                                  ║
║  - SQLite / PostgreSQL for structured data                      ║
║  - Redis for fast key-value access                              ║
║  - Vector DB (Pinecone, Weaviate) for semantic retrieval        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class MemoryEntry:
    """A single long-term memory record."""
    key: str                          # Unique identifier
    value: Any                        # The stored information
    category: str = "general"         # preference | fact | instruction | context
    importance: float = 0.5           # 0.0 to 1.0 — used for retrieval ranking
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    source: str = "conversation"      # Where this memory came from
    
    def touch(self):
        """Update access metadata when this memory is retrieved."""
        self.last_accessed = time.time()
        self.access_count += 1


class LongTermMemory:
    """
    Long-Term Memory: Persistent storage that survives across sessions.
    
    This implementation uses a JSON file as the backend (for simplicity).
    In production, you'd use a proper database.
    
    KEY FEATURES:
    1. Categorized storage (preferences, facts, instructions)
    2. Importance scoring for retrieval prioritization
    3. Decay mechanism — unused memories fade over time
    4. Deduplication — don't store the same fact twice
    5. Persistence — saves to disk, loads on startup
    
    🔑 OPTIMIZATION OPPORTUNITIES (for your internship):
    - Importance-based eviction (forget low-importance, rarely-used memories)
    - Consolidation (merge similar memories into one)
    - Conflict resolution (what if two memories contradict each other?)
    - Compression (summarize verbose memories)
    """

    def __init__(self, storage_path: str = "ltm_store.json", max_entries: int = 1000):
        self._storage_path = Path(storage_path)
        self._max_entries = max_entries
        self._memories: dict[str, MemoryEntry] = {}
        self._load()

    # ── Core Operations ──────────────────────────────────────────────

    def store(
        self,
        key: str,
        value: Any,
        category: str = "general",
        importance: float = 0.5,
        source: str = "conversation",
    ) -> None:
        """
        Store a memory. If the key exists, UPDATE it (don't duplicate).
        
        In a real system, you'd extract memories automatically from
        conversations using an LLM:
        
            "I prefer Python" → store("lang_pref", "Python", "preference", 0.8)
            "My deadline is Friday" → store("deadline", "Friday", "fact", 0.9)
        """
        if key in self._memories:
            # Update existing memory — preserve access history
            existing = self._memories[key]
            existing.value = value
            existing.importance = max(existing.importance, importance)
            existing.touch()
        else:
            self._memories[key] = MemoryEntry(
                key=key, value=value, category=category,
                importance=importance, source=source,
            )

        self._enforce_limits()
        self._save()

    def retrieve(self, key: str) -> Any | None:
        """Retrieve a specific memory by key."""
        if key in self._memories:
            self._memories[key].touch()
            return self._memories[key].value
        return None

    def search(self, query: str, category: str | None = None, top_k: int = 5) -> list[MemoryEntry]:
        """
        Search memories by keyword matching (simple version).
        
        In production, you'd use:
        - Vector similarity search (embed query, find nearest memories)
        - Full-text search (BM25, TF-IDF)
        - Graph traversal (follow relationships between memories)
        
        🔑 OPTIMIZATION: The retrieval strategy is crucial.
        Too many memories injected = wasted tokens.
        Too few = agent misses critical context.
        """
        results = []
        query_lower = query.lower()

        for entry in self._memories.values():
            if category and entry.category != category:
                continue

            # Simple relevance: keyword match in key or value
            value_str = str(entry.value).lower()
            key_str = entry.key.lower()

            if query_lower in value_str or query_lower in key_str:
                results.append(entry)

        # Sort by importance * recency
        results.sort(
            key=lambda e: e.importance * (1 / (1 + time.time() - e.last_accessed)),
            reverse=True,
        )
        return results[:top_k]

    def get_by_category(self, category: str) -> list[MemoryEntry]:
        """Get all memories of a specific category."""
        return [m for m in self._memories.values() if m.category == category]

    def forget(self, key: str) -> bool:
        """Explicitly forget a memory."""
        if key in self._memories:
            del self._memories[key]
            self._save()
            return True
        return False

    # ── Memory Optimization ──────────────────────────────────────────

    def decay(self, decay_rate: float = 0.01) -> int:
        """
        Apply time-based decay to all memories.
        
        Memories that haven't been accessed recently lose importance.
        When importance drops below a threshold, they get evicted.
        
        🔑 THIS IS A KEY OPTIMIZATION for your project:
        - What's the right decay rate?
        - Should all categories decay equally?
        - Can you make decay adaptive based on usage patterns?
        """
        evicted = 0
        threshold = 0.05  # Below this importance → forget

        to_remove = []
        for key, entry in self._memories.items():
            age_hours = (time.time() - entry.last_accessed) / 3600
            entry.importance *= (1 - decay_rate) ** age_hours

            if entry.importance < threshold:
                to_remove.append(key)

        for key in to_remove:
            del self._memories[key]
            evicted += 1

        if evicted:
            self._save()
        return evicted

    def consolidate(self) -> int:
        """
        Merge similar memories to reduce redundancy.
        
        Example:
          "user likes Python" + "user prefers Python for ML" 
          → "user prefers Python, especially for ML work"
        
        In production, you'd use an LLM to generate the merged memory.
        """
        # Simple demo: find memories with the same category and overlapping keys
        merged = 0
        seen_values = {}

        for key, entry in list(self._memories.items()):
            value_key = (entry.category, str(entry.value)[:50])
            if value_key in seen_values:
                # Merge into existing
                existing_key = seen_values[value_key]
                existing = self._memories[existing_key]
                existing.importance = max(existing.importance, entry.importance)
                existing.access_count += entry.access_count
                del self._memories[key]
                merged += 1
            else:
                seen_values[value_key] = key

        if merged:
            self._save()
        return merged

    def _enforce_limits(self) -> None:
        """Evict lowest-importance memories when over capacity."""
        if len(self._memories) > self._max_entries:
            sorted_memories = sorted(
                self._memories.items(),
                key=lambda x: x[1].importance,
            )
            to_remove = len(self._memories) - self._max_entries
            for key, _ in sorted_memories[:to_remove]:
                del self._memories[key]

    # ── Persistence ──────────────────────────────────────────────────

    def _save(self) -> None:
        """Save to JSON file."""
        data = {}
        for key, entry in self._memories.items():
            data[key] = asdict(entry)
        self._storage_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        """Load from JSON file."""
        if self._storage_path.exists():
            data = json.loads(self._storage_path.read_text())
            for key, entry_dict in data.items():
                self._memories[key] = MemoryEntry(**entry_dict)

    # ── Context Injection ────────────────────────────────────────────

    def get_context_for_prompt(self, query: str = "", max_tokens: int = 500) -> str:
        """
        Build a context string to inject into the LLM's prompt.
        
        This is the bridge between LTM and the context window.
        Only the most relevant memories get injected.
        """
        # Get relevant memories
        relevant = self.search(query, top_k=10) if query else list(self._memories.values())

        # Sort by importance
        relevant.sort(key=lambda m: m.importance, reverse=True)

        # Build context string, respecting token budget
        lines = ["[Long-term memory — known facts about the user]:"]
        char_budget = max_tokens * 4  # rough token-to-char conversion
        used = len(lines[0])

        for mem in relevant:
            line = f"- [{mem.category}] {mem.key}: {mem.value}"
            if used + len(line) > char_budget:
                break
            lines.append(line)
            used += len(line)

        return "\n".join(lines)

    @property
    def stats(self) -> dict:
        categories = {}
        for m in self._memories.values():
            categories[m.category] = categories.get(m.category, 0) + 1
        return {
            "total_memories": len(self._memories),
            "categories": categories,
            "avg_importance": (
                sum(m.importance for m in self._memories.values()) / len(self._memories)
                if self._memories else 0
            ),
        }

    def __repr__(self) -> str:
        return f"LongTermMemory(entries={len(self._memories)}, path={self._storage_path})"


# ── Demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TYPE 2: LONG-TERM MEMORY DEMO")
    print("=" * 60)

    # Clean up any previous demo file
    ltm_path = "/tmp/ltm_demo.json"
    if os.path.exists(ltm_path):
        os.remove(ltm_path)

    ltm = LongTermMemory(storage_path=ltm_path)

    # Simulate extracting memories from conversations
    print("\n📥 Storing memories extracted from conversations:\n")

    memories_to_store = [
        ("lang_preference",   "Python",                   "preference",   0.9),
        ("framework",         "FastAPI for backend",      "preference",   0.8),
        ("project_name",      "AgentMem Optimizer",       "fact",         0.7),
        ("deadline",          "August 15, 2026",          "fact",         0.95),
        ("mentor_name",       "Dr. Sharma",               "fact",         0.6),
        ("style_pref",        "Prefers concise answers",  "preference",   0.85),
        ("db_choice",         "PostgreSQL for production", "preference",  0.7),
        ("team_size",         "3 interns + 1 mentor",     "fact",         0.5),
    ]

    for key, value, category, importance in memories_to_store:
        ltm.store(key, value, category, importance)
        print(f"  ✅ [{category:>11}] {key}: {value}  (importance={importance})")

    print(f"\n📊 Stats: {ltm.stats}")

    # Search
    print("\n🔍 Searching for 'Python':")
    results = ltm.search("Python")
    for r in results:
        print(f"  → {r.key}: {r.value} (importance={r.importance:.2f})")

    # Get context for prompt injection
    print("\n🪟 Context to inject into LLM prompt:")
    print("-" * 50)
    print(ltm.get_context_for_prompt("What language should I use?"))

    # Show persistence
    print(f"\n💾 Saved to: {ltm_path}")
    print("  (This persists across sessions — restart and it's still there!)")

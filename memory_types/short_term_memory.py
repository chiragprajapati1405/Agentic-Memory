"""
╔══════════════════════════════════════════════════════════════════╗
║  TYPE 1: SHORT-TERM MEMORY (STM) — The Conversation Buffer     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  WHAT: Stores the current conversation (user msgs + agent        ║
║        responses) in a rolling buffer.                           ║
║                                                                  ║
║  WHY:  So the agent can understand follow-ups like "make it      ║
║        shorter" — it needs to know what "it" refers to.          ║
║                                                                  ║
║  ANALOGY: RAM in a computer — fast, active, volatile.            ║
║                                                                  ║
║  LIFESPAN: Single session only. Cleared when session ends.       ║
║                                                                  ║
║  KEY CHALLENGE: Context windows have token limits. When the      ║
║  buffer grows too large, older messages must be truncated or     ║
║  summarized. This is where OPTIMIZATION comes in.                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import time


@dataclass
class Message:
    """A single conversation message."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0  # approximate token count

    def __post_init__(self):
        # Rough approximation: 1 token ≈ 4 characters
        if self.token_count == 0:
            self.token_count = len(self.content) // 4 + 1


class ShortTermMemory:
    """
    Short-Term Memory: A rolling conversation buffer with token-aware management.
    
    This is the simplest but most fundamental memory type. Every LLM-based
    agent uses this, often without realizing it.
    
    OPTIMIZATION STRATEGIES implemented here:
    1. Token-based truncation (drop oldest messages when limit hit)
    2. Sliding window (keep last N turns)
    3. Summarization of old messages (compress, don't lose)
    
    In production, you'd also consider:
    - Priority-based eviction (keep important messages longer)
    - Semantic deduplication (don't store repeated info)
    """

    def __init__(self, max_tokens: int = 4096, max_turns: int | None = None):
        """
        Args:
            max_tokens: Maximum tokens to keep in buffer (simulates context window).
            max_turns: If set, also limit by number of conversation turns.
        """
        self._buffer: list[Message] = []
        self._max_tokens = max_tokens
        self._max_turns = max_turns
        self._total_tokens = 0
        self._evicted_summary: str = ""  # Summary of evicted messages

    def add(self, role: Literal["user", "assistant", "system"], content: str) -> None:
        """Add a message to the conversation buffer."""
        msg = Message(role=role, content=content)
        self._buffer.append(msg)
        self._total_tokens += msg.token_count
        self._enforce_limits()

    def _enforce_limits(self) -> None:
        """
        CORE OPTIMIZATION: Evict old messages when limits are exceeded.
        
        Strategy: Summarize evicted messages instead of just dropping them.
        This preserves key context while freeing tokens.
        """
        # Enforce turn limit
        if self._max_turns and len(self._buffer) > self._max_turns:
            evicted = self._buffer[:-self._max_turns]
            self._buffer = self._buffer[-self._max_turns:]
            self._summarize_evicted(evicted)

        # Enforce token limit
        while self._total_tokens > self._max_tokens and len(self._buffer) > 1:
            evicted_msg = self._buffer.pop(0)
            self._total_tokens -= evicted_msg.token_count
            self._summarize_evicted([evicted_msg])

    def _summarize_evicted(self, messages: list[Message]) -> None:
        """
        Instead of losing evicted messages entirely, create a compressed summary.
        
        In production, you'd call an LLM here to generate a proper summary.
        For this demo, we do a simple concatenation with truncation.
        
        🔑 THIS IS A KEY OPTIMIZATION POINT for your internship:
        - How do you summarize without losing critical information?
        - Can you use importance scores to decide what to keep?
        - Can you do hierarchical summarization (summary of summaries)?
        """
        for msg in messages:
            snippet = msg.content[:100] + ("..." if len(msg.content) > 100 else "")
            self._evicted_summary += f"[{msg.role}]: {snippet}\n"

        # Keep the summary itself from growing too large
        max_summary_chars = self._max_tokens  # rough limit
        if len(self._evicted_summary) > max_summary_chars:
            self._evicted_summary = self._evicted_summary[-max_summary_chars:]

    def get_context(self) -> list[dict]:
        """
        Build the context to inject into the LLM's context window.
        
        This is what gets sent to the model. The order matters:
        1. System message (if any)
        2. Summary of evicted history (compressed past)
        3. Recent messages (full detail)
        """
        context = []

        # Inject evicted summary as a system-level context hint
        if self._evicted_summary:
            context.append({
                "role": "system",
                "content": f"[Earlier conversation summary]:\n{self._evicted_summary.strip()}"
            })

        # Add current buffer messages
        for msg in self._buffer:
            context.append({"role": msg.role, "content": msg.content})

        return context

    def clear(self) -> None:
        """Clear the buffer (session end)."""
        self._buffer.clear()
        self._total_tokens = 0
        self._evicted_summary = ""

    @property
    def stats(self) -> dict:
        return {
            "messages_in_buffer": len(self._buffer),
            "total_tokens": self._total_tokens,
            "max_tokens": self._max_tokens,
            "utilization": f"{self._total_tokens / self._max_tokens * 100:.1f}%",
            "has_evicted_summary": bool(self._evicted_summary),
        }

    def __repr__(self) -> str:
        return f"ShortTermMemory(msgs={len(self._buffer)}, tokens={self._total_tokens}/{self._max_tokens})"


# ── Demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TYPE 1: SHORT-TERM MEMORY DEMO")
    print("=" * 60)

    # Create a small STM to see eviction in action
    stm = ShortTermMemory(max_tokens=200, max_turns=6)

    # Simulate a conversation
    conversation = [
        ("user", "Hi! I'm working on a Python project about memory systems."),
        ("assistant", "Great! Memory systems are a key part of agentic AI. What aspect interests you most?"),
        ("user", "I want to understand how agents remember things across conversations."),
        ("assistant", "That's long-term memory! There are several approaches: vector databases, key-value stores, and graph-based memory."),
        ("user", "Can you focus on vector databases?"),
        ("assistant", "Sure! Vector DBs store information as embeddings. When you need to recall something, you search by semantic similarity."),
        ("user", "How is that different from just saving everything to a file?"),
        ("assistant", "Files store exact text. Vector DBs store meaning — so 'happy' and 'joyful' would be close together in the vector space."),
        ("user", "Make it shorter."),  # <-- This requires STM to know what "it" refers to!
    ]

    print("\n📝 Simulating conversation with max_tokens=200:\n")
    for role, content in conversation:
        stm.add(role, content)
        print(f"  [{role:>9}]: {content[:60]}{'...' if len(content) > 60 else ''}")
        print(f"             {stm}")

    print(f"\n📊 Stats: {stm.stats}")

    print("\n🪟 Context window that would be sent to LLM:")
    print("-" * 50)
    for msg in stm.get_context():
        role = msg["role"]
        content = msg["content"][:80] + ("..." if len(msg["content"]) > 80 else "")
        print(f"  [{role:>9}]: {content}")

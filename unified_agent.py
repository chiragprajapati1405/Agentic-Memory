"""
╔══════════════════════════════════════════════════════════════════╗
║  UNIFIED AGENT — Combining All 5 Memory Types                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  This demonstrates how a real agent orchestrates all five        ║
║  memory systems together, deciding what to store where and      ║
║  what to retrieve when.                                          ║
║                                                                  ║
║  THE MEMORY LIFECYCLE:                                           ║
║  1. User sends a message                                         ║
║  2. Agent retrieves relevant context from all memory types      ║
║  3. Context is assembled and injected into the LLM prompt       ║
║  4. LLM generates a response                                    ║
║  5. Agent extracts new memories and stores them appropriately   ║
║  6. Working memory is updated with task progress                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import os
import time
from memory_types import (
    ShortTermMemory,
    LongTermMemory,
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
)


class UnifiedAgent:
    """
    An agent with all 5 memory types working together.
    
    This shows the ARCHITECTURE of how memory systems compose.
    In a real system, the LLM would be generating responses —
    here we simulate it to focus on the memory mechanics.
    
    MEMORY FLOW:
    ┌─────────────┐
    │ User Message │
    └──────┬──────┘
           │
    ┌──────▼──────────────────────────────────────┐
    │         RETRIEVAL PHASE                       │
    │                                               │
    │  STM  → recent conversation turns             │
    │  LTM  → user preferences & known facts        │
    │  WM   → current task state & progress         │
    │  EM   → relevant past experiences             │
    │  SM   → relevant domain knowledge             │
    │                                               │
    └──────┬──────────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────────┐
    │       CONTEXT ASSEMBLY                        │
    │  Combine all retrieved memories into a        │
    │  single prompt that fits the context window   │
    └──────┬──────────────────────────────────────┘
           │
    ┌──────▼──────┐
    │   LLM Call   │  (simulated in this demo)
    └──────┬──────┘
           │
    ┌──────▼──────────────────────────────────────┐
    │        STORAGE PHASE                          │
    │                                               │
    │  STM  → add this turn to conversation buffer  │
    │  LTM  → extract & store new user preferences  │
    │  WM   → update task progress                  │
    │  EM   → record this interaction as episode    │
    │  SM   → store any new facts learned           │
    │                                               │
    └─────────────────────────────────────────────┘
    """

    def __init__(self, user_id: str = "default", storage_dir: str = "/tmp/agent_memory"):
        self.user_id = user_id
        os.makedirs(storage_dir, exist_ok=True)

        # Initialize all 6 memory systems
        self.stm = ShortTermMemory(max_tokens=2048, max_turns=20)
        self.ltm = LongTermMemory(storage_path=f"{storage_dir}/ltm_{user_id}.json")
        self.wm = WorkingMemory(max_context_tokens=1024)
        self.em = EpisodicMemory(storage_path=f"{storage_dir}/episodic_{user_id}.json")
        self.sm = SemanticMemory(embedding_dim=64, storage_path=f"{storage_dir}/semantic_{user_id}.json")
        self.pm = ProceduralMemory(storage_path=f"{storage_dir}/procedural_{user_id}.json")

        self._turn_count = 0

    def build_prompt(self, user_message: str) -> str:
        """
        🔑 CORE METHOD: Assemble context from all memory types.
        
        This is where the magic happens — deciding what goes into
        the context window and in what order.
        
        TOKEN BUDGET ALLOCATION (example for 4096 token window):
        - System instructions: ~500 tokens (fixed)
        - Semantic memory:     ~300 tokens (relevant knowledge)
        - Long-term memory:    ~200 tokens (user preferences)
        - Episodic memory:     ~200 tokens (relevant past experiences)
        - Working memory:      ~300 tokens (current task state)
        - Short-term memory:   ~2000 tokens (recent conversation)
        - Current message:     ~496 tokens (user's input)
        
        🔑 OPTIMIZATION: These allocations should be DYNAMIC.
        If there's no active task, Working Memory gets 0 tokens.
        If the query is novel, Episodic Memory might get more.
        """
        sections = []

        # 1. System instructions (always present)
        sections.append(
            "You are a helpful AI assistant with persistent memory. "
            "Use the context provided to give personalized, informed responses."
        )

        # 2. Semantic memory — relevant domain knowledge
        sem_context = self.sm.get_context_for_prompt(user_message, max_items=3)
        if sem_context:
            sections.append(sem_context)

        # 3. Long-term memory — user preferences and facts
        ltm_context = self.ltm.get_context_for_prompt(user_message, max_tokens=200)
        if ltm_context:
            sections.append(ltm_context)

        # 4. Episodic memory — relevant past experiences
        ep_context = self.em.get_context_for_prompt(user_message, max_episodes=2)
        if ep_context:
            sections.append(ep_context)

        # 5. Procedural memory — known workflows and behavioral modes
        pm_context = self.pm.get_context_for_prompt(user_message)
        if pm_context:
            sections.append(pm_context)

        # 6. Working memory — current task state
        wm_context = self.wm.get_context()
        if "Goal:" in wm_context:  # Only include if there's an active task
            sections.append(wm_context)

        # 7. Short-term memory — recent conversation
        stm_context = self.stm.get_context()
        if stm_context:
            sections.append("[Recent Conversation]:")
            for msg in stm_context:
                sections.append(f"  {msg['role']}: {msg['content']}")

        # 7. Current message
        sections.append(f"\n[Current User Message]: {user_message}")

        return "\n\n".join(sections)

    def process_message(self, user_message: str) -> str:
        """
        Process a user message through the full memory pipeline.
        
        In a real system, steps 3 & 4 would involve calling an LLM.
        Here we simulate the response to focus on memory mechanics.
        """
        self._turn_count += 1

        # ── STEP 1: RETRIEVAL — Build context from all memories ──
        prompt = self.build_prompt(user_message)

        # ── STEP 2: STORE in STM — Record this turn ──
        self.stm.add("user", user_message)

        # ── STEP 3: "LLM CALL" — Simulated response ──
        response = self._simulate_response(user_message)

        # ── STEP 4: STORE in STM — Record agent response ──
        self.stm.add("assistant", response)

        # ── STEP 5: EXTRACT & STORE — Update other memories ──
        self._extract_and_store(user_message, response)

        return response

    def _simulate_response(self, user_message: str) -> str:
        """Simulate an LLM response (placeholder for actual LLM call)."""
        msg_lower = user_message.lower()

        if "preference" in msg_lower or "prefer" in msg_lower or "like" in msg_lower:
            return "I've noted your preference. I'll keep that in mind for future interactions."
        elif "deploy" in msg_lower:
            return "Based on past experience, I recommend running tests before deployment. Last time we skipped tests, the deployment failed."
        elif "remember" in msg_lower:
            return f"Let me check what I know... I have {self.ltm.stats['total_memories']} stored facts and {self.em.stats['total_episodes']} past experiences."
        else:
            return f"I understand your message about '{user_message[:50]}'. How can I help further?"

    def _extract_and_store(self, user_message: str, response: str) -> None:
        """
        Extract memories from the interaction and store them.
        
        🔑 In production, you'd use an LLM to extract:
        - User preferences ("I prefer X" → LTM)
        - Facts ("My project is about Y" → LTM + SM)
        - Task-relevant info → Working Memory
        
        This extraction step is a MAJOR optimization opportunity:
        - What's worth remembering?
        - At what confidence level?
        - In which memory system?
        """
        msg_lower = user_message.lower()

        # Simple rule-based extraction (LLM would do this in production)
        if "prefer" in msg_lower or "like" in msg_lower or "i use" in msg_lower:
            # Extract preference → Long-Term Memory
            self.ltm.store(
                key=f"pref_{self._turn_count}",
                value=user_message,
                category="preference",
                importance=0.8,
            )

        if any(w in msg_lower for w in ["is", "are", "means", "defined"]):
            # Potentially a factual statement → Semantic Memory
            self.sm.add_knowledge(
                content=user_message,
                category="user",
                confidence=0.7,
                source="conversation",
            )

        # Always record as an episode
        self.em.record(
            situation=f"Turn {self._turn_count} of conversation",
            task="Respond to user",
            action=f"Processed: {user_message[:80]}",
            result=f"Responded: {response[:80]}",
            outcome="success",
            tags=["conversation"],
        )

    def get_memory_dashboard(self) -> str:
        """Get a summary of all memory systems."""
        lines = [
            "╔══════════════════════════════════════════╗",
            "║       AGENT MEMORY DASHBOARD             ║",
            "╠══════════════════════════════════════════╣",
            f"║  👤 User: {self.user_id:<30}║",
            f"║  🔄 Turns: {self._turn_count:<29}║",
            "╠══════════════════════════════════════════╣",
        ]
        
        stm_stats = self.stm.stats
        lines.append(f"║  📱 STM:  {stm_stats['messages_in_buffer']} msgs, "
                      f"{stm_stats['utilization']} used     ║")
        
        ltm_stats = self.ltm.stats
        lines.append(f"║  💾 LTM:  {ltm_stats['total_memories']} memories stored"
                      f"{'':>14}║")
        
        wm_stats = self.wm.stats
        lines.append(f"║  📋 WM:   {'Active' if wm_stats['has_plan'] else 'Idle':<30}║")
        
        em_stats = self.em.stats
        lines.append(f"║  📖 EM:   {em_stats['total_episodes']} episodes recorded"
                      f"{'':>12}║")
        
        sm_stats = self.sm.stats
        lines.append(f"║  📚 SM:   {sm_stats['total_knowledge']} knowledge items"
                      f"{'':>13}║")
        
        pm_stats = self.pm.stats
        lines.append(f"║  ⚙️  PM:   {pm_stats['procedures']} procedures learned"
                      f"{'':>11}║")
        
        lines.append("╚══════════════════════════════════════════╝")
        return "\n".join(lines)


# ── Demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  UNIFIED AGENT — All 5 Memory Types Working Together")
    print("=" * 60)

    agent = UnifiedAgent(user_id="intern_001", storage_dir="/tmp/unified_agent_demo")

    # Seed some background knowledge
    agent.sm.add_knowledge("Python is excellent for rapid prototyping", "domain")
    agent.sm.add_knowledge("FAISS enables fast vector similarity search", "domain")
    agent.sm.add_knowledge("Agentic AI requires multiple memory systems", "domain")
    agent.ltm.store("org", "TCS Research", "fact", 0.9)

    # Simulate a conversation
    messages = [
        "Hi! I prefer using Python for all my projects.",
        "I'm working on optimizing memory systems for AI agents.",
        "What do you remember about me?",
        "Help me deploy the memory module to production.",
    ]

    print()
    for msg in messages:
        print(f"👤 User: {msg}")
        response = agent.process_message(msg)
        print(f"🤖 Agent: {response}")
        print()

    # Show the dashboard
    print(agent.get_memory_dashboard())

    # Show what the full prompt looks like
    print("\n🪟 FULL PROMPT (what the LLM sees):")
    print("=" * 60)
    prompt = agent.build_prompt("What database should I use for vector search?")
    print(prompt)

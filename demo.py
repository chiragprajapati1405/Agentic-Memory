#!/usr/bin/env python3
"""
Interactive demo of all 5 agentic memory types.
Run: python demo.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_header(title: str, char: str = "═") -> None:
    width = 65
    print(f"\n{'╔' + char * (width - 2) + '╗'}")
    print(f"║  {title:<{width - 4}}║")
    print(f"{'╚' + char * (width - 2) + '╝'}\n")


def print_section(title: str) -> None:
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print(f"{'─' * 65}")


def demo_1_short_term_memory():
    """Demonstrate Short-Term Memory (Conversation Buffer)."""
    print_header("TYPE 1: SHORT-TERM MEMORY (STM)")
    
    print("""  📌 CONCEPT: STM is the conversation history buffer.
  It stores recent user↔agent messages in a rolling window.
  When the buffer exceeds the token limit, old messages
  are summarized and dropped.
  
  Think of it like RAM — fast, active, but volatile.""")

    from memory_types import ShortTermMemory

    stm = ShortTermMemory(max_tokens=300, max_turns=6)

    conversation = [
        ("user", "I need help with my Python project about memory optimization."),
        ("assistant", "I'd be happy to help! What specific aspect of memory optimization are you working on?"),
        ("user", "I'm trying to reduce the context window usage in my LLM agent."),
        ("assistant", "Great focus area! You can try: 1) Summarizing old messages, 2) Semantic compression, 3) Priority-based eviction."),
        ("user", "Tell me more about semantic compression."),
        ("assistant", "Semantic compression uses embeddings to represent messages as dense vectors, keeping meaning while reducing tokens."),
        ("user", "Can you show me code for that?"),
        ("assistant", "Sure! Here's a basic approach using sentence-transformers..."),
        ("user", "Make it shorter."),  # Requires STM context!
    ]

    print_section("Simulating conversation (max_tokens=300)")
    for role, content in conversation:
        stm.add(role, content)
        print(f"  [{role:>9}]: {content[:55]}{'...' if len(content) > 55 else ''}")
        print(f"  {'':>11}{stm}")

    print_section("Context window (what the LLM actually sees)")
    for msg in stm.get_context():
        content = msg['content'][:70] + ('...' if len(msg['content']) > 70 else '')
        print(f"  [{msg['role']:>9}]: {content}")

    print(f"\n  📊 Stats: {stm.stats}")
    print(f"\n  💡 KEY INSIGHT: Notice how old messages were evicted but")
    print(f"     summarized — the agent still has context about them!")


def demo_2_long_term_memory():
    """Demonstrate Long-Term Memory (Persistent Preferences)."""
    print_header("TYPE 2: LONG-TERM MEMORY (LTM)")

    print("""  📌 CONCEPT: LTM persists across sessions. It stores user
  preferences, facts, and learned information on disk.
  The agent remembers "you prefer Python" even after restart.
  
  Think of it like a Hard Disk — slow but persistent.""")

    from memory_types import LongTermMemory

    path = "/tmp/demo_ltm.json"
    if os.path.exists(path):
        os.remove(path)

    ltm = LongTermMemory(storage_path=path)

    # Store memories
    print_section("Storing extracted memories")
    memories = [
        ("language", "Python", "preference", 0.9),
        ("framework", "FastAPI", "preference", 0.8),
        ("project", "Agentic memory optimization for AI agents", "fact", 0.95),
        ("style", "Prefers concise code with comments", "preference", 0.85),
        ("db", "Uses PostgreSQL for structured data", "preference", 0.7),
    ]
    for key, val, cat, imp in memories:
        ltm.store(key, val, cat, imp)
        print(f"  ✅ [{cat:>10}] {key}: {val} (importance={imp})")

    # Search
    print_section("Searching for 'database'")
    results = ltm.search("database")
    for r in results:
        print(f"  → {r.key}: {r.value} (importance={r.importance:.2f})")

    # Context injection
    print_section("Context for LLM prompt")
    print(ltm.get_context_for_prompt("What tools should I use?"))

    print(f"\n  💡 KEY INSIGHT: LTM bridges sessions. Close this program,")
    print(f"     run it again, and these memories are still there!")


def demo_3_working_memory():
    """Demonstrate Working Memory (Task Scratchpad)."""
    print_header("TYPE 3: WORKING MEMORY (WM)")

    print("""  📌 CONCEPT: WM is the agent's internal scratchpad while
  working on a multi-step task. It tracks the plan, stores
  intermediate results, and keeps notes.
  
  Think of it like a Whiteboard — active during work, erased after.
  
  DIFFERENCE FROM STM:
  • STM = conversation history (visible to user)
  • WM  = agent's internal state (invisible to user)""")

    from memory_types import WorkingMemory

    wm = WorkingMemory(max_context_tokens=800)

    print_section("Agent receives: 'Analyze my dataset and create a report'")

    # Create plan
    wm.set_plan(
        goal="Analyze uploaded dataset and generate a summary report",
        steps=[
            "Load and validate the CSV file",
            "Compute descriptive statistics",
            "Identify outliers and missing values",
            "Generate visualizations",
            "Compile findings into a report",
        ]
    )

    # Execute steps
    steps_execution = [
        ("csv_loaded", {"rows": 1523, "columns": 8, "size_mb": 2.4},
         "CSV loaded. 1523 rows, 8 columns, no encoding issues."),
        ("statistics", {"mean_revenue": 45230, "median": 38500, "std": 12400},
         "Revenue column has high variance. Possible outliers."),
        ("outliers", {"count": 23, "pct": "1.5%", "action": "flagged"},
         "23 outliers found (1.5%). Flagged but not removed."),
    ]

    for name, output, note in steps_execution:
        step = wm.get_current_step()
        print(f"\n  ▶ Step: {step}")
        wm.store_result(name, output)
        wm.add_note(note)
        wm.advance_step()
        print(f"    Result: {output}")
        print(f"    Note: {note}")

    print_section("Working Memory state (injected into LLM)")
    print(wm.get_context())

    print(f"\n  💡 KEY INSIGHT: WM lets the agent carry state between steps.")
    print(f"     Step 3 knows what Step 1 found, without re-processing!")


def demo_4_episodic_memory():
    """Demonstrate Episodic Memory (Experience Diary)."""
    print_header("TYPE 4: EPISODIC MEMORY (EM)")

    print("""  📌 CONCEPT: EM records specific past experiences as timestamped
  episodes. It enables learning from past successes and failures.
  
  DIFFERENCE FROM LTM:
  • LTM: "User prefers Python" (timeless fact)
  • EM:  "On May 5, we deployed with Python and it took 3 hours
         because of dependency issues" (specific experience)
  
  Think of it like a Personal Diary — what happened, when, why.""")

    from memory_types import EpisodicMemory

    path = "/tmp/demo_episodic.json"
    if os.path.exists(path):
        os.remove(path)

    em = EpisodicMemory(storage_path=path)

    # Record episodes
    print_section("Recording past experiences")
    episodes = [
        ("User asked to optimize query", "Optimize SQL query", "Added database index",
         "Query sped up 50x", "success", ["database", "optimization"],
         "Indexing is the first optimization to try"),
        ("User asked to fix memory leak", "Debug memory issue", "Used tracemalloc profiler",
         "Found unbounded cache growing per request", "success", ["debugging", "memory"],
         "Always check caches for unbounded growth"),
        ("User asked to deploy without tests", "Deploy to production", "Deployed directly",
         "3 endpoints broken in production", "failure", ["deployment"],
         "Never deploy without running tests"),
        ("User asked to deploy with tests", "Deploy to production", "Ran tests then deployed",
         "Clean deployment, zero issues", "success", ["deployment"],
         "Test-first deployment works reliably"),
    ]

    for sit, task, action, result, outcome, tags, lesson in episodes:
        icon = "✅" if outcome == "success" else "❌"
        em.record(sit, task, action, result, outcome, tags, lesson)
        print(f"  {icon} [{outcome:>7}] {task}: {lesson}")

    # Recall
    print_section("Recall similar to 'deploy my application'")
    similar = em.recall_similar("deploy my application")
    for ep in similar:
        icon = "✅" if ep.outcome == "success" else "❌"
        print(f"  {icon} {ep.task} → {ep.lesson}")

    # Pattern extraction
    print_section("Extracted patterns")
    patterns = em.extract_patterns()
    print(f"  Success rate: {patterns['success_rate']:.0%}")
    print(f"  Lessons: {patterns['lessons_learned']}")

    # Consolidation → semantic knowledge
    print_section("Consolidation: Episodes → Semantic Knowledge")
    facts = em.consolidate_to_semantic()
    for f in facts:
        print(f"  📚 [{f['topic']}] {f['recommendation']}")

    print(f"\n  💡 KEY INSIGHT: Episodic → Semantic consolidation is how")
    print(f"     agents truly LEARN. Specific experiences become general rules!")


def demo_5_semantic_memory():
    """Demonstrate Semantic Memory (Knowledge + Vector Search)."""
    print_header("TYPE 5: SEMANTIC MEMORY (SM)")

    print("""  📌 CONCEPT: SM stores factual knowledge as vector embeddings.
  Retrieval works by MEANING, not keywords.
  
  "What database should I use?" matches "PostgreSQL is great for
  structured data" because the vectors are close in semantic space.
  
  This is the most technically interesting type for optimization:
  • Embedding model quality → retrieval quality
  • Index structure → search speed at scale
  • Chunking strategy → what gets stored as one unit""")

    from memory_types import SemanticMemory

    sm = SemanticMemory(embedding_dim=64)

    # Build knowledge base
    print_section("Building knowledge base")
    entries = [
        ("Python is the most popular language for machine learning", "domain"),
        ("FastAPI is a high-performance Python web framework", "domain"),
        ("PostgreSQL handles complex queries with ACID compliance", "domain"),
        ("FAISS library enables billion-scale similarity search", "domain"),
        ("Redis provides sub-millisecond key-value data access", "domain"),
        ("Sentence transformers create dense text embeddings", "domain"),
        ("The user's project focuses on agentic memory optimization", "user"),
        ("Vector databases are essential for semantic retrieval", "domain"),
    ]

    for content, cat in entries:
        sm.add_knowledge(content, category=cat)
        print(f"  ✅ [{cat:>6}] {content[:55]}...")

    # Semantic search — the core feature
    print_section("Semantic Search (retrieval by meaning)")
    queries = [
        "fast database for caching",
        "how to search through embeddings",
        "what is the user working on",
    ]
    for q in queries:
        print(f"\n  🔍 Query: '{q}'")
        results = sm.search(q, top_k=3)
        for node, score in results:
            bar = "█" * int(score * 20)
            print(f"     [{score:.3f}] {bar} {node.content[:50]}...")

    print(f"\n  💡 KEY INSIGHT: 'fast database for caching' matched Redis")
    print(f"     even though the query didn't contain 'Redis'!")
    print(f"     That's the power of semantic/vector search.")


def demo_unified():
    """Show all 5 memories working together."""
    print_header("UNIFIED: All 5 Memories Working Together")

    from unified_agent import UnifiedAgent

    agent = UnifiedAgent(user_id="demo_user", storage_dir="/tmp/demo_unified")

    # Seed knowledge
    agent.sm.add_knowledge("Python is excellent for prototyping", "domain")
    agent.sm.add_knowledge("Always run tests before deployment", "domain")
    agent.ltm.store("org", "Personal AI Research", "fact", 0.9)

    messages = [
        "Hi! I prefer Python for backend development.",
        "What do you remember about me?",
        "Help me deploy my application.",
    ]

    for msg in messages:
        print(f"\n  👤 User: {msg}")
        response = agent.process_message(msg)
        print(f"  🤖 Agent: {response}")

    print(f"\n{agent.get_memory_dashboard()}")

    print_section("Full LLM Prompt (assembled from all 5 memories)")
    prompt = agent.build_prompt("What tools should I use for my project?")
    # Show just first 600 chars to keep output manageable
    print(prompt[:800])
    if len(prompt) > 800:
        print(f"\n  ... [{len(prompt) - 800} more characters]")


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🧠 SIX TYPES OF AGENTIC AI MEMORY                         ║
    ║      Implementation Guide for Agentic AI Applications        ║
    ║                                                              ║
    ║   Project: Optimizing Memory for Agentic AI Systems          ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║   1. Short-Term Memory  — Conversation buffer (RAM)          ║
    ║   2. Long-Term Memory   — Persistent facts (Hard Disk)       ║
    ║   3. Working Memory     — Task scratchpad (Whiteboard)       ║
    ║   4. Episodic Memory    — Experience diary (Personal Log)    ║
    ║   5. Semantic Memory    — Knowledge + vectors (Encyclopedia) ║
    ║   6. Unified Agent      — All 5 working together             ║
    ║   7. Run ALL demos                                           ║
    ║   0. Exit                                                    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    demos = {
        "1": demo_1_short_term_memory,
        "2": demo_2_long_term_memory,
        "3": demo_3_working_memory,
        "4": demo_4_episodic_memory,
        "5": demo_5_semantic_memory,
        "6": demo_unified,
    }

    while True:
        choice = input("\n  Select demo (1-7, 0 to exit): ").strip()

        if choice == "0":
            print("\n  👋 Happy coding!")
            break
        elif choice == "7":
            for demo_fn in demos.values():
                demo_fn()
        elif choice in demos:
            demos[choice]()
        else:
            print("  Invalid choice. Try 1-7 or 0.")


if __name__ == "__main__":
    # If run non-interactively, run all demos
    if not sys.stdin.isatty():
        for name, fn in [
            ("STM", demo_1_short_term_memory),
            ("LTM", demo_2_long_term_memory),
            ("WM", demo_3_working_memory),
            ("EM", demo_4_episodic_memory),
            ("SM", demo_5_semantic_memory),
            ("Unified", demo_unified),
        ]:
            fn()
    else:
        main()

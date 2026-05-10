# 🧠 Six Types of Agentic AI Memory — Implementation Guide

This project implements all **six types of memory** used in Agentic AI systems,
inspired by cognitive science and adapted for LLM-powered agents.

---

## The Six Memory Types

| # | Memory Type | Analogy | What It Stores | Lifespan |
|---|------------|---------|----------------|----------|
| 1 | **Short-Term Memory (STM)** | RAM | Current conversation turns | Single session |
| 2 | **Long-Term Memory (LTM)** | Hard disk | User prefs, facts across sessions | Persistent |
| 3 | **Working Memory** | Scratchpad | Mid-task reasoning state | Single task |
| 4 | **Episodic Memory** | Diary | Past experiences with timestamps | Persistent |
| 5 | **Semantic Memory** | Encyclopedia | Facts, knowledge, relationships | Persistent |
| 6 | **Procedural Memory** | Muscle memory | How-to steps, workflows, skills | Persistent |

## Why These Six?

The LLM at the core of any agent is **stateless** — every inference call starts fresh.
Memory is NOT a model problem; it's an **infrastructure problem**. You build memory
*around* the model, deciding what gets injected into the context window and when.

Different situations need different information:
- Recent conversation → **Short-Term Memory**
- User preferences across sessions → **Long-Term Memory**
- Current task's intermediate steps → **Working Memory**
- "Last time we tried X and it failed" → **Episodic Memory**
- "Python is a programming language" → **Semantic Memory**
- "How to send an email via API" → **Procedural Memory**

## Project Structure

```
agentic_memory/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── memory_types/
│   ├── __init__.py
│   ├── short_term_memory.py     # Type 1: Conversation buffer
│   ├── long_term_memory.py      # Type 2: Persistent cross-session store
│   ├── working_memory.py        # Type 3: Task scratchpad
│   ├── episodic_memory.py       # Type 4: Experience diary
│   ├── semantic_memory.py       # Type 5: Knowledge & facts (vector-based)
│   └── procedural_memory.py     # Type 6: Workflows & how-to skills
├── unified_agent.py             # Agent combining all 6 memory types
└── demo.py                      # Interactive demo
```

## Quick Start

```bash
pip install -r requirements.txt
python demo.py
```

## Key Concepts

1. **Context Window is the bottleneck** — everything the agent knows must fit in it
2. **Memory ≠ storing everything** — it's about *what* to store and *when* to retrieve
3. **Optimization opportunities**: compression, summarization, intelligent eviction
4. **The memory lifecycle**: Ingest → Store → Retrieve → Inject → Forget

"""
╔══════════════════════════════════════════════════════════════════╗
║  TYPE 4: EPISODIC MEMORY — The Experience Diary                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  WHAT: Records specific past interactions/experiences with       ║
║        timestamps, context, actions taken, and outcomes.         ║
║                                                                  ║
║  WHY:  "Last Tuesday, we tried Approach A and it failed.         ║
║         Let's try Approach B this time."                         ║
║                                                                  ║
║  ANALOGY: Personal diary — "what happened, when, and how it     ║
║           turned out."                                           ║
║                                                                  ║
║  LIFESPAN: Persistent. The agent's autobiography.                ║
║                                                                  ║
║  DIFFERENCE FROM LTM:                                            ║
║  - LTM stores FACTS: "user prefers Python"                      ║
║  - Episodic stores EVENTS: "On May 5, we debugged a Python      ║
║    error. The fix was to update the dependency."                 ║
║                                                                  ║
║  KEY CHALLENGE: Episodes pile up fast. How to retrieve the       ║
║  RIGHT past experience for the CURRENT situation?                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import json
import time
import uuid
from pathlib import Path


@dataclass
class Episode:
    """A single recorded experience/event."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    # WHAT happened
    situation: str = ""       # Context/trigger for the episode
    task: str = ""            # What the agent was trying to do
    action: str = ""          # What action was taken
    result: str = ""          # What happened as a result
    outcome: str = "neutral"  # success | failure | partial | neutral

    # METADATA
    user_id: str = "default"
    tags: list[str] = field(default_factory=list)
    lesson: str = ""          # What was learned (extracted insight)

    @property
    def age_hours(self) -> float:
        return (time.time() - self.timestamp) / 3600

    def to_narrative(self) -> str:
        """Convert episode to natural language for context injection."""
        time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.timestamp))
        parts = [f"[{time_str}]"]
        if self.situation:
            parts.append(f"Situation: {self.situation}")
        if self.task:
            parts.append(f"Task: {self.task}")
        if self.action:
            parts.append(f"Action: {self.action}")
        if self.result:
            parts.append(f"Result: {self.result}")
        parts.append(f"Outcome: {self.outcome}")
        if self.lesson:
            parts.append(f"Lesson: {self.lesson}")
        return " | ".join(parts)


class EpisodicMemory:
    """
    Episodic Memory: The agent's diary of past experiences.
    
    Inspired by human episodic memory (Tulving, 1972):
    - Records WHAT happened, WHEN, in WHAT context
    - Enables learning from past successes and failures
    - Supports pattern recognition across experiences
    
    KEY CAPABILITIES:
    1. Record new episodes (append-only log)
    2. Recall by similarity (find relevant past experiences)
    3. Recall by recency (what happened recently?)
    4. Extract patterns (what strategies work/fail?)
    5. Consolidation → convert episodes into semantic knowledge
    
    🔑 OPTIMIZATION OPPORTUNITIES:
    - Semantic search over episodes (not just keyword matching)
    - Hierarchical episodes (daily summaries → weekly → monthly)
    - Importance-weighted retention (keep impactful episodes longer)
    - Episode compression (merge similar experiences)
    """

    def __init__(self, storage_path: str = "episodic_store.json", max_episodes: int = 500):
        self._storage_path = Path(storage_path)
        self._episodes: list[Episode] = []
        self._max_episodes = max_episodes
        self._load()

    # ── Recording Episodes ───────────────────────────────────────────

    def record(
        self,
        situation: str,
        task: str,
        action: str,
        result: str,
        outcome: str = "neutral",
        tags: list[str] | None = None,
        lesson: str = "",
    ) -> Episode:
        """
        Record a new episode.
        
        This should be called at key moments:
        - After completing a task (success or failure)
        - When the user gives feedback
        - When an unexpected situation occurs
        - When the agent learns something new
        
        Example:
            episodic.record(
                situation="User asked to deploy to production",
                task="Run deployment pipeline",
                action="Executed `deploy.sh` without running tests first",
                result="Deployment failed due to broken unit test",
                outcome="failure",
                tags=["deployment", "testing"],
                lesson="Always run tests before deployment"
            )
        """
        episode = Episode(
            situation=situation,
            task=task,
            action=action,
            result=result,
            outcome=outcome,
            tags=tags or [],
            lesson=lesson,
        )
        self._episodes.append(episode)
        self._enforce_limits()
        self._save()
        return episode

    # ── Recall Strategies ────────────────────────────────────────────

    def recall_recent(self, n: int = 5) -> list[Episode]:
        """Recall the N most recent episodes."""
        return sorted(self._episodes, key=lambda e: e.timestamp, reverse=True)[:n]

    def recall_by_tag(self, tag: str) -> list[Episode]:
        """Recall episodes tagged with a specific label."""
        return [e for e in self._episodes if tag in e.tags]

    def recall_by_outcome(self, outcome: str) -> list[Episode]:
        """Recall episodes with a specific outcome (success/failure/etc)."""
        return [e for e in self._episodes if e.outcome == outcome]

    def recall_similar(self, query: str, top_k: int = 5) -> list[Episode]:
        """
        Find episodes most relevant to the current situation.
        
        🔑 Simple keyword matching here. In production, you'd:
        - Embed episodes and query with a sentence transformer
        - Use cosine similarity to find nearest episodes
        - This is where vector databases like Pinecone/Weaviate shine
        """
        query_words = set(query.lower().split())
        scored = []

        for episode in self._episodes:
            # Combine all episode text for matching
            episode_text = f"{episode.situation} {episode.task} {episode.action} {episode.result} {episode.lesson}"
            episode_words = set(episode_text.lower().split())

            # Jaccard-like similarity
            overlap = len(query_words & episode_words)
            if overlap > 0:
                score = overlap / len(query_words | episode_words)
                scored.append((score, episode))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    # ── Pattern Recognition ──────────────────────────────────────────

    def extract_patterns(self) -> dict:
        """
        Analyze episodes to find patterns.
        
        🔑 THIS IS POWERFUL for your internship project:
        - What actions lead to success vs failure?
        - Which tags are associated with failures?
        - Are there recurring lessons?
        
        In production, you'd use an LLM to analyze episodes and
        extract higher-level insights automatically.
        """
        patterns = {
            "total_episodes": len(self._episodes),
            "success_rate": 0.0,
            "common_failure_tags": {},
            "common_success_tags": {},
            "lessons_learned": [],
        }

        successes = [e for e in self._episodes if e.outcome == "success"]
        failures = [e for e in self._episodes if e.outcome == "failure"]

        if self._episodes:
            patterns["success_rate"] = len(successes) / len(self._episodes)

        # Find tags associated with failures
        for ep in failures:
            for tag in ep.tags:
                patterns["common_failure_tags"][tag] = patterns["common_failure_tags"].get(tag, 0) + 1

        # Find tags associated with successes
        for ep in successes:
            for tag in ep.tags:
                patterns["common_success_tags"][tag] = patterns["common_success_tags"].get(tag, 0) + 1

        # Collect unique lessons
        patterns["lessons_learned"] = list(set(
            ep.lesson for ep in self._episodes if ep.lesson
        ))

        return patterns

    def consolidate_to_semantic(self) -> list[dict]:
        """
        🔑 MEMORY CONSOLIDATION: Convert episodes → semantic knowledge.
        
        This is inspired by how human memory works:
        - You experience many episodes of "touching a hot stove"
        - Over time, these consolidate into semantic knowledge: "stoves are hot"
        
        In agentic AI:
        - Many episodes of "deployment failed when tests weren't run"
        → Semantic fact: "Always run tests before deployment"
        
        This bridges Episodic Memory → Semantic Memory.
        """
        # Group episodes by tags and extract common lessons
        tag_groups: dict[str, list[Episode]] = {}
        for ep in self._episodes:
            for tag in ep.tags:
                tag_groups.setdefault(tag, []).append(ep)

        semantic_facts = []
        for tag, episodes in tag_groups.items():
            if len(episodes) >= 2:  # Need multiple episodes to form a pattern
                successes = [e for e in episodes if e.outcome == "success"]
                failures = [e for e in episodes if e.outcome == "failure"]
                lessons = [e.lesson for e in episodes if e.lesson]

                fact = {
                    "topic": tag,
                    "derived_from": f"{len(episodes)} episodes",
                    "success_rate": len(successes) / len(episodes) if episodes else 0,
                    "key_lessons": list(set(lessons)),
                    "recommendation": (
                        f"Based on {len(episodes)} past experiences with '{tag}': "
                        f"{len(successes)} succeeded, {len(failures)} failed."
                    ),
                }
                semantic_facts.append(fact)

        return semantic_facts

    # ── Context Injection ────────────────────────────────────────────

    def get_context_for_prompt(self, current_situation: str, max_episodes: int = 3) -> str:
        """
        Build context from relevant past episodes to inject into the prompt.
        
        This helps the agent learn from past experience:
        "I've seen a similar situation before — here's what happened."
        """
        relevant = self.recall_similar(current_situation, top_k=max_episodes)

        if not relevant:
            return ""

        lines = ["[Episodic Memory — Relevant Past Experiences]:"]
        for ep in relevant:
            lines.append(f"  • {ep.to_narrative()}")
        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────────

    def _save(self) -> None:
        data = [asdict(ep) for ep in self._episodes]
        self._storage_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        if self._storage_path.exists():
            data = json.loads(self._storage_path.read_text())
            self._episodes = [Episode(**d) for d in data]

    def _enforce_limits(self) -> None:
        """Keep only the most important episodes when over limit."""
        if len(self._episodes) > self._max_episodes:
            # Keep failures and lessons (more valuable than neutral episodes)
            important = [e for e in self._episodes if e.outcome == "failure" or e.lesson]
            neutral = [e for e in self._episodes if e not in important]
            # Drop oldest neutral episodes first
            keep = len(self._max_episodes) - len(important)
            self._episodes = important + neutral[-max(0, keep):]

    @property
    def stats(self) -> dict:
        outcomes = {}
        for ep in self._episodes:
            outcomes[ep.outcome] = outcomes.get(ep.outcome, 0) + 1
        return {
            "total_episodes": len(self._episodes),
            "outcomes": outcomes,
            "unique_tags": len(set(t for ep in self._episodes for t in ep.tags)),
            "with_lessons": sum(1 for ep in self._episodes if ep.lesson),
        }


# ── Demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    print("=" * 60)
    print("  TYPE 4: EPISODIC MEMORY DEMO")
    print("=" * 60)

    # Clean up
    ep_path = "/tmp/episodic_demo.json"
    if os.path.exists(ep_path):
        os.remove(ep_path)

    em = EpisodicMemory(storage_path=ep_path)

    # Record a series of episodes
    print("\n📖 Recording agent experiences:\n")

    episodes_data = [
        {
            "situation": "User asked to deploy FastAPI app to production",
            "task": "Run deployment pipeline",
            "action": "Ran deploy.sh directly without tests",
            "result": "Deployment failed — 3 unit tests broken",
            "outcome": "failure",
            "tags": ["deployment", "testing", "fastapi"],
            "lesson": "Always run test suite before deployment",
        },
        {
            "situation": "User asked to deploy FastAPI app (second attempt)",
            "task": "Run deployment with tests first",
            "action": "Ran pytest, fixed 3 failures, then deployed",
            "result": "Deployment succeeded, app running on port 8000",
            "outcome": "success",
            "tags": ["deployment", "testing", "fastapi"],
            "lesson": "Test-first deployment is reliable",
        },
        {
            "situation": "User asked to optimize a slow database query",
            "task": "Improve query performance",
            "action": "Added index on frequently filtered column",
            "result": "Query time dropped from 2.3s to 0.05s",
            "outcome": "success",
            "tags": ["database", "optimization", "postgres"],
            "lesson": "Check for missing indexes before complex query rewrites",
        },
        {
            "situation": "User asked to set up CI/CD pipeline",
            "task": "Configure GitHub Actions",
            "action": "Created workflow with test + deploy stages",
            "result": "Pipeline runs on every push, deploys on main merge",
            "outcome": "success",
            "tags": ["cicd", "github", "deployment"],
            "lesson": "Separate test and deploy stages for clarity",
        },
        {
            "situation": "User asked to debug memory leak in Python app",
            "task": "Find and fix memory leak",
            "action": "Used tracemalloc to identify leak in cache",
            "result": "Found unbounded dict growing with each request",
            "outcome": "success",
            "tags": ["debugging", "memory", "python"],
            "lesson": "Always set max size for in-memory caches",
        },
    ]

    for ep_data in episodes_data:
        ep = em.record(**ep_data)
        icon = "✅" if ep.outcome == "success" else "❌"
        print(f"  {icon} [{ep.outcome:>7}] {ep.task}")

    # Recall similar episodes
    print("\n🔍 Recalling episodes similar to 'deploy application to server':")
    similar = em.recall_similar("deploy application to server")
    for ep in similar:
        print(f"  → [{ep.outcome}] {ep.task} — Lesson: {ep.lesson}")

    # Extract patterns
    print("\n📊 Patterns extracted from episodes:")
    patterns = em.extract_patterns()
    print(f"  Success rate: {patterns['success_rate']:.0%}")
    print(f"  Failure tags: {patterns['common_failure_tags']}")
    print(f"  Lessons: {patterns['lessons_learned']}")

    # Consolidation: Episodes → Semantic knowledge
    print("\n🔄 Consolidating episodes into semantic knowledge:")
    facts = em.consolidate_to_semantic()
    for fact in facts:
        print(f"  📚 [{fact['topic']}]: {fact['recommendation']}")
        if fact['key_lessons']:
            for lesson in fact['key_lessons']:
                print(f"      💡 {lesson}")

    # Context for prompt
    print("\n🪟 Context for 'deploy my app' (injected into LLM):")
    print("-" * 50)
    print(em.get_context_for_prompt("deploy my application"))

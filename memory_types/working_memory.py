"""
╔══════════════════════════════════════════════════════════════════╗
║  TYPE 3: WORKING MEMORY — The Task Scratchpad                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  WHAT: Stores intermediate reasoning state, sub-task results,   ║
║        and scratch notes while an agent works on a complex task. ║
║                                                                  ║
║  WHY:  When an agent breaks "Plan a trip to Japan" into steps    ║
║        (flights → hotels → itinerary), it needs to carry        ║
║        intermediate results between steps.                       ║
║                                                                  ║
║  ANALOGY: Whiteboard during a meeting — used actively, erased   ║
║           when the task is done.                                 ║
║                                                                  ║
║  LIFESPAN: Single task. Cleared when the task completes.         ║
║                                                                  ║
║  DIFFERENCE FROM STM:                                            ║
║  - STM = conversation history (user ↔ agent messages)            ║
║  - Working Memory = agent's internal notes (not shown to user)  ║
║                                                                  ║
║  KEY CHALLENGE: How much intermediate state to keep? Agent       ║
║  tool outputs can be huge (API responses, search results).       ║
║  You need to compress/extract what matters.                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import time


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubTaskResult:
    """Result from a sub-task or tool call."""
    name: str
    output: Any
    status: TaskStatus = TaskStatus.COMPLETED
    timestamp: float = field(default_factory=time.time)
    tokens_used: int = 0  # track how much context this consumes


@dataclass
class TaskPlan:
    """A high-level plan for the current task."""
    goal: str
    steps: list[str]
    current_step: int = 0
    status: TaskStatus = TaskStatus.PENDING


class WorkingMemory:
    """
    Working Memory: The agent's internal scratchpad during task execution.
    
    Think of this as the agent's "thinking space" — where it:
    1. Stores the task plan (what steps to take)
    2. Records intermediate results (tool outputs, sub-task results)
    3. Maintains key variables and context across steps
    4. Keeps running notes about what's working and what's not
    
    CRITICAL DIFFERENCE from STM:
    - STM is the conversation (visible to the user)
    - Working Memory is the agent's internal state (invisible to user)
    
    🔑 OPTIMIZATION OPPORTUNITIES:
    - Compress large tool outputs to only essential info
    - Prioritize which sub-results to keep in context
    - Detect when a step's output is no longer needed
    - Parallelize sub-tasks that don't depend on each other
    """

    def __init__(self, max_context_tokens: int = 2048):
        self._plan: TaskPlan | None = None
        self._results: list[SubTaskResult] = []
        self._notes: list[str] = []  # Agent's internal notes
        self._variables: dict[str, Any] = {}  # Key-value scratchpad
        self._max_tokens = max_context_tokens
        self._total_tokens = 0

    # ── Task Planning ────────────────────────────────────────────────

    def set_plan(self, goal: str, steps: list[str]) -> TaskPlan:
        """
        Set the task plan — what the agent intends to do.
        
        Example:
            goal = "Find the best restaurant near the user's hotel"
            steps = [
                "1. Get user's hotel location from LTM",
                "2. Search for restaurants within 1km",
                "3. Filter by rating > 4.0",
                "4. Check if any match user's cuisine preferences",
                "5. Format and present top 3 options"
            ]
        """
        self._plan = TaskPlan(goal=goal, steps=steps, status=TaskStatus.IN_PROGRESS)
        return self._plan

    def advance_step(self) -> str | None:
        """Move to the next step in the plan."""
        if self._plan and self._plan.current_step < len(self._plan.steps) - 1:
            self._plan.current_step += 1
            return self._plan.steps[self._plan.current_step]
        elif self._plan:
            self._plan.status = TaskStatus.COMPLETED
        return None

    def get_current_step(self) -> str | None:
        """Get the current step description."""
        if self._plan and self._plan.current_step < len(self._plan.steps):
            return self._plan.steps[self._plan.current_step]
        return None

    # ── Result Storage ───────────────────────────────────────────────

    def store_result(self, name: str, output: Any, compress: bool = True) -> None:
        """
        Store a sub-task result (e.g., tool output).
        
        🔑 OPTIMIZATION: If compress=True, large outputs are truncated.
        In production, you'd use an LLM to extract key information.
        
        Example: A web search returns 10 pages of results.
        You only need the top 3 relevant snippets.
        """
        if compress:
            output = self._compress_output(output)

        result = SubTaskResult(name=name, output=output)
        result.tokens_used = len(str(output)) // 4 + 1
        self._results.append(result)
        self._total_tokens += result.tokens_used
        self._enforce_limits()

    def _compress_output(self, output: Any, max_chars: int = 500) -> Any:
        """
        Compress large outputs to fit in working memory.
        
        🔑 In production, you'd:
        - Use an LLM to summarize ("Extract the 3 most relevant facts")
        - Use structured extraction (pull specific fields from JSON)
        - Apply domain-specific compression (e.g., only keep prices from a pricing API)
        """
        output_str = str(output)
        if len(output_str) > max_chars:
            return output_str[:max_chars] + f"\n... [truncated, {len(output_str) - max_chars} chars omitted]"
        return output

    def get_result(self, name: str) -> Any | None:
        """Get a specific sub-task result by name."""
        for result in reversed(self._results):  # latest first
            if result.name == name:
                return result.output
        return None

    # ── Notes & Variables ────────────────────────────────────────────

    def add_note(self, note: str) -> None:
        """
        Add an internal note (agent's thoughts/observations).
        
        Example notes:
        - "User seems impatient, keep responses shorter"
        - "API rate limit hit, need to wait 60 seconds"
        - "Search results were poor, trying different query"
        """
        self._notes.append(f"[{time.strftime('%H:%M:%S')}] {note}")

    def set_variable(self, key: str, value: Any) -> None:
        """Store a working variable (like a local variable in code)."""
        self._variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a working variable."""
        return self._variables.get(key, default)

    # ── Context Building ─────────────────────────────────────────────

    def get_context(self) -> str:
        """
        Build the working memory context for injection into the prompt.
        
        This goes into the system prompt or a special section, giving
        the LLM awareness of the current task state.
        """
        parts = ["[Working Memory — Current Task State]:"]

        # Plan status
        if self._plan:
            parts.append(f"\n📋 Goal: {self._plan.goal}")
            parts.append(f"   Status: {self._plan.status.value}")
            parts.append(f"   Step {self._plan.current_step + 1}/{len(self._plan.steps)}: "
                         f"{self.get_current_step()}")

        # Key variables
        if self._variables:
            parts.append("\n📌 Variables:")
            for k, v in self._variables.items():
                v_str = str(v)[:100]
                parts.append(f"   {k} = {v_str}")

        # Recent results (last 3 to save tokens)
        if self._results:
            parts.append("\n📦 Recent Results:")
            for result in self._results[-3:]:
                out_str = str(result.output)[:150]
                parts.append(f"   [{result.name}]: {out_str}")

        # Notes
        if self._notes:
            parts.append("\n📝 Notes:")
            for note in self._notes[-5:]:
                parts.append(f"   {note}")

        return "\n".join(parts)

    def _enforce_limits(self) -> None:
        """Drop oldest results when over token budget."""
        while self._total_tokens > self._max_tokens and len(self._results) > 1:
            removed = self._results.pop(0)
            self._total_tokens -= removed.tokens_used

    def clear(self) -> None:
        """Clear working memory (task complete)."""
        self._plan = None
        self._results.clear()
        self._notes.clear()
        self._variables.clear()
        self._total_tokens = 0

    @property
    def stats(self) -> dict:
        return {
            "has_plan": self._plan is not None,
            "plan_status": self._plan.status.value if self._plan else None,
            "results_stored": len(self._results),
            "variables": len(self._variables),
            "notes": len(self._notes),
            "tokens_used": self._total_tokens,
        }


# ── Demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TYPE 3: WORKING MEMORY DEMO")
    print("=" * 60)

    wm = WorkingMemory(max_context_tokens=1000)

    # Simulate an agent planning a trip
    print("\n🎯 Agent receives task: 'Find me a good restaurant near my hotel'\n")

    # Step 1: Create plan
    plan = wm.set_plan(
        goal="Find the best restaurant near the user's hotel",
        steps=[
            "Get user's hotel location from long-term memory",
            "Search for restaurants within 1km radius",
            "Filter by rating and cuisine preference",
            "Present top 3 options to user",
        ]
    )
    print(f"📋 Plan created: {len(plan.steps)} steps")

    # Step 2: Execute step 1
    print(f"\n▶ Step 1: {wm.get_current_step()}")
    wm.store_result("hotel_location", {"name": "Taj Mahal Palace", "lat": 18.9217, "lon": 72.8332})
    wm.set_variable("hotel_lat", 18.9217)
    wm.set_variable("hotel_lon", 72.8332)
    wm.add_note("Hotel found in LTM. Located in Colaba, Mumbai.")
    wm.advance_step()

    # Step 3: Execute step 2
    print(f"▶ Step 2: {wm.get_current_step()}")
    wm.store_result("nearby_restaurants", [
        {"name": "Trishna", "rating": 4.5, "cuisine": "Seafood", "distance": "0.3km"},
        {"name": "Bademiya", "rating": 4.2, "cuisine": "Kebabs", "distance": "0.5km"},
        {"name": "Indigo", "rating": 4.4, "cuisine": "European", "distance": "0.7km"},
        {"name": "Leopold Cafe", "rating": 4.0, "cuisine": "Multi-cuisine", "distance": "0.2km"},
    ])
    wm.add_note("Found 4 restaurants within 1km. All have good ratings.")
    wm.advance_step()

    # Step 4: Execute step 3
    print(f"▶ Step 3: {wm.get_current_step()}")
    wm.set_variable("user_cuisine_pref", "Seafood")
    wm.store_result("filtered_restaurants", [
        {"name": "Trishna", "rating": 4.5, "cuisine": "Seafood", "distance": "0.3km"},
    ])
    wm.add_note("User prefers Seafood (from LTM). Trishna is the top match.")
    wm.advance_step()

    # Show the working memory state
    print(f"\n📊 Working Memory Stats: {wm.stats}")
    print("\n🧠 Full Working Memory Context (injected into LLM prompt):")
    print("-" * 50)
    print(wm.get_context())

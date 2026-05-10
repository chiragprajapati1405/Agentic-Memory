"""
╔══════════════════════════════════════════════════════════════════╗
║  TYPE 6: PROCEDURAL MEMORY — Learned Skills & Workflows         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  WHAT: Stores HOW to do things — reusable procedures,           ║
║        workflows, tool-use patterns, and system prompts that    ║
║        the agent has learned work well.                          ║
║                                                                  ║
║  WHY:  First time: agent figures out step-by-step how to do X.  ║
║        Next time: agent recalls the procedure and executes it   ║
║        directly — faster, fewer errors, no re-discovery.        ║
║                                                                  ║
║  ANALOGY: Muscle memory / recipe book — you once learned to     ║
║           ride a bike step by step, now you just DO it.          ║
║                                                                  ║
║  LIFESPAN: Persistent. Improves over time with feedback.        ║
║                                                                  ║
║  DIFFERENCE FROM OTHER MEMORY TYPES:                             ║
║  ┌────────────┬────────────────────────────────────────┐        ║
║  │ Semantic   │ "FAISS is a similarity search library" │        ║
║  │ Episodic   │ "Last time I used FAISS, indexing took │        ║
║  │            │  3 min and search was 0.5ms"           │        ║
║  │ Procedural │ "To build a FAISS index:               │        ║
║  │            │  1. Embed documents with MiniLM         │        ║
║  │            │  2. Normalize vectors                   │        ║
║  │            │  3. Create IndexFlatIP                  │        ║
║  │            │  4. Add vectors in batches of 1000"     │        ║
║  └────────────┴────────────────────────────────────────┘        ║
║                                                                  ║
║  Semantic = WHAT is true (facts)                                ║
║  Episodic = WHAT happened (experiences)                         ║
║  Procedural = HOW to do it (skills)                             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal


# ════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════

@dataclass
class ProcedureStep:
    """A single step within a procedure."""
    order: int                          # Execution order (1, 2, 3, ...)
    instruction: str                    # What to do
    tool: str | None = None             # Tool/function to call (if any)
    tool_args: dict | None = None       # Arguments for the tool
    expected_output: str = ""           # What success looks like
    fallback: str = ""                  # What to do if this step fails
    is_conditional: bool = False        # Does this step depend on a condition?
    condition: str = ""                 # The condition (if conditional)


@dataclass
class Procedure:
    """
    A complete learned procedure — a reusable multi-step workflow.
    
    This is the core unit of procedural memory. Think of it as a
    recipe that the agent has learned and can execute again.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""                      # Human-readable name
    description: str = ""               # What this procedure accomplishes
    trigger: str = ""                   # When to use this procedure
    steps: list[dict] = field(default_factory=list)  # ProcedureSteps as dicts
    
    # Metadata
    category: str = "general"           # deployment, debugging, data, coding, etc.
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # Performance tracking
    times_executed: int = 0
    times_succeeded: int = 0
    avg_duration_sec: float = 0.0
    last_feedback: str = ""
    
    # Versioning — procedures improve over time
    version: int = 1
    parent_id: str | None = None        # ID of the procedure this evolved from

    @property
    def success_rate(self) -> float:
        if self.times_executed == 0:
            return 0.0
        return self.times_succeeded / self.times_executed

    @property
    def reliability_score(self) -> float:
        """
        Combined score: success_rate * confidence_from_usage.
        
        A procedure used 50 times with 90% success is more reliable
        than one used 2 times with 100% success.
        """
        if self.times_executed == 0:
            return 0.0
        usage_confidence = min(1.0, self.times_executed / 10)  # Caps at 10 uses
        return self.success_rate * usage_confidence


@dataclass
class SystemPromptTemplate:
    """
    A learned system prompt / behavioral pattern.
    
    Procedural memory doesn't just store tool-use workflows.
    It also stores BEHAVIORAL procedures — system prompts and
    interaction patterns that the agent has learned work well.
    
    Example: "When the user asks about deployment, use a cautious
    tone and always suggest running tests first."
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    prompt: str = ""                    # The actual system prompt text
    context: str = ""                   # When to apply this prompt
    effectiveness_score: float = 0.5    # 0-1, updated with feedback
    times_used: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# ════════════════════════════════════════════════════════════════════
#  PROCEDURAL MEMORY
# ════════════════════════════════════════════════════════════════════

class ProceduralMemory:
    """
    Procedural Memory: The agent's skill library.
    
    Stores HOW to do things — reusable procedures, workflows,
    tool-use patterns, and behavioral templates.
    
    TWO FORMS (following cognitive science):
    
    1. EXPLICIT PROCEDURES — Step-by-step workflows
       "To deploy: run tests → build → push → verify"
       Stored as ordered lists of steps with tools and fallbacks.
    
    2. IMPLICIT PROCEDURES — Behavioral patterns / system prompts
       "When user is frustrated, be more empathetic and concise"
       Stored as system prompt templates selected by context.
    
    KEY CAPABILITIES:
    1. Store new procedures (manually or from successful episodes)
    2. Retrieve the right procedure for a given task
    3. Execute procedures step-by-step
    4. Learn from feedback — refine procedures over time
    5. Version procedures — track how they evolve
    6. Auto-generate procedures from episodic memory patterns
    
    🔑 OPTIMIZATION OPPORTUNITIES:
    - Procedure selection: which procedure best fits the task?
    - Step optimization: can steps be parallelized? reordered?
    - Feedback loops: how to incorporate user feedback?
    - Procedure merging: combine similar procedures
    - Automatic procedure generation from repeated episodes
    """

    def __init__(self, storage_path: str = "procedural_store.json"):
        self._storage_path = Path(storage_path)
        self._procedures: dict[str, Procedure] = {}
        self._prompts: dict[str, SystemPromptTemplate] = {}
        self._execution_log: list[dict] = []  # Track executions
        self._load()

    # ── PROCEDURE MANAGEMENT ─────────────────────────────────────────

    def add_procedure(
        self,
        name: str,
        description: str,
        trigger: str,
        steps: list[dict],
        category: str = "general",
        tags: list[str] | None = None,
    ) -> Procedure:
        """
        Store a new procedure.
        
        Args:
            name: Human-readable name ("Deploy FastAPI App")
            description: What it accomplishes
            trigger: When to use it ("user asks to deploy", "CI fails")
            steps: List of step dicts, each with:
                   {order, instruction, tool, tool_args, expected_output, fallback}
            category: Grouping tag
            tags: Additional searchable tags
        
        Example:
            pm.add_procedure(
                name="Deploy FastAPI to Production",
                description="Full deployment pipeline with safety checks",
                trigger="user asks to deploy application",
                steps=[
                    {"order": 1, "instruction": "Run pytest suite",
                     "tool": "bash", "tool_args": {"cmd": "pytest -v"},
                     "expected_output": "All tests passed",
                     "fallback": "Fix failing tests before proceeding"},
                    {"order": 2, "instruction": "Build Docker image",
                     "tool": "bash", "tool_args": {"cmd": "docker build -t app ."},
                     "expected_output": "Image built successfully"},
                    ...
                ]
            )
        """
        proc = Procedure(
            name=name,
            description=description,
            trigger=trigger,
            steps=steps,
            category=category,
            tags=tags or [],
        )
        self._procedures[proc.id] = proc
        self._save()
        return proc

    def get_procedure(self, proc_id: str) -> Procedure | None:
        return self._procedures.get(proc_id)

    def find_procedure(self, task_description: str, category: str | None = None) -> list[tuple[Procedure, float]]:
        """
        Find the best procedure(s) for a given task.
        
        🔑 This is where procedure SELECTION happens.
        In production, you'd:
        - Embed the task description and procedures
        - Use cosine similarity to find the closest match
        - Weight by reliability_score (prefer battle-tested procedures)
        
        Here we use keyword matching + reliability weighting.
        """
        query_words = set(task_description.lower().split())
        scored = []

        for proc in self._procedures.values():
            if category and proc.category != category:
                continue

            # Match against trigger, name, description, tags
            proc_text = f"{proc.trigger} {proc.name} {proc.description} {' '.join(proc.tags)}"
            proc_words = set(proc_text.lower().split())

            overlap = len(query_words & proc_words)
            if overlap == 0:
                continue

            # Relevance = keyword overlap * reliability
            text_score = overlap / max(len(query_words | proc_words), 1)
            reliability = proc.reliability_score if proc.times_executed > 0 else 0.5
            final_score = text_score * 0.6 + reliability * 0.4

            scored.append((proc, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ── PROCEDURE EXECUTION ──────────────────────────────────────────

    def execute_procedure(self, proc_id: str, dry_run: bool = True) -> dict:
        """
        Execute a procedure step-by-step.
        
        In a real agent, each step would:
        1. Call the specified tool with the given arguments
        2. Check if the output matches expected_output
        3. If not, execute the fallback
        4. Log the result
        
        Args:
            proc_id: ID of the procedure to execute
            dry_run: If True, simulate execution without calling tools
        
        Returns:
            Execution report with step results
        """
        proc = self._procedures.get(proc_id)
        if not proc:
            return {"error": f"Procedure {proc_id} not found"}

        start_time = time.time()
        execution = {
            "procedure_id": proc_id,
            "procedure_name": proc.name,
            "started_at": start_time,
            "steps": [],
            "status": "running",
        }

        all_passed = True
        for step_dict in sorted(proc.steps, key=lambda s: s.get("order", 0)):
            step_result = {
                "order": step_dict.get("order"),
                "instruction": step_dict.get("instruction"),
                "tool": step_dict.get("tool"),
                "status": "pending",
            }

            # Check conditional steps
            if step_dict.get("is_conditional") and step_dict.get("condition"):
                step_result["condition"] = step_dict["condition"]
                # In a real system, evaluate the condition here

            if dry_run:
                # Simulate execution
                step_result["status"] = "simulated_success"
                step_result["output"] = f"[DRY RUN] Would execute: {step_dict.get('instruction', '')}"
            else:
                # In production: actually call the tool
                # result = tool_executor.run(step_dict["tool"], step_dict.get("tool_args", {}))
                step_result["status"] = "success"
                step_result["output"] = step_dict.get("expected_output", "")

            execution["steps"].append(step_result)

            if step_result["status"] in ("failed", "error"):
                all_passed = False
                step_result["fallback_triggered"] = step_dict.get("fallback", "")
                # Don't necessarily stop — some procedures continue after failures

        # Update execution stats
        duration = time.time() - start_time
        proc.times_executed += 1
        if all_passed:
            proc.times_succeeded += 1
        
        # Running average of duration
        if proc.avg_duration_sec == 0:
            proc.avg_duration_sec = duration
        else:
            proc.avg_duration_sec = (proc.avg_duration_sec * 0.8) + (duration * 0.2)

        execution["status"] = "success" if all_passed else "partial_failure"
        execution["duration_sec"] = duration
        execution["success_rate_after"] = proc.success_rate

        self._execution_log.append(execution)
        self._save()
        return execution

    # ── PROCEDURE LEARNING & REFINEMENT ──────────────────────────────

    def refine_from_feedback(
        self,
        proc_id: str,
        feedback: str,
        rating: Literal["good", "bad", "neutral"] = "neutral",
        modifications: dict | None = None,
    ) -> Procedure | None:
        """
        Refine a procedure based on user feedback.
        
        🔑 THIS IS HOW AGENTS LEARN SKILLS OVER TIME.
        
        Three refinement strategies:
        1. MINOR: Just update feedback text (rating = neutral)
        2. MODERATE: Adjust step order/instructions (rating = bad + mods)
        3. MAJOR: Create a new version (significant changes)
        
        Example:
            pm.refine_from_feedback(
                proc_id="abc123",
                feedback="Step 3 should come before step 2",
                rating="bad",
                modifications={"reorder": {2: 3, 3: 2}}
            )
        """
        proc = self._procedures.get(proc_id)
        if not proc:
            return None

        proc.last_feedback = feedback
        proc.updated_at = time.time()

        if rating == "bad" and modifications:
            # Create a new version (preserve history)
            new_proc = Procedure(
                name=proc.name,
                description=proc.description,
                trigger=proc.trigger,
                steps=list(proc.steps),  # Copy steps
                category=proc.category,
                tags=list(proc.tags),
                version=proc.version + 1,
                parent_id=proc.id,
            )

            # Apply modifications
            if "reorder" in modifications:
                mapping = modifications["reorder"]
                for step in new_proc.steps:
                    old_order = step.get("order")
                    if old_order in mapping:
                        step["order"] = mapping[old_order]

            if "add_step" in modifications:
                new_proc.steps.append(modifications["add_step"])

            if "remove_step" in modifications:
                order_to_remove = modifications["remove_step"]
                new_proc.steps = [s for s in new_proc.steps if s.get("order") != order_to_remove]

            if "update_step" in modifications:
                target_order = modifications["update_step"]["order"]
                for step in new_proc.steps:
                    if step.get("order") == target_order:
                        step.update(modifications["update_step"])

            self._procedures[new_proc.id] = new_proc
            self._save()
            return new_proc

        self._save()
        return proc

    def generate_from_episodes(self, episodes: list[dict], min_occurrences: int = 2) -> list[Procedure]:
        """
        🔑 AUTO-GENERATE procedures from repeated episodic patterns.
        
        This bridges Episodic Memory → Procedural Memory:
        - If the agent has done the same task 3+ times...
        - ...extract the common action sequence...
        - ...and create a reusable procedure.
        
        This is how humans develop "muscle memory":
        Do something enough times → it becomes automatic.
        
        Args:
            episodes: List of episode dicts from EpisodicMemory
            min_occurrences: How many times a pattern must appear
        
        Returns:
            List of newly generated procedures
        """
        # Group episodes by task similarity
        task_groups: dict[str, list[dict]] = {}
        for ep in episodes:
            task = ep.get("task", "").lower().strip()
            if not task:
                continue
            # Simple grouping by first 3 words (production: use embeddings)
            key = " ".join(task.split()[:3])
            task_groups.setdefault(key, []).append(ep)

        generated = []
        for key, group in task_groups.items():
            if len(group) < min_occurrences:
                continue

            # Find the most successful episodes
            successes = [ep for ep in group if ep.get("outcome") == "success"]
            if not successes:
                continue

            # Extract the common action pattern
            actions = [ep.get("action", "") for ep in successes]
            lessons = [ep.get("lesson", "") for ep in group if ep.get("lesson")]

            # Build a procedure from the successful pattern
            steps = []
            for i, action in enumerate(actions[:1]):  # Use best example
                steps.append({
                    "order": i + 1,
                    "instruction": action,
                    "expected_output": successes[0].get("result", ""),
                })

            # Add lessons as cautionary steps
            for lesson in lessons[:2]:
                steps.append({
                    "order": len(steps) + 1,
                    "instruction": f"⚠️ Remember: {lesson}",
                    "is_conditional": True,
                    "condition": "always",
                })

            proc = self.add_procedure(
                name=f"Auto-learned: {key.title()}",
                description=f"Procedure auto-generated from {len(group)} past experiences",
                trigger=key,
                steps=steps,
                category="auto_learned",
                tags=["auto-generated", "from-episodes"],
            )
            generated.append(proc)

        return generated

    # ── SYSTEM PROMPT TEMPLATES (Implicit Procedures) ────────────────

    def add_prompt_template(
        self,
        name: str,
        prompt: str,
        context: str,
        effectiveness: float = 0.5,
    ) -> SystemPromptTemplate:
        """
        Store a behavioral pattern as a system prompt template.
        
        These are IMPLICIT procedures — not step-by-step workflows,
        but learned behavioral patterns that shape how the agent acts.
        
        Example:
            pm.add_prompt_template(
                name="Cautious Deployment Mode",
                prompt="You are in deployment mode. Double-check every command. "
                       "Always suggest running tests. Confirm before any destructive action.",
                context="user is deploying to production",
                effectiveness=0.85
            )
        """
        template = SystemPromptTemplate(
            name=name, prompt=prompt, context=context,
            effectiveness_score=effectiveness,
        )
        self._prompts[template.id] = template
        self._save()
        return template

    def get_best_prompt(self, situation: str) -> SystemPromptTemplate | None:
        """
        Find the best system prompt for the current situation.
        
        🔑 This is how the agent adapts its BEHAVIOR based on context.
        Different situations trigger different behavioral modes.
        """
        query_words = set(situation.lower().split())
        best = None
        best_score = 0.0

        for template in self._prompts.values():
            context_words = set(template.context.lower().split())
            overlap = len(query_words & context_words)
            if overlap == 0:
                continue

            score = (overlap / max(len(query_words | context_words), 1)) * template.effectiveness_score
            if score > best_score:
                best_score = score
                best = template

        if best:
            best.times_used += 1
            self._save()
        return best

    def update_prompt_effectiveness(self, prompt_id: str, delta: float) -> None:
        """Adjust effectiveness score based on interaction outcome."""
        if prompt_id in self._prompts:
            t = self._prompts[prompt_id]
            t.effectiveness_score = max(0.0, min(1.0, t.effectiveness_score + delta))
            t.updated_at = time.time()
            self._save()

    # ── CONTEXT INJECTION ────────────────────────────────────────────

    def get_context_for_prompt(self, task_description: str) -> str:
        """
        Build procedural context for injection into the LLM prompt.
        
        Returns:
        1. The best-matching procedure (if any)
        2. The best behavioral template (if any)
        """
        lines = []

        # Find relevant procedures
        matches = self.find_procedure(task_description)
        if matches:
            best_proc, score = matches[0]
            lines.append("[Procedural Memory — Known Workflow]:")
            lines.append(f"  Procedure: {best_proc.name} (v{best_proc.version}, "
                         f"reliability={best_proc.reliability_score:.0%})")
            lines.append(f"  Steps:")
            for step in sorted(best_proc.steps, key=lambda s: s.get("order", 0)):
                prefix = f"    {step.get('order', '?')}."
                lines.append(f"{prefix} {step.get('instruction', '')}")
                if step.get("fallback"):
                    lines.append(f"       ↳ Fallback: {step['fallback']}")
            if best_proc.last_feedback:
                lines.append(f"  ⚠️ Last feedback: {best_proc.last_feedback}")

        # Find relevant behavioral template
        best_prompt = self.get_best_prompt(task_description)
        if best_prompt:
            lines.append(f"\n[Procedural Memory — Behavioral Mode]:")
            lines.append(f"  Mode: {best_prompt.name}")
            lines.append(f"  {best_prompt.prompt}")

        return "\n".join(lines) if lines else ""

    # ── ANALYTICS ────────────────────────────────────────────────────

    def get_skill_report(self) -> dict:
        """
        Comprehensive report on the agent's procedural capabilities.
        
        Useful for understanding what the agent has learned to do.
        """
        procedures = list(self._procedures.values())

        report = {
            "total_procedures": len(procedures),
            "total_prompt_templates": len(self._prompts),
            "total_executions": sum(p.times_executed for p in procedures),
            "categories": {},
            "top_procedures": [],
            "needs_improvement": [],
            "auto_learned": [],
        }

        # Group by category
        for p in procedures:
            cat = p.category
            if cat not in report["categories"]:
                report["categories"][cat] = {"count": 0, "avg_reliability": 0.0}
            report["categories"][cat]["count"] += 1

        # Top procedures by reliability
        reliable = sorted(procedures, key=lambda p: p.reliability_score, reverse=True)
        report["top_procedures"] = [
            {"name": p.name, "reliability": f"{p.reliability_score:.0%}",
             "executions": p.times_executed}
            for p in reliable[:5]
        ]

        # Procedures needing improvement
        for p in procedures:
            if p.times_executed >= 3 and p.success_rate < 0.7:
                report["needs_improvement"].append({
                    "name": p.name,
                    "success_rate": f"{p.success_rate:.0%}",
                    "feedback": p.last_feedback,
                })

        # Auto-learned procedures
        report["auto_learned"] = [
            {"name": p.name, "version": p.version}
            for p in procedures if "auto-generated" in p.tags
        ]

        return report

    # ── PERSISTENCE ──────────────────────────────────────────────────

    def _save(self) -> None:
        data = {
            "procedures": {pid: asdict(p) for pid, p in self._procedures.items()},
            "prompts": {pid: asdict(p) for pid, p in self._prompts.items()},
        }
        self._storage_path.write_text(json.dumps(data, indent=2, default=str))

    def _load(self) -> None:
        if self._storage_path.exists():
            data = json.loads(self._storage_path.read_text())
            for pid, pdata in data.get("procedures", {}).items():
                self._procedures[pid] = Procedure(**pdata)
            for pid, pdata in data.get("prompts", {}).items():
                self._prompts[pid] = SystemPromptTemplate(**pdata)

    @property
    def stats(self) -> dict:
        return {
            "procedures": len(self._procedures),
            "prompt_templates": len(self._prompts),
            "total_executions": sum(p.times_executed for p in self._procedures.values()),
            "avg_reliability": (
                sum(p.reliability_score for p in self._procedures.values()) / len(self._procedures)
                if self._procedures else 0
            ),
        }

    def __repr__(self) -> str:
        return f"ProceduralMemory(procedures={len(self._procedures)}, prompts={len(self._prompts)})"


# ════════════════════════════════════════════════════════════════════
#  DEMO
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os

    print("=" * 65)
    print("  TYPE 6: PROCEDURAL MEMORY DEMO")
    print("  'Knowing HOW to do things'")
    print("=" * 65)

    # Clean up
    pm_path = "/tmp/procedural_demo.json"
    if os.path.exists(pm_path):
        os.remove(pm_path)

    pm = ProceduralMemory(storage_path=pm_path)

    # ── 1. Add explicit procedures ──────────────────────────────────

    print("\n📋 PART 1: Storing Explicit Procedures (step-by-step workflows)")
    print("─" * 65)

    deploy_proc = pm.add_procedure(
        name="Deploy FastAPI to Production",
        description="Full deployment pipeline with safety checks and rollback",
        trigger="deploy application to production server",
        category="deployment",
        tags=["fastapi", "docker", "production"],
        steps=[
            {
                "order": 1,
                "instruction": "Run the full test suite",
                "tool": "bash",
                "tool_args": {"cmd": "pytest tests/ -v --tb=short"},
                "expected_output": "All tests passed",
                "fallback": "Fix failing tests. Do NOT proceed until green.",
            },
            {
                "order": 2,
                "instruction": "Build Docker image with version tag",
                "tool": "bash",
                "tool_args": {"cmd": "docker build -t app:$(git rev-parse --short HEAD) ."},
                "expected_output": "Image built successfully",
                "fallback": "Check Dockerfile for errors",
            },
            {
                "order": 3,
                "instruction": "Run smoke tests against the Docker container",
                "tool": "bash",
                "tool_args": {"cmd": "docker run -d -p 8000:8000 app && curl localhost:8000/health"},
                "expected_output": "Health check returns 200 OK",
                "fallback": "Check application logs for startup errors",
            },
            {
                "order": 4,
                "instruction": "Push image to container registry",
                "tool": "bash",
                "tool_args": {"cmd": "docker push registry.example.com/app:latest"},
                "expected_output": "Push complete",
            },
            {
                "order": 5,
                "instruction": "Deploy to Kubernetes with rolling update",
                "tool": "bash",
                "tool_args": {"cmd": "kubectl set image deployment/app app=app:latest"},
                "expected_output": "Rollout started",
            },
            {
                "order": 6,
                "instruction": "Monitor for 5 minutes for errors",
                "tool": "bash",
                "tool_args": {"cmd": "kubectl rollout status deployment/app --timeout=300s"},
                "expected_output": "Rollout completed successfully",
                "fallback": "kubectl rollout undo deployment/app (ROLLBACK)",
            },
        ]
    )
    print(f"  ✅ Added: {deploy_proc.name} ({len(deploy_proc.steps)} steps)")

    debug_proc = pm.add_procedure(
        name="Debug Memory Leak in Python",
        description="Systematic approach to finding and fixing memory leaks",
        trigger="debug memory leak python application",
        category="debugging",
        tags=["memory", "debugging", "python", "profiling"],
        steps=[
            {
                "order": 1,
                "instruction": "Enable tracemalloc at application startup",
                "tool": "code_edit",
                "tool_args": {"add": "import tracemalloc; tracemalloc.start()"},
                "expected_output": "tracemalloc enabled",
            },
            {
                "order": 2,
                "instruction": "Take memory snapshot before and after suspected operation",
                "tool": "bash",
                "tool_args": {"cmd": "python -c 'import tracemalloc; ...'"},
                "expected_output": "Two snapshots captured",
            },
            {
                "order": 3,
                "instruction": "Compare snapshots to find top memory allocations",
                "expected_output": "Top 10 memory-consuming lines identified",
            },
            {
                "order": 4,
                "instruction": "Check for common leak patterns: unbounded caches, circular refs, global lists",
                "expected_output": "Leak source identified",
                "fallback": "Use objgraph to visualize object reference graphs",
            },
            {
                "order": 5,
                "instruction": "Apply fix: add max size to caches, break circular refs, use weakrefs",
                "expected_output": "Memory usage stabilizes over time",
            },
        ]
    )
    print(f"  ✅ Added: {debug_proc.name} ({len(debug_proc.steps)} steps)")

    index_proc = pm.add_procedure(
        name="Build FAISS Vector Index",
        description="Create and optimize a FAISS index for semantic search",
        trigger="build vector index similarity search embeddings",
        category="data",
        tags=["faiss", "vectors", "search", "embeddings"],
        steps=[
            {
                "order": 1,
                "instruction": "Load and chunk documents (max 512 tokens per chunk)",
                "expected_output": "N chunks created from M documents",
            },
            {
                "order": 2,
                "instruction": "Generate embeddings using sentence-transformers (all-MiniLM-L6-v2)",
                "tool": "python",
                "tool_args": {"model": "all-MiniLM-L6-v2"},
                "expected_output": "Embeddings matrix: (N, 384)",
            },
            {
                "order": 3,
                "instruction": "Normalize vectors for cosine similarity",
                "tool": "python",
                "tool_args": {"code": "faiss.normalize_L2(embeddings)"},
                "expected_output": "All vectors L2-normalized",
            },
            {
                "order": 4,
                "instruction": "Create FAISS index (IndexFlatIP for <100K vectors, IndexIVFFlat for larger)",
                "expected_output": "Index created and populated",
                "is_conditional": True,
                "condition": "Choose index type based on dataset size",
            },
            {
                "order": 5,
                "instruction": "Add vectors in batches of 1000 to avoid memory spikes",
                "expected_output": "All vectors added to index",
            },
            {
                "order": 6,
                "instruction": "Save index to disk: faiss.write_index(index, 'vectors.faiss')",
                "expected_output": "Index saved, size reported",
            },
        ]
    )
    print(f"  ✅ Added: {index_proc.name} ({len(index_proc.steps)} steps)")

    # ── 2. Add behavioral templates (implicit procedures) ────────────

    print("\n🎭 PART 2: Storing Behavioral Templates (implicit procedures)")
    print("─" * 65)

    pm.add_prompt_template(
        name="Cautious Deployment Mode",
        prompt=(
            "You are assisting with a production deployment. Be extra cautious. "
            "Always confirm before destructive actions. Suggest running tests at every "
            "opportunity. If anything looks risky, flag it explicitly. Prefer rollback-safe "
            "strategies. Never skip verification steps even if the user is in a hurry."
        ),
        context="deploy production server release",
        effectiveness=0.9,
    )
    print("  ✅ Added: Cautious Deployment Mode")

    pm.add_prompt_template(
        name="Debugging Deep-Dive Mode",
        prompt=(
            "You are helping debug a complex issue. Be systematic and methodical. "
            "Start by reproducing the issue. Form hypotheses and test them one at a time. "
            "Ask clarifying questions about error messages and environment. Don't jump "
            "to conclusions. Show your reasoning step by step."
        ),
        context="debug error fix issue broken",
        effectiveness=0.85,
    )
    print("  ✅ Added: Debugging Deep-Dive Mode")

    pm.add_prompt_template(
        name="Learning & Explanation Mode",
        prompt=(
            "The user is learning a new concept. Use simple language with concrete examples. "
            "Build from basics to advanced. Use analogies the user would relate to. "
            "Check understanding before moving on. Provide runnable code examples. "
            "Celebrate progress and encourage experimentation."
        ),
        context="explain learn understand how does what is tutorial",
        effectiveness=0.88,
    )
    print("  ✅ Added: Learning & Explanation Mode")

    # ── 3. Find procedures for a task ────────────────────────────────

    print("\n🔍 PART 3: Finding the Right Procedure")
    print("─" * 65)

    queries = [
        "I need to deploy my app to production",
        "There's a memory leak in my Python service",
        "Build a search index for my documents",
    ]

    for query in queries:
        print(f"\n  Query: '{query}'")
        matches = pm.find_procedure(query)
        for proc, score in matches[:2]:
            print(f"    → [{score:.2f}] {proc.name} (v{proc.version}, "
                  f"used {proc.times_executed}x, reliability={proc.reliability_score:.0%})")

    # ── 4. Execute a procedure ───────────────────────────────────────

    print("\n🚀 PART 4: Executing a Procedure (dry run)")
    print("─" * 65)

    result = pm.execute_procedure(deploy_proc.id, dry_run=True)
    print(f"\n  Procedure: {result['procedure_name']}")
    print(f"  Status: {result['status']}")
    for step in result["steps"]:
        icon = "✅" if "success" in step["status"] else "❌"
        print(f"    {icon} Step {step['order']}: {step['instruction'][:50]}...")
    print(f"\n  Duration: {result['duration_sec']:.3f}s")
    print(f"  Success rate after: {result['success_rate_after']:.0%}")

    # ── 5. Refine from feedback ──────────────────────────────────────

    print("\n🔄 PART 5: Learning from Feedback")
    print("─" * 65)

    print(f"\n  Original: v{deploy_proc.version}, {len(deploy_proc.steps)} steps")

    refined = pm.refine_from_feedback(
        proc_id=deploy_proc.id,
        feedback="Should add a database migration step before deployment",
        rating="bad",
        modifications={
            "add_step": {
                "order": 3,  # Insert between build and smoke test
                "instruction": "Run database migrations (alembic upgrade head)",
                "tool": "bash",
                "tool_args": {"cmd": "alembic upgrade head"},
                "expected_output": "All migrations applied",
                "fallback": "alembic downgrade -1 (rollback last migration)",
            }
        }
    )
    print(f"  Refined:  v{refined.version}, {len(refined.steps)} steps "
          f"(new step added based on feedback)")
    print(f"  Parent:   {refined.parent_id} (version history preserved)")

    # ── 6. Auto-generate from episodes ───────────────────────────────

    print("\n🤖 PART 6: Auto-generating Procedures from Episodes")
    print("─" * 65)

    # Simulate episodic memory data
    fake_episodes = [
        {"task": "optimize database query", "action": "Added composite index on (user_id, created_at)",
         "result": "Query went from 2s to 0.01s", "outcome": "success",
         "lesson": "Always check EXPLAIN output first"},
        {"task": "optimize database query", "action": "Rewrote subquery as JOIN",
         "result": "Query went from 5s to 0.3s", "outcome": "success",
         "lesson": "JOINs usually outperform subqueries"},
        {"task": "optimize database query", "action": "Tried adding random index",
         "result": "No improvement, index unused", "outcome": "failure",
         "lesson": "Only index columns in WHERE/JOIN clauses"},
        {"task": "optimize database query", "action": "Added LIMIT and pagination",
         "result": "Response time consistent at 0.05s", "outcome": "success",
         "lesson": "Never return unbounded result sets"},
    ]

    generated = pm.generate_from_episodes(fake_episodes, min_occurrences=2)
    for proc in generated:
        print(f"\n  🆕 Auto-learned: {proc.name}")
        print(f"     Trigger: '{proc.trigger}'")
        print(f"     Steps:")
        for step in proc.steps:
            prefix = "    ⚠️" if step.get("is_conditional") else f"    {step.get('order', '?')}."
            print(f"     {prefix} {step.get('instruction', '')[:60]}")

    # ── 7. Context injection ─────────────────────────────────────────

    print("\n\n🪟 PART 7: Context Injection (what goes into the LLM prompt)")
    print("─" * 65)

    context = pm.get_context_for_prompt("deploy my application to production")
    print(context)

    # ── 8. Skill report ──────────────────────────────────────────────

    print("\n\n📊 PART 8: Agent Skill Report")
    print("─" * 65)

    report = pm.get_skill_report()
    print(f"  Total procedures:     {report['total_procedures']}")
    print(f"  Prompt templates:     {report['total_prompt_templates']}")
    print(f"  Total executions:     {report['total_executions']}")
    print(f"  Categories:           {report['categories']}")
    print(f"  Auto-learned skills:  {len(report['auto_learned'])}")

    if report["top_procedures"]:
        print(f"\n  🏆 Top procedures:")
        for p in report["top_procedures"]:
            print(f"     {p['name']} — reliability={p['reliability']}, used {p['executions']}x")

    print("\n" + "=" * 65)
    print("  💡 KEY INSIGHTS FOR YOUR INTERNSHIP:")
    print("=" * 65)
    print("""
  1. Procedural = HOW (workflows), Semantic = WHAT (facts),
     Episodic = WHAT HAPPENED (experiences). All three complement.

  2. Procedures have TWO forms:
     • Explicit: step-by-step tool-use workflows
     • Implicit: behavioral patterns (system prompts)

  3. The LEARNING LOOP is the most powerful part:
     Episodes (past experiences)
       → Pattern recognition (what works?)
       → Procedure generation (automate it)
       → Feedback refinement (improve it)
       → Versioned procedures (track evolution)

  4. OPTIMIZATION OPPORTUNITIES:
     • Procedure selection (which one to use?)
     • Step parallelization (which steps are independent?)
     • Adaptive feedback (auto-detect when a procedure fails)
     • Transfer learning (adapt procedures to new domains)
    """)

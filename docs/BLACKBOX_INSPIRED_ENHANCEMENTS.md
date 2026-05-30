# Enhancement Plan: Blackbox AI-Inspired Features for Multi-LLM Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-25  
> **Source:** Gap analysis between Blackbox AI and Multi-LLM Orchestrator v6.0  
> **Status:** Draft — complements all 10 prior enhancement plans

---

## Overview

Blackbox AI (4.2M+ VS Code installs) has two genuinely novel capabilities not seen in any of the 10 previously analyzed platforms: **Multi-Agent Competition** (multiple agents solve the SAME task, results compared) and **Sub-Agent Orchestration** (master spawns sub-agents for parallel decomposition). Also unique: encrypted GPU enclave models, voice interaction, and proactive error monitoring.

Five enhancements identified.

```
Phase Z1: Multi-Agent Competition with AI Judge                         (Highest ROI, 3-4 days)
Phase Z2: Sub-Agent Orchestration (Master Decompose → Spawn → Merge)    (High ROI, 3-4 days)
Phase Z3: Voice Mode for Hands-Free Development                        (Medium ROI, 2-3 days)
Phase Z4: Intelligent Provider Routing (Cost/Performance/Reliability)   (Medium ROI, 1-2 days)
Phase Z5: Proactive Error Monitoring (Scheduled Log Scan → Auto PR)     (Lower ROI, 2-3 days)
```

---

## What Blackbox AI Has — Quick Reference

| Feature | Description | Translates? |
|---------|-------------|:-----------:|
| **Multi-Agent Competition** | 2-5 agents solve same task, compare results, AI judge selects best | ✅ Phase Z1 |
| **Sub-Agent Orchestration** | Master decomposes task, spawns sub-agents, merges results | ✅ Phase Z2 |
| **Voice Mode** | ElevenLabs speech-to-text/text-to-speech, hands-free coding | ✅ Phase Z3 |
| **Provider Routing** | Route requests across providers for cost/performance/reliability | ✅ Phase Z4 |
| **Full Self-Coding** | Scheduled log monitoring → auto-detect errors → create PRs | ✅ Phase Z5 |
| **Encrypted GPU Enclave** | End-to-end encrypted AI with zero data retention | ✗ (Platform-specific) |
| **SMS/Slack Integration** | Task management via SMS and Slack | ✗ (Platform-specific) |
| **Vercel Auto Deploy** | Auto-deploy to Vercel from tasks | ✗ (Platform-specific) |
| **Desktop Agent** | Desktop app with extensions | Partial — CLI + IDE backend exist |
| **Mobile App** | Code on the go via iOS/Android app | ✗ (Different product category) |

---

## Phase Z1: Multi-Agent Competition with AI Judge

### Objective

Instead of using one model per task, run the **same task on 2-5 different agents simultaneously**, compare their outputs, and have an AI judge evaluate which solution is best along multiple dimensions (code quality, completeness, efficiency, error handling, documentation). This is fundamentally different from parallel agents working on different tasks.

### Current State

- `CritiqueCycle` — single model generates, different provider critiques
- `FallbackHandler` — switches models on failure, not on competition
- `Phase X2` (Parallel Agents) — multiple agents on DIFFERENT tasks
- `ARA Pipelines` — Debate (2-agent adversarial + judge) but for reasoning, not full code gen
- No mechanism to run multiple agents on the identical task and compare results

### Implementation

```python
"""
Multi-Agent Competition — same task, multiple agents, AI judge.
=================================================================

Run 2-5 agents simultaneously on the identical task, compare
outputs, and have an AI judge select the best solution.
"""

@dataclass
class AgentContender:
    """One agent in the competition."""
    agent_type: str  # "claude", "blackbox", "codex", "gemini"
    model: Model
    result: TaskResult | None = None
    execution_id: str = ""
    status: str = "pending"  # "pending", "running", "completed", "failed"


@dataclass
class CompetitionResult:
    """Results of a multi-agent competition."""
    task_id: str
    contenders: list[AgentContender]
    judge_evaluation: dict[str, float]  # contender_id → score
    winner: AgentContender | None
    comparison_metrics: dict

    # Judge's reasoning
    judge_rationale: str = ""

    # Merge strategy
    best_aspects: dict[str, str]  # aspect → which contender was best


class CompetitiveAgentOrchestrator:
    """Runs the same task on multiple agents and selects the best.

    Configuration:
        min_contenders: Minimum agents to compete (default: 2)
        max_contenders: Maximum agents to compete (default: 5)
        evaluation_dimensions: What the judge scores on
        merge_strategy: "winner_takes_all" or "best_aspects" or "ensemble"
    """

    EVALUATION_DIMENSIONS = [
        "code_quality",       # Readability, maintainability, best practices
        "completeness",       # Fully addresses the prompt
        "efficiency",         # Performance and resource usage
        "error_handling",     # Edge cases and robustness
        "documentation",      # Comments, explanations, types
        "test_quality",       # Test coverage and quality
        "consistency",        # Adherence to project conventions
    ]

    def __init__(
        self,
        client: UnifiedClient,
        max_contenders: int = 3,
        merge_strategy: str = "best_aspects",
    ):
        self._client = client
        self._max_contenders = max_contenders
        self._merge_strategy = merge_strategy

    async def compete(
        self,
        task: Task,
        contenders: list[tuple[str, Model]],
    ) -> CompetitionResult:
        """Run the same task on multiple agents and select the best.

        Args:
            task: The task to execute
            contenders: List of (agent_type, model) tuples

        Returns:
            CompetitionResult with winner and judge evaluation
        """
        # Validate contender count
        if len(contenders) < 2:
            raise ValueError("Minimum 2 contenders required")
        contenders = contenders[:self._max_contenders]

        # Create contender slots
        entries = [
            AgentContender(agent_type=at, model=model)
            for at, model in contenders
        ]

        # Run all contenders in parallel
        tasks = [
            self._execute_contender(entry, task)
            for entry in entries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update entries with results
        for entry, result in zip(entries, results):
            if isinstance(result, Exception):
                entry.status = "failed"
            else:
                entry.result = result
                entry.status = "completed"

        # Judge the competition
        judge_result = await self._judge_competition(task, entries)

        # Find winner
        winner = max(entries, key=lambda e: judge_result.get(e.execution_id, 0))

        return CompetitionResult(
            task_id=task.id,
            contenders=entries,
            judge_evaluation=judge_result,
            winner=winner,
            comparison_metrics=self._build_comparison(entries),
            judge_rationale=judge_result.get("rationale", ""),
        )

    async def _execute_contender(
        self, contender: AgentContender, task: Task
    ) -> TaskResult:
        """Execute a single contender on the task."""
        response = await self._client.call(
            model=contender.model,
            prompt=task.prompt,
            max_tokens=task.max_output_tokens,
        )
        contender.execution_id = response.id or str(uuid4())[:8]
        return TaskResult(
            task_id=task.id,
            output=response.text,
            model_used=contender.model,
            cost_usd=response.cost_usd,
            tokens_used={
                "input": response.input_tokens,
                "output": response.output_tokens,
            },
        )

    async def _judge_competition(
        self, task: Task, contenders: list[AgentContender]
    ) -> dict:
        """Have an AI judge evaluate all contenders.

        Uses a neutral model (not any of the contenders) as the judge.
        """
        judge_prompt = self._build_judge_prompt(task, contenders)
        judge_response = await self._client.call(
            model=self._get_judge_model(contenders),  # Neutral model
            prompt=judge_prompt,
            max_tokens=1000,
            temperature=0.1,
        )
        return self._parse_judge_response(judge_response.text)

    def _build_judge_prompt(
        self, task: Task, contenders: list[AgentContender]
    ) -> str:
        """Build the judge's evaluation prompt."""
        prompt = (
            f"Evaluate {len(contenders)} solutions to the same task. "
            f"Score each on a scale of 0-10 for these dimensions:\n"
            f"{chr(10).join(f'- {d}' for d in self.EVALUATION_DIMENSIONS)}\n\n"
            f"## Task\n{task.prompt}\n\n"
        )

        for i, contender in enumerate(contenders):
            prompt += (
                f"## Solution {i + 1} ({contender.agent_type})\n"
                f"```\n{contender.result.output[:3000]}\n```\n\n"
            )

        prompt += (
            "Return JSON:\n"
            '{\n'
            '  "scores": {"solution_1": 8.5, "solution_2": 7.2},\n'
            '  "winner": "solution_1",\n'
            '  "rationale": "Solution 1 is better because...",\n'
            '  "dimension_scores": {\n'
            '    "solution_1": {"code_quality": 9, "completeness": 8},\n'
            '    "solution_2": {"code_quality": 7, "completeness": 9}\n'
            '  }\n'
            '}'
        )

        return prompt

    def _get_judge_model(self, contenders: list[AgentContender]) -> Model:
        """Select a neutral model as judge — not used by any contender."""
        contender_models = {c.model for c in contenders}
        judge_candidates = [Model.GPT_5, Model.CLAUDE_SONNET_4_6, Model.GEMINI_2_5_PRO]
        for model in judge_candidates:
            if model not in contender_models:
                return model
        return Model.GPT_5  # Fallback
```

**Use cases:**
```
# Compare approaches for critical features
orchestrator.compete(
    task="Implement Stripe payment processing",
    contenders=[
        ("claude", Model.CLAUDE_SONNET_4_6),
        ("codex", Model.GPT_5_4_CODEX),
        ("gemini", Model.GEMINI_2_5_PRO),
    ]
)

# Validate refactoring across agents
orchestrator.compete(
    task="Refactor auth module to use dependency injection",
    contenders=[
        ("claude", Model.CLAUDE_SONNET_4_6),
        ("blackbox", Model.XIAOMI_MIMO_V2_PRO),
    ]
)
```

**File changes:**
- NEW: `orchestrator/competitive_agent.py` (~500 lines)
- MODIFY: `orchestrator/engine.py` — add `compete()` method
- MODIFY: `orchestrator/cli.py` — add `compete` command
- MODIFY: `orchestrator/api_clients.py` — support concurrent multi-model execution

#### Verification Gate

```bash
# Run same task on 3 agents, verify judge selects winner
python -m orchestrator compete --task "Add JWT auth" --agents claude,codex,gemini
# Verify: 3 solutions, judge evaluation, winner selected, rationale provided
```

---

## Phase Z2: Sub-Agent Orchestration

### Objective

When a task is too large for a single agent execution context, the master agent decomposes it into sub-tasks, spawns sub-agents (each with its own terminal/context), monitors progress, and consolidates results. Similar to Blackbox AI's Orchestration feature.

### Current State

- `Orchestrator._decompose()` — decomposes project into tasks
- `Orchestrator._execute_all()` — executes tasks sequentially or in parallel
- Tasks execute in the same orchestrator process, not as spawned sub-agents
- No concept of a "master agent" that spawns and coordinates sub-agents

### Implementation

```python
"""
Sub-Agent Orchestration — master agent spawns and coordinates sub-agents.
============================================================================

For large features: spawn specialized sub-agents for different components.
Master monitors progress, resolves conflicts, and consolidates results.
"""

@dataclass
class SubAgent:
    """A spawned sub-agent working on a component of a larger task."""
    id: str
    task_description: str
    agent_type: str
    model: Model
    context: str  # What files/types/patterns this sub-agent needs to know
    status: str = "pending"
    result: str = ""
    modified_files: list[str] = field(default_factory=list)


@dataclass
class OrchestrationPlan:
    """A plan decomposed by the master agent for sub-agent execution."""
    original_task: str
    sub_tasks: list[SubAgent]
    integration_strategy: str  # "sequential" or "parallel"
    conflict_resolution: str  # "first_wins", "manual", "ai_merge"


class SubAgentOrchestrator:
    """Master agent that decomposes and coordinates sub-agents.

    Workflow:
    1. Master agent analyzes the large task
    2. Decomposes into component sub-tasks
    3. Spawns sub-agents with appropriate context
    4. Monitors sub-agent progress
    5. Detects conflicts (same file modified by multiple sub-agents)
    6. Consolidates results into a unified solution
    7. Runs integration tests
    """

    async def orchestrate(
        self, task_description: str, project_context: str
    ) -> OrchestrationPlan:
        """Plan the orchestration — decompose task into sub-agents."""

    async def spawn_sub_agents(
        self, plan: OrchestrationPlan
    ) -> list[SubAgent]:
        """Spawn sub-agents to execute decomposed tasks."""

    async def monitor_progress(
        self, sub_agents: list[SubAgent]
    ) -> None:
        """Monitor sub-agent execution, detect conflicts, resolve issues."""

    async def consolidate_results(
        self, sub_agents: list[SubAgent]
    ) -> str:
        """Merge all sub-agent outputs into a unified solution."""

    async def run_integration_tests(
        self, consolidated_output: str
    ) -> bool:
        """Verify the consolidated solution works end-to-end."""
```

**File changes:**
- NEW: `orchestrator/sub_agent.py` (~400 lines)
- MODIFY: `orchestrator/engine.py` — add `orchestrate()` method for large tasks

#### Verification Gate

```bash
# Run orchestration on a large e-commerce feature
python -m orchestrator orchestrate --task "Add checkout flow with payment, email confirmation, and order tracking"
# Verify: decomposed into 3 sub-agents, parallel execution, consolidated result
```

---

## Phase Z3: Voice Mode

### Objective

Interact with the orchestrator through voice — speak prompts, hear responses, control execution via voice commands. Uses ElevenLabs for speech-to-text/text-to-speech.

### Current State

- All interaction is text-based (CLI, chatbox, prompts)
- No speech-to-text or text-to-speech capability
- ElevenLabs API exists as a potential integration

### Implementation

```python
class VoiceMode:
    """Voice interaction for the orchestrator.

    Speech → Text → Orchestrator → Text → Speech
    """

    def __init__(self, stt_model: str = "elevenlabs", tts_model: str = "elevenlabs"):
        self._stt = ElevenLabsSTT()
        self._tts = ElevenLabsTTS()
        self._active = False
        self._muted = False

    async def listen(self) -> str:
        """Capture speech and convert to text."""

    async def speak(self, text: str) -> None:
        """Convert text to speech and play audio."""

    async def voice_prompt(self) -> str:
        """Full voice interaction loop: listen → process → speak."""
```

**File changes:**
- NEW: `orchestrator/voice_mode.py` (~200 lines)
- MODIFY: `ide_frontend/src/components/VoiceButton.tsx` — UI component

---

## Phase Z4: Intelligent Provider Routing

### Objective

Route requests across providers based on cost, performance, and reliability metrics. Not a simple model selection table — dynamic routing that optimizes for different goals (cheapest, fastest, most reliable).

### Current State

- `ROUTING_TABLE` — static mapping of TaskType → Model
- `FallbackHandler` — health-based routing
- `ObserveService` — per-model metrics but not used for routing
- No dynamic routing based on live cost/performance data

### Implementation

```python
class ProviderRouter:
    """Intelligent routing across providers.

    Strategies:
    - cheapest: Route to the lowest-cost provider that meets quality threshold
    - fastest: Route to the lowest-latency provider
    - most_reliable: Route to the provider with lowest error rate
    - balanced: Weighted combination of all factors
    """

    def __init__(self, strategy: str = "balanced"):
        self._strategy = strategy
        self._cost_data: dict[str, float] = {}
        self._latency_data: dict[str, float] = {}
        self._error_rates: dict[str, float] = {}

    def select_provider(self, task_type: TaskType, quality_threshold: float = 0.7) -> Model:
        """Select the optimal provider based on routing strategy."""
```

**File changes:**
- MODIFY: `orchestrator/model_selector.py` — add dynamic routing strategies

---

## Phase Z5: Proactive Error Monitoring

### Objective

Schedule periodic log scanning of deployed/generated projects. When errors are detected, automatically create a fix task and optionally open a PR.

### Current State

- `AutoFixer` (V5) — reactive: fix when user clicks button
- No proactive monitoring
- No scheduled error detection

### Implementation

```python
class ProactiveMonitor:
    """Scheduled log monitoring → auto-detect errors → create fix PRs."""

    def __init__(self, client: UnifiedClient, scheduler: AsyncIOScheduler):
        self._client = client
        self._scheduler = scheduler

    async def schedule_scan(
        self, log_source: str, interval_minutes: int = 30
    ) -> None:
        """Schedule periodic log scanning."""

    async def scan_logs(self, log_source: str) -> list[str]:
        """Scan logs for errors using AI analysis."""

    async def auto_fix_and_pr(
        self, errors: list[str], repo_url: str
    ) -> str:
        """Fix detected errors and create a PR."""
```

---

## Integration

```
Phase Z1 (Multi-Agent Competition) — independent, uses existing models + CritiqueCycle judge
Phase Z2 (Sub-Agent Orchestration) — independent, extends task decomposition
Phase Z3 (Voice Mode) — independent UI enhancement
Phase Z4 (Provider Routing) — extends existing model_selector
Phase Z5 (Proactive Monitoring) — depends on V5 (AutoFixer) + B3 (Automations)
```

## Effort Estimate

| Phase | Days |
|-------|------|
| Z1 | 3-4 |
| Z2 | 3-4 |
| Z3 | 2-3 |
| Z4 | 1-2 |
| Z5 | 2-3 |
| **Total** | **11-16** |

---

## Updated Grand Total (All 11 Sources)

| # | Source | Phases | Days |
|---|--------|--------|------|
| 1-10 | Prior 10 platforms | 61 | 132-190 |
| 11 | Blackbox AI | 5 | 11-16 |
| **Grand Total** | **11 platforms** | **66** | **143-206** |

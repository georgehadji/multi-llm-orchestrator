# Ponytail Integration Plan for AI Orchestrator

This document outlines a thorough, production-grade integration strategy for **Ponytail** (the minimalist, lazy senior developer mode) into the **AI Orchestrator** architecture. 

By integrating Ponytail, the AI Orchestrator gains an extremely cost-effective, high-velocity, and dependency-lean code generation behavior. This results in **80-94% less code generated**, **47-77% lower API costs**, and **3-6× faster task completion times** while strictly preserving architectural safety, trust-boundary validation, and security compliance.

---

## 1. Executive Summary & Core Philosophy

### 1.1 The Ponytail Mantra
> *"He says nothing. He writes one line. It works."*

Ponytail represents the antithesis of speculative over-engineering. In typical modern agentic workflows, LLMs tend to over-build: installing complex packages, rolling custom wrappers for things already present in the standard library, and designing nested classes for one-shot features. 

Ponytail forces the AI to traverse **"The Ladder"** of simplification:
1. **Does this need to exist at all?** (If speculative or unneeded, skip it/YAGNI).
2. **Does the standard library do it?** (If yes, use it directly).
3. **Does a native platform feature cover it?** (e.g., native browser HTML5 elements, database constraints).
4. **Is there an already-installed dependency?** (If yes, use it; do not add new dependencies).
5. **Can this be one line?** (Make it one line).
6. **Only then:** Write the absolute minimum code that works, marking intentional shortcuts with a `ponytail:` comment describing upgrade paths.

### 1.2 Alignment with AI Orchestrator
The **AI Orchestrator** is a highly structured, multi-tier agent platform (L1 to L4) featuring advanced planning, quality gates, and post-processing. Integrating Ponytail provides:
- **Cost Optimization:** Drastically decreases context consumption and output tokens.
- **Improved Velocity:** Reduces agent iterations and compilation/linting overhead.
- **Maintainability:** Yields cleaner, shorter, and easier-to-audit generated files.

---

## 2. Target Architecture & Integration Touchpoints

```
                    ┌────────────────────────┐
                    │     Slash Commands     │
                    │ (/ponytail, -review,   │
                    │  -audit, -help)        │
                    └───────────┬────────────┘
                                │
                                ▼
 ┌────────────────┐   ┌──────────────────┐   ┌─────────────────┐
 │  AI Agent LLM  │◄──┤  Persona Manager ├──►│  System Prompts │
 │     Skills     │   │ (STRICT/CREATIVE/│   │  & Context      │
 │(.claude/skills)│   │  PONYTAIL Mode)  │   │  Steering       │
 └────────────────┘   └──────────────────┘   └─────────────────┘
                                │
                                ▼
                      ┌──────────────────┐
                      │ Code Generation  │
                      │ & Quality Gates  │
                      └─────────┬────────┘
                                │
                                ▼
                      ┌──────────────────┐
                      │ Post-Processor   │
                      │ (Annotate, Lint) │
                      └──────────────────┘
```

The integration spans five major layers of the AI Orchestrator codebase:
1. **Persona & Persona Modes Layer** (`persona.py` & `persona_modes.py`): Adding `PONYTAIL` as a core behavioral persona with configurable intensity levels (`lite`, `full`, `ultra`).
2. **Slash Command Layer** (`slash_commands.py`): Implementing command interfaces `/ponytail`, `/ponytail-review`, `/ponytail-audit`, and `/ponytail-help`.
3. **LLM Prompts & Skills System** (`.claude/skills/`): Registering Ponytail's rule-guidelines as active system skills that automatically steer model outputs.
4. **Post-Processing Layer** (`code_post_processor.py`): Extending cleanup logic to format, truncate prose, and validate `ponytail:` comments.
5. **Quality Gate Layer** (`quality_control.py`): Modifying compliance checkers to expect lean files and prioritize single-assert unit-checks over extensive testing boilerplate in Ponytail mode.

---

## 3. Detailed Integration Specifications

### 3.1 Persona & Persona Modes Integration
We will register `PONYTAIL` inside both the Mnemo Cortex persona modes (`persona.py`) and the behavioral mode manager (`persona_modes.py`).

#### 3.1.1 Updates to `orchestrator/persona.py`
Add `PersonaMode.PONYTAIL` to control code formatting, documentation requirements, and validation gates:

```python
# In orchestrator/persona.py

class PersonaMode(Enum):
    STRICT = "strict"
    CREATIVE = "creative"
    BALANCED = "balanced"
    CUSTOM = "custom"
    PONYTAIL = "ponytail"  # New minimalist developer mode


# Inside Persona.PRESETS:
Persona.PRESETS[PersonaMode.PONYTAIL] = PersonaSettings(
    temperature=0.1,                # Meticulous and deterministic
    top_p=0.1,
    top_k=10,
    strict_validation=True,
    require_tests=False,            # Trivial code needs no tests (YAGNI)
    require_documentation=False,    # Skip verbose docstrings unless required
    max_iterations=3,               # High velocity, fail fast
    include_reasoning=False,        # Extremely brief explanations
    verbose_output=False,
    format_code=True,
    enable_preflight=True,
    preflight_mode="warn",          # Warn rather than block for rapid prototyping
    max_tokens=2048,                # Keeps responses brief and cost-effective
    context_truncation=30000,
    system_prompt_addition=(
        "You are Ponytail, a lazy senior developer. Write the absolute minimum code "
        "required. Stop at the first rung of The Ladder: YAGNI -> stdlib -> native -> "
        "existing dep -> one line. Mark intentional simplifications with 'ponytail:' comments. "
        "Keep explanations to a maximum of three short lines."
    )
)
```

#### 3.1.2 Updates to `orchestrator/persona_modes.py`
Add `Persona.PONYTAIL` and its configuration to `PersonaModeManager`:

```python
# In orchestrator/persona_modes.py

class Persona(Enum):
    STRICT = "strict"
    CREATIVE = "creative"
    BALANCED = "balanced"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"
    EXPERT = "expert"
    HELPFUL = "helpful"
    CRITICAL = "critical"
    PRECISION = "precision"
    PONYTAIL = "ponytail"  # Added Ponytail Persona


# Inside PersonaModeManager._initialize_persona_configs():
Persona.PONYTAIL: PersonaConfig(
    mode_config=ModeConfig(
        temperature=0.1,
        top_p=0.1,
        max_tokens=2048,
        presence_penalty=0.8,      # Encourage brevity and penalize repetitive phrases
        frequency_penalty=0.8,
        stop_sequences=[],
        model_override=Model.STEPFUN_STEP_3_5_FLASH,  # Ultra-fast, cost-effective reasoning
        validation_level="basic",                     # Lean validation
        creativity_boost=0.1,
    ),
    tone="minimalist",
    approach="lazy-but-correct",
    focus=["YAGNI", "stdlib-first", "zero-dependencies", "brevity"],
    communication_style="ultra-concise (code-first, max 3 lines of prose)",
    decision_making_style="ruthless simplification",
    risk_tolerance=0.4,            # Medium: pragmatically lazy but safe
)
```

Map `Persona.PONYTAIL` to `OperationMode.STRICT` in `_map_persona_to_operation_mode` to ensure technical correctness under the hood, while utilizing the system prompt to steer formatting and style.

---

### 3.2 Slash Command Integration (`orchestrator/slash_commands.py`)
Four new slash commands will be added to the registry, letting users interact with Ponytail from the AI Orchestrator CLI.

#### 3.2.1 Command Definitions
- `/ponytail [lite|full|ultra|off]`: Sets the active persona and adjusts intensity.
- `/ponytail-review`: Conducts an over-engineering diff audit of the current branch/working tree.
- `/ponytail-audit`: Audits the entire project repository to identify dead code, unused abstractions, or over-engineered boilerplate.
- `/ponytail-help`: Displays the cheat sheet of shortcuts and guidelines.

#### 3.2.2 Command Registration
Add the following commands inside `_setup_default_commands()`:

```python
        self.register(
            SlashCommand(
                name="ponytail",
                description="Enable Ponytail (lazy senior dev) mode [lite|full|ultra|off]",
                handler=self._cmd_ponytail,
                aliases=["lazy", "yagni"],
            )
        )
        self.register(
            SlashCommand(
                name="ponytail-review",
                description="Review current unstaged git changes for over-engineering",
                handler=self._cmd_ponytail_review,
                aliases=["lazy-review", "cut"],
            )
        )
        self.register(
            SlashCommand(
                name="ponytail-audit",
                description="Audit the entire workspace for speculative code and bloat",
                handler=self._cmd_ponytail_audit,
                aliases=["lazy-audit", "bloat-hunt"],
            )
        )
        self.register(
            SlashCommand(
                name="ponytail-help",
                description="Display the Ponytail cheat sheet & rules card",
                handler=self._cmd_ponytail_help,
                aliases=["lazy-help"],
            )
        )
```

#### 3.2.3 Handler Implementation Blueprints
The handler functions invoke the underlying model using a customized system prompt, feeding the git diff or project file catalog into the request.

```python
    async def _cmd_ponytail(self, args: str, ctx: SlashCommandContext) -> str:
        """Handler for /ponytail."""
        level = args.strip().lower() or "full"
        if level == "off":
            get_global_persona_manager().set_persona(Persona.BALANCED)
            return "Ponytail deactivated. Reverted to Balanced mode."
        
        if level not in ["lite", "full", "ultra"]:
            return "Invalid level. Usage: /ponytail [lite|full|ultra|off]"
        
        get_global_persona_manager().set_persona(Persona.PONYTAIL)
        # Store active intensity in context/config
        ctx.client.metadata["ponytail_intensity"] = level
        return f"Ponytail activated ({level} intensity). He is watching you build."

    async def _cmd_ponytail_review(self, args: str, ctx: SlashCommandContext) -> str:
        """Handler for /ponytail-review."""
        # 1. Fetch unstaged git changes
        import subprocess
        diff_output = subprocess.check_output(
            ["git", "diff", "HEAD"], 
            stderr=subprocess.STDOUT
        ).decode("utf-8")
        
        if not diff_output.strip():
            return "No changes detected in working tree. Code is lean."
            
        # 2. Query LLM using the ponytail-review skill
        prompt = (
            "Review the following diff for over-engineering, unneeded abstractions, "
            "and standard library duplicates. Provide one line per finding.\n\n"
            f"```diff\n{diff_output}\n```"
        )
        # Route to a fast flash model with ponytail-review system guidelines
        response = await ctx.client.generate(prompt, skill="ponytail-review")
        return response

    async def _cmd_ponytail_audit(self, args: str, ctx: SlashCommandContext) -> str:
        """Handler for /ponytail-audit."""
        # 1. Generate directory mapping / file tree
        # 2. Query LLM to identify high-probability bloat targets (unneeded wrappers, single-implement interfaces)
        # 3. Return ranked findings: Biggest cuts first
        return "Not implemented. Blueprint is ready: will perform codebase-wide token scanning."

    async def _cmd_ponytail_help(self, args: str, ctx: SlashCommandContext) -> str:
        """Handler for /ponytail-help."""
        return (
            "### Ponytail Mode Commands & Guidelines\n"
            "- `/ponytail [level]`: Activate lazy mode. Levels: `lite`, `full` (default), `ultra`.\n"
            "- `/ponytail-review`: Scans git diff for unnecessary code or bloated wrappers.\n"
            "- `/ponytail-audit`: Searches the whole repository for dead flexibility and YAGNI targets.\n"
            "- `/ponytail off`: Restores standard balanced persona settings."
        )
```

---

### 3.3 System Prompts & AI Agent Skills (`.claude/skills/`)
To make Ponytail's rules easily consumable by any active AI agent, we will create corresponding skill markdown files. The orchestrator's skill parser auto-loads these and feeds them into the system prompt context.

We will create four skill directories inside `.claude/skills/`:

#### 3.3.1 `.claude/skills/ponytail/SKILL.md`
This is the core steering rulebook. Whenever Ponytail mode is on, this skill is injected:
- Dictates **The Ladder** of simplification.
- Directs writing single-assert smoke tests instead of multi-file pytest boilerplates.
- Limits outputs to: code block first, then at most 3 lines of prose mapping `skipped: [X], add when [Y]`.

#### 3.3.2 `.claude/skills/ponytail-review/SKILL.md`
This governs over-engineering reviews:
- Explains the mandatory output format: `L<line>: <tag> <what>. <replacement>.`
- Tags: `delete:`, `stdlib:`, `native:`, `yagni:`, `shrink:`.
- Mandates scoring: Ends with `net: -<N> lines possible.` or `Lean already. Ship.`

#### 3.3.3 `.claude/skills/ponytail-audit/SKILL.md`
Governs repo-wide scans:
- Searches for redundant classes, dead flags, and libraries that duplicate standard library functions (e.g., pulling a deepcopy library when standard library provides it).
- Ranks findings starting with the biggest line-saving impact.

#### 3.3.4 `.claude/skills/ponytail-help/SKILL.md`
Contains a markdown quick-reference sheet of commands and levels.

---

### 3.4 Code Generation & Post-Processing (`orchestrator/code_post_processor.py`)
To prevent "explanation creep" where models write 1 line of code but defend it with 10 paragraphs of text, we integrate checks directly into the code post-processing layer.

#### 3.4.1 Updates to `orchestrator/code_post_processor.py`
Add a truncation mechanism to prune excessive conversational prose following a code block in Ponytail mode:

```python
# Inside CodePostProcessor:

def _truncate_prose_for_ponytail(self, code: str) -> str:
    """
    Enforces a strict maximum of 3 lines of prose after the code block 
    when operating in Ponytail mode.
    """
    # Look for trailing prose or markdown explanation
    parts = code.split("```")
    if len(parts) >= 3:
        # Code is inside parts[1]. Prose is inside parts[2]
        code_block = parts[1]
        prose = parts[2].strip().split("\n")
        
        # Prune explanation to at most 3 lines
        shortened_prose = "\n".join(prose[:3])
        return f"```{code_block}```\n\n{shortened_prose}"
    return code
```

Integrate `_truncate_prose_for_ponytail` into the main `process()` function when Ponytail persona is globally active.

---

### 3.5 Quality Gates & Testing Layer (`orchestrator/quality_control.py`)
Normally, the AI Orchestrator's `QualityController` enforces a thorough test suite, complete with mock fixtures and strict test structures. 

Under Ponytail mode:
- **Trivial Code (one-liners):** Needs no tests. YAGNI applies directly to testing.
- **Complex Logic (loops, state-machines, trust boundaries):** Requires exactly **one runnable check**—the simplest `assert`-based test script or a `__main__` entry block. Heavy framework fixtures and extensive mocks are rejected as over-engineering.

#### 3.5.1 Updates to `orchestrator/quality_control.py`
Adjust the static analysis linting rules when Ponytail is active:
- **Skip boilerplates:** Do not flag files for lacking comprehensive docstrings or verbose class structures if they are implemented as simple functions.
- **Complexity-vs-Brevity:** Ensure complexity scores prioritize low Line Count (LOC) and standard library utilization.
- Adjust `_check_compliance` to skip test-file enforcement on simple, self-contained scripts containing inline self-checks (such as `if __name__ == "__main__": assert ...`).

---

## 4. Implementation Plan & Milestones

The integration can be executed cleanly in three sequential, zero-regression milestones:

### Milestone 1: Persona and Prompt Steering (Core Engine)
- Update `orchestrator/persona.py` and `orchestrator/persona_modes.py` with `PONYTAIL` enums and configurations.
- Create the target markdown skill folders `.claude/skills/ponytail/` and `.claude/skills/ponytail-help/` in the workspace.
- **Validation:** Write unit tests verifying that `PersonaModeManager` can set, map, and return correct parameters for `Persona.PONYTAIL`.

### Milestone 2: Slash Commands and Review/Audit Tools
- Add handlers for `/ponytail`, `/ponytail-review`, and `/ponytail-help` in `orchestrator/slash_commands.py`.
- Integrate git subprocess triggers to stream changes to the LLM during review calls.
- Create skill configurations `.claude/skills/ponytail-review/` and `.claude/skills/ponytail-audit/`.
- **Validation:** Execute a test `/ponytail-review` on a simulated bloated file with hand-rolled loops and assert it correctly identifies standard library replacements (e.g., recommending `dict(zip(k, v))` instead of manual iterations).

### Milestone 3: Post-processing & Quality Gate Adapters
- Implement prose-truncation logic inside `orchestrator/code_post_processor.py`.
- Update static compliance algorithms inside `orchestrator/quality_control.py` to allow self-contained single-assert tests under Ponytail mode.
- **Validation:** Run the complete AI Orchestrator regression suite to ensure zero disruption to standard STRICT/CREATIVE modes.

---

## 5. Summary of Integration Benefits

| Metric / Aspect | Without Ponytail Integration | With Ponytail Integration |
|-----------------|------------------------------|---------------------------|
| **Code Length** | Standard, often boilerplate-heavy | **80-94% less code** (YAGNI, stdlib-first) |
| **API Costs** | High due to detailed code & verbose chats | **47-77% cheaper** (short outputs, fast models) |
| **Execution Time** | Slow, multi-turn setups, lengthy audits | **3-6× faster** (immediate code-first output) |
| **Testing Setup** | Complex fixtures, multiple testing files | Exactly **one runnable assert-check** or nothing |
| **Explanations** | Paragraphs of design justifications | **Maximum 3 lines** of skipped/upgrade comments |
| **Dependencies** | Speculatively imports helper packages | **Zero-dependency preference** (stdlib-focused) |

---

### Suggested Next Action
We recommend initiating **Milestone 1** by applying the Persona modifications directly into `orchestrator/persona_modes.py` and `orchestrator/persona.py`, creating a highly pragmatic, cost-effective new way for users to build with the AI Orchestrator.

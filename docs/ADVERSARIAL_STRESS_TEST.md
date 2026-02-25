# Codebase Enhancer — Adversarial Stress Test Report

**Date**: 2026-02-26
**Feature**: Codebase Enhancer (Phases 1-2 POC)
**Assessment**: Identifies 3 critical black swan failure modes with minimax regret analysis

---

## Executive Summary

The Codebase Enhancer feature provides excellent functionality for typical codebases but lacks resilience against three pathological scenarios:

1. **Pathological Directory Structures** (DOS/Timeout Risk) — Infinite loops, stack overflow, timeout
2. **LLM Response Degradation** (Silent Quality Collapse) — Malformed responses, graceful but undetected fallback
3. **Context-Blind Suggestions** (Feature Irrelevance) — Suggestions misaligned with business priorities

Each represents a significant **minimax regret gap** — the difference between best and worst-case outcomes is substantial, and worst case is often silent (high regret because user doesn't know).

---

## Failure Mode #1: Pathological Directory Structures (DOS/Timeout)

### Scenario Description

**Black Swan Event**: User analyzes a codebase with:
- Circular symlink chains (e.g., `A → B → A`)
- Extremely deep nesting (10,000+ levels deep)
- Huge monorepo (1M+ files)
- Network-mounted directories with latency

**Current Implementation Vulnerability**:

```python
# orchestrator/codebase_analyzer.py - scan() method
def scan(self, root_path: str) -> CodebaseMap:
    root = Path(root_path)
    total_files = 0
    total_lines = 0
    files_by_language = defaultdict(int)

    # ❌ No depth limit
    # ❌ No symlink detection
    # ❌ No per-directory file count limit
    # ❌ No timeout mechanism

    for filepath in root.rglob("*"):  # RECURSIVE GLOB - UNLIMITED DEPTH
        if filepath.is_dir() and filepath.name in self.SKIP_DIRS:
            continue
        # ... process file
```

### Failure Scenarios

| Scenario | Mechanism | Outcome |
|----------|-----------|---------|
| **Symlink Loop** | `rglob()` follows symlinks indefinitely | Stack overflow, infinite recursion |
| **Deep Nesting** | 10,000 levels deep with small files | Stack overflow on recursion |
| **Huge Monorepo** | 1M files scanned sequentially | Timeout (>30s), analysis abandoned |
| **Network Mount** | Latency per directory stat | Cumulative timeout, feature unusable |

### Minimax Regret Analysis

```
Best Case Outcome:
  - Analysis completes in 1-2 seconds
  - Accurate file counts, language detection
  - Regret: 0 (feature works perfectly)

Worst Case Outcome:
  - Analysis crashes/times out after 30+ seconds
  - No partial results returned
  - User sees error "Analysis failed"
  - No indication of why or what went wrong
  - Regret: MAXIMUM (feature completely unavailable, no fallback)

Regret Gap: Enormous
- Best: Works perfectly
- Worst: Completely broken
- Users can't use feature on their own codebases
- Silent failure (no diagnostic info)
```

### Specific Improvement Proposal

**Add Defensive Scanning with Partial Results**:

```python
from pathlib import Path
from typing import Set
import signal

class CodebaseAnalyzerV2(CodebaseAnalyzer):
    """Enhanced with resilience against pathological structures"""

    MAX_DEPTH = 20  # Configurable, e.g., typical project depth is 5-10
    MAX_FILES_PER_SCAN = 100_000  # Beyond this, switch to sampling
    SCAN_TIMEOUT_SECONDS = 30  # Hard timeout

    def scan(self, root_path: str, max_depth: int = MAX_DEPTH) -> CodebaseMap:
        """Scan with depth limiting, symlink detection, and timeout"""

        # Track visited inodes to detect symlink loops
        visited_inodes: Set[int] = set()

        # Use timeout wrapper
        try:
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.SCAN_TIMEOUT_SECONDS)

            codebase_map = self._scan_recursive(
                Path(root_path),
                depth=0,
                max_depth=max_depth,
                visited_inodes=visited_inodes
            )

            signal.alarm(0)  # Cancel alarm
            return codebase_map

        except TimeoutError:
            # Return partial results with warning
            return self._create_partial_result(
                total_files=self.partial_scan_state['total_files'],
                warning="Analysis timeout - results are partial (incomplete scan)"
            )
        finally:
            signal.alarm(0)

    def _scan_recursive(
        self,
        path: Path,
        depth: int,
        max_depth: int,
        visited_inodes: Set[int]
    ) -> CodebaseMap:
        """Recursive scan with depth and symlink checking"""

        # ✅ Depth guard
        if depth > max_depth:
            self.warnings.append(f"Max depth {max_depth} reached at {path}")
            return CodebaseMap()

        # ✅ Symlink detection (prevent loops)
        try:
            stat = path.stat()
            inode = stat.st_ino
            if inode in visited_inodes:
                self.warnings.append(f"Symlink loop detected at {path}, skipping")
                return CodebaseMap()
            visited_inodes.add(inode)
        except OSError:
            self.warnings.append(f"Cannot stat {path}, skipping")
            return CodebaseMap()

        # ✅ Process files with file count guard
        total_files = 0
        for item in path.iterdir():  # iterdir() instead of rglob() for control
            if total_files > self.MAX_FILES_PER_SCAN:
                self.warnings.append(
                    f"File limit {self.MAX_FILES_PER_SCAN} exceeded, "
                    f"switching to sampling"
                )
                # Could implement sampling here
                break

            if item.is_dir():
                if item.name in self.SKIP_DIRS:
                    continue
                # Recurse with depth control
                sub_map = self._scan_recursive(
                    item, depth + 1, max_depth, visited_inodes
                )
                # Merge results
            elif item.is_file():
                total_files += 1
                # Process file

        return codebase_map

    def _timeout_handler(self, signum, frame):
        raise TimeoutError("Codebase analysis exceeded timeout")

    def _create_partial_result(self, total_files: int, warning: str) -> CodebaseMap:
        """Create best-effort result with degradation warning"""
        return CodebaseMap(
            total_files=total_files,
            warning=warning,
            is_partial=True  # ✅ Flag partial results
        )
```

### Risk Mitigation Summary

| Risk | Mitigation |
|------|-----------|
| **Infinite loops** | Inode-based symlink cycle detection |
| **Stack overflow** | Depth limit (MAX_DEPTH=20) |
| **Timeout** | Signal-based timeout with partial results |
| **Huge monorepos** | File count limit with sampling fallback |
| **Silent failure** | Warning flags and partial result indication |

**Expected Impact**:
- Worst case now returns partial results instead of crash
- Users get diagnostic info (warnings) about what happened
- Regret gap reduced from "completely broken" to "incomplete but usable"

---

## Failure Mode #2: LLM Response Degradation (Silent Quality Collapse)

### Scenario Description

**Black Swan Event**: DeepSeek Reasoner returns:
- Malformed JSON (missing fields, extra fields)
- Network timeout (partial response)
- Unexpected format due to prompt ambiguity
- Rate limit error message instead of JSON

**Current Implementation Vulnerability**:

```python
# orchestrator/codebase_understanding.py - _call_llm_async()
async def _call_llm_async(self, prompt: str) -> dict:
    try:
        response = await self.orchestrator.run_task(...)

        # ❌ Brittle regex extraction
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json.loads(json_match.group())

    except Exception:
        # ❌ Too basic fallback - user doesn't know quality degraded
        return self._default_analysis()

def _default_analysis(self) -> dict:
    return {
        "purpose": "Unable to determine",
        "patterns": [],
        "anti_patterns": [],
        "test_coverage": "unknown"
    }
```

### Failure Scenarios

| Scenario | Mechanism | Outcome |
|----------|-----------|---------|
| **Malformed JSON** | Missing `test_coverage` field | Regex matches, JSON parse fails, fallback to default |
| **Timeout** | Network latency, partial response | Regex extracts garbage, invalid JSON |
| **Rate Limit** | DeepSeek returns 429 error message | Regex extracts HTML/error, parse fails silently |
| **Format Change** | Prompt misunderstood, returned different structure | Regex matches something, parses, but wrong fields |

### Minimax Regret Analysis

```
Best Case Outcome:
  - DeepSeek returns perfect JSON
  - All fields present and valid
  - CodebaseProfile populated correctly
  - Suggestions are accurate
  - Regret: 0 (feature works perfectly)

Worst Case Outcome:
  - DeepSeek returns malformed response
  - Regex extraction fails, falls back to heuristic
  - CodebaseProfile has placeholder values ("Unable to determine")
  - Suggestions are generic/incorrect
  - ❌ User has NO WAY TO KNOW quality degraded
  - ❌ User trust eroded (bad suggestions treated as feature output)
  - Regret: MAXIMUM (feature generates disinformation)

Regret Gap: Catastrophic
- Best: High-quality analysis
- Worst: Garbage analysis that user trusts
- Silent degradation (no confidence flag)
- Reputation damage (feature blamed for bad suggestions)
```

### Specific Improvement Proposal

**Add Multi-Strategy Parsing with Confidence Scoring**:

```python
from enum import Enum
from pydantic import BaseModel, ValidationError, Field

class AnalysisConfidence(str, Enum):
    HIGH = "high"      # Direct LLM JSON, all fields valid
    MEDIUM = "medium"  # Regex extraction or 1+ fields from heuristic
    LOW = "low"        # All heuristic fallback

class CodebaseProfileV2(BaseModel):
    """Enhanced with confidence tracking"""
    purpose: str
    primary_patterns: List[str]
    anti_patterns: List[str]
    test_coverage: str  # high, moderate, low, unknown
    documentation: str
    primary_language: str
    project_type: str

    # ✅ New fields
    confidence_level: AnalysisConfidence = AnalysisConfidence.HIGH
    confidence_sources: Dict[str, str] = Field(default_factory=dict)
    # e.g., {"test_coverage": "llm", "project_type": "heuristic"}

    class Config:
        json_schema_extra = {
            "example": {
                "confidence_level": "medium",
                "confidence_sources": {
                    "purpose": "llm",
                    "test_coverage": "heuristic"
                }
            }
        }

class CodebaseUnderstandingV2(CodebaseUnderstanding):
    """Enhanced with robust parsing and confidence tracking"""

    async def _call_llm_async(self, prompt: str) -> CodebaseProfileV2:
        """Multi-strategy parsing with confidence scoring"""

        # Strategy 1: Direct JSON parsing (most reliable)
        try:
            response = await self.orchestrator.run_task(
                description="Analyze codebase",
                task_type=TaskType.REASONING,
                input_data={"prompt": prompt}
            )

            profile = self._parse_json_strict(response)
            if profile:
                profile.confidence_level = AnalysisConfidence.HIGH
                return profile

        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.warning(f"Direct JSON parsing failed: {e}")

        # Strategy 2: Regex extraction with validation
        try:
            profile = self._parse_json_regex(response)
            if profile:
                profile.confidence_level = AnalysisConfidence.MEDIUM
                self.logger.warning(f"Using regex extraction (confidence: medium)")
                return profile

        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.warning(f"Regex parsing failed: {e}")

        # Strategy 3: Heuristic fallback with confidence tracking
        profile = self._parse_heuristic(response)
        profile.confidence_level = AnalysisConfidence.LOW
        self.logger.warning(f"Using heuristic fallback (confidence: low)")
        return profile

    def _parse_json_strict(self, response: str) -> Optional[CodebaseProfileV2]:
        """Try direct JSON parsing with schema validation"""
        try:
            data = json.loads(response)
            # ✅ Pydantic validates schema
            return CodebaseProfileV2(**data)
        except Exception:
            return None

    def _parse_json_regex(self, response: str) -> Optional[CodebaseProfileV2]:
        """Extract JSON with regex, validate with schema"""
        try:
            match = re.search(r'\{[\s\S]*\}', response)
            if not match:
                return None

            json_str = match.group()
            data = json.loads(json_str)

            # ✅ Fill missing fields with heuristic defaults
            filled_data = self._fill_missing_fields(data, response)

            # ✅ Pydantic validates
            profile = CodebaseProfileV2(**filled_data)

            # Track which fields came from heuristic
            profile.confidence_sources = self._track_sources(data)
            return profile

        except Exception:
            return None

    def _parse_heuristic(self, response: str) -> CodebaseProfileV2:
        """Full heuristic fallback - analyzes response text"""
        return CodebaseProfileV2(
            purpose=self._extract_heuristic_purpose(response),
            primary_patterns=self._extract_heuristic_patterns(response),
            anti_patterns=self._extract_heuristic_antipatterns(response),
            test_coverage=self._extract_heuristic_test_coverage(response),
            documentation=self._extract_heuristic_documentation(response),
            confidence_level=AnalysisConfidence.LOW,
            confidence_sources={
                "purpose": "heuristic",
                "primary_patterns": "heuristic",
                "anti_patterns": "heuristic",
                "test_coverage": "heuristic",
                "documentation": "heuristic"
            }
        )

    def _track_sources(self, data: dict) -> Dict[str, str]:
        """Track which fields were from JSON vs heuristic"""
        sources = {}
        for key in data.keys():
            sources[key] = "llm"
        for key in ["purpose", "test_coverage", "documentation", "primary_patterns", "anti_patterns"]:
            if key not in data:
                sources[key] = "heuristic"
        return sources
```

### Usage Impact

```python
# User code now has visibility into analysis quality
understanding = CodebaseUnderstandingV2()
profile = await understanding.analyze(codebase_path="/path/to/project")

# ✅ Can check confidence before trusting suggestions
if profile.confidence_level == AnalysisConfidence.LOW:
    logger.warning(f"Analysis has LOW confidence: {profile.confidence_sources}")
    # Could require manual review or skip suggestion generation

# ✅ UI can show confidence indicators
print(f"Analysis Confidence: {profile.confidence_level}")
for field, source in profile.confidence_sources.items():
    status = "✓" if source == "llm" else "⚠️"
    print(f"  {status} {field}: {source}")
```

### Risk Mitigation Summary

| Risk | Mitigation |
|------|-----------|
| **Malformed JSON** | Multi-strategy parsing (direct → regex → heuristic) |
| **Silent degradation** | Confidence level tracking and source tracking |
| **User trust eroded** | Explicit confidence indicators in output |
| **Undetected errors** | Structured logging of parsing strategy used |
| **Field validation** | Pydantic schema validation at each step |

**Expected Impact**:
- Worst case now returns low-confidence analysis instead of undetected garbage
- Users can see which fields were LLM-derived vs heuristic
- Regret gap reduced from "trusted garbage" to "transparently degraded with warnings"

---

## Failure Mode #3: Context-Blind Suggestions (Feature Irrelevance)

### Scenario Description

**Black Swan Event**: ImprovementSuggester generates suggestions that are misaligned with:
- Business priorities (recommending quality improvements for a speed-to-market startup)
- Industry vertical (suggesting tests for physics simulation where testing is computationally expensive)
- Technical context (recommending type hints for Python 2 legacy codebase)
- Team capacity (suggesting 40-hour improvements when team has 4 hours/week available)

**Current Implementation Vulnerability**:

```python
# orchestrator/improvement_suggester.py
class ImprovementSuggester:
    def suggest(self, profile: CodebaseProfile) -> List[Improvement]:
        improvements = []

        # ❌ Hard-coded rules with no business context
        if profile.test_coverage in ["low", "moderate"]:
            improvements.append(Improvement(
                title="Add comprehensive test suite",
                description="Increase test coverage to >80%",
                effort_hours=6,  # ❌ No context: might not be priority
                priority="HIGH"  # ❌ Always HIGH for testing
            ))

        # ❌ No industry-specific rules
        # ❌ No team capacity consideration
        # ❌ No business impact estimation
        # ❌ No learning from past suggestions

        return improvements
```

### Failure Scenarios

| Scenario | Mechanism | Outcome |
|----------|-----------|---------|
| **Speed-to-market startup** | Suggests comprehensive testing (6h) | Misaligned with MVP priority; feature ignored |
| **Scientific computing** | Suggests tests for numerical code | Computationally expensive; not practical |
| **Legacy codebase** | Suggests type hints for Python 2 | Impossible; technical debt, not improvement |
| **Tiny team** | Suggests 4 different 3-4h improvements | 12+ hours total; team capacity is 4h/week; overwhelming |

### Minimax Regret Analysis

```
Best Case Outcome:
  - Suggestions match team priorities exactly
  - Team adopts suggestions
  - Each suggestion has high ROI (high impact, low effort)
  - Feature becomes competitive advantage
  - Regret: 0 (feature perfectly aligned with needs)

Worst Case Outcome:
  - Suggestions misaligned with business priorities
  - Team ignores feature suggestions as "generic consultant noise"
  - Feature provides no value
  - Reputation damage: "Feature generates boilerplate recommendations"
  - ROI negative: effort to review ignored suggestions > value provided
  - Regret: MAXIMUM (feature becomes liability, not asset)

Regret Gap: Severe
- Best: High-ROI differentiated recommendations
- Worst: Generic irrelevant noise
- Users can't tune feature to their context
- Silent irrelevance (no indication that suggestions are misaligned)
```

### Specific Improvement Proposal

**Add Business Context Scoring and Learning**:

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

class BusinessPriority(str, Enum):
    SPEED_TO_MARKET = "speed"      # MVP, fast iteration
    QUALITY_FIRST = "quality"      # Critical systems, regulatory
    BALANCED = "balanced"           # Typical production
    SCALE_FIRST = "scalability"    # High-growth or infrastructure

class TeamSize(str, Enum):
    SOLO = "solo"              # 1 person
    SMALL = "small"            # 2-5 people
    MEDIUM = "medium"          # 6-20 people
    LARGE = "large"            # 20+ people

@dataclass
class ImprovementContext:
    """Business context for suggestion generation"""
    business_priority: BusinessPriority = BusinessPriority.BALANCED
    team_size: TeamSize = TeamSize.MEDIUM
    maturity_stage: str = "production"  # mvp, growth, mature, legacy
    industry_vertical: Optional[str] = None  # web, finance, scientific, game, etc.
    team_hours_per_week: int = 40  # Available capacity
    max_effort_per_suggestion: int = 8  # Won't recommend >Xh items to small teams

@dataclass
class ImprovementV2(Improvement):
    """Enhanced with business impact"""

    # Original fields
    title: str
    description: str
    impact: str
    effort_hours: int
    priority: str
    category: str

    # ✅ New fields
    business_impact: str  # Which business metric improves? "reduces bugs", "faster deployment", etc.
    roi_estimate: float  # Rough ROI: (benefit_score) / effort_hours
    compatibility_notes: Optional[str] = None  # "Requires Python 3.8+"
    context_relevance: float = 1.0  # How relevant to team context? 0.0-1.0

class ImprovementSuggesterV2(ImprovementSuggester):
    """Enhanced with business context awareness"""

    def suggest(
        self,
        profile: CodebaseProfile,
        context: ImprovementContext = ImprovementContext()
    ) -> List[ImprovementV2]:
        """Generate context-aware suggestions"""

        improvements = []

        # Apply context-specific rules
        if context.business_priority == BusinessPriority.SPEED_TO_MARKET:
            improvements.extend(self._suggest_for_speed(profile, context))
        elif context.business_priority == BusinessPriority.QUALITY_FIRST:
            improvements.extend(self._suggest_for_quality(profile, context))
        elif context.business_priority == BusinessPriority.SCALE_FIRST:
            improvements.extend(self._suggest_for_scalability(profile, context))
        else:
            improvements.extend(self._suggest_balanced(profile, context))

        # Apply industry-specific overrides
        if context.industry_vertical == "scientific":
            improvements = self._filter_for_scientific(improvements, profile)
        elif context.industry_vertical == "finance":
            improvements = self._filter_for_finance(improvements, profile)

        # Filter by team capacity
        improvements = self._filter_by_capacity(improvements, context)

        # Sort by ROI (impact per hour)
        improvements.sort(
            key=lambda x: (x.roi_estimate, -x.effort_hours),
            reverse=True
        )

        return improvements

    def _suggest_for_speed(
        self,
        profile: CodebaseProfile,
        context: ImprovementContext
    ) -> List[ImprovementV2]:
        """Suggestions aligned with speed-to-market"""
        improvements = []

        # ✅ Prioritize speed-enabling changes (CI/CD, deployment)
        # ❌ Don't suggest comprehensive test coverage (costs time)
        # ❌ Don't suggest refactoring (slows iteration)

        if "no logging" in profile.anti_patterns:
            improvements.append(ImprovementV2(
                title="Add structured logging",
                description="Implement logging to catch production bugs faster",
                impact="Faster debugging in production",
                effort_hours=2,
                priority="HIGH",
                category="features",
                business_impact="reduces_mttr",  # Mean time to recovery
                roi_estimate=10.0,  # High ROI for speed teams
                context_relevance=1.0
            ))

        # Suggest CI/CD improvements for speed teams
        if not profile.has_tests and context.team_size == TeamSize.SMALL:
            improvements.append(ImprovementV2(
                title="Add GitHub Actions or CI/CD pipeline",
                description="Automate deploy process for faster iteration",
                impact="1-click deploys, less manual work",
                effort_hours=3,
                priority="HIGH",
                category="features",
                business_impact="accelerates_deployment",
                roi_estimate=8.0,
                context_relevance=1.0
            ))

        # De-prioritize testing for MVP stage
        if context.maturity_stage == "mvp" and profile.test_coverage == "low":
            improvements.append(ImprovementV2(
                title="Add basic smoke tests",
                description="Minimum viable test coverage (API endpoints only)",
                impact="Catch critical bugs without slowing iteration",
                effort_hours=2,  # ✅ Reduced from 6
                priority="MEDIUM",  # ✅ Demoted from HIGH
                category="testing",
                business_impact="reduces_critical_bugs",
                roi_estimate=5.0,
                context_relevance=0.8  # ✅ Note reduced relevance
            ))

        return improvements

    def _suggest_for_quality(
        self,
        profile: CodebaseProfile,
        context: ImprovementContext
    ) -> List[ImprovementV2]:
        """Suggestions aligned with quality-first"""
        improvements = []

        # ✅ Prioritize testing, type hints, error handling
        if profile.test_coverage in ["low", "moderate"]:
            improvements.append(ImprovementV2(
                title="Add comprehensive test suite",
                description="Increase test coverage to >80% with integration tests",
                impact="High confidence in changes, prevent regressions",
                effort_hours=6,
                priority="HIGH",
                category="testing",
                business_impact="reduces_bugs_in_production",
                roi_estimate=12.0,  # ✅ High ROI for quality teams
                context_relevance=1.0
            ))

        return improvements

    def _suggest_for_scalability(
        self,
        profile: CodebaseProfile,
        context: ImprovementContext
    ) -> List[ImprovementV2]:
        """Suggestions aligned with scalability"""
        improvements = []

        # ✅ Prioritize caching, database optimization, async patterns
        if profile.primary_language == "python":
            improvements.append(ImprovementV2(
                title="Add async/await patterns",
                description="Convert blocking operations to async for better concurrency",
                impact="Handle 10x more requests with same hardware",
                effort_hours=8,
                priority="HIGH",
                category="refactoring",
                business_impact="increases_throughput",
                roi_estimate=15.0,  # ✅ Very high ROI for scale teams
                context_relevance=1.0
            ))

        return improvements

    def _filter_for_scientific(
        self,
        improvements: List[ImprovementV2],
        profile: CodebaseProfile
    ) -> List[ImprovementV2]:
        """Remove impractical suggestions for scientific computing"""

        filtered = []
        for imp in improvements:
            # ❌ Remove comprehensive testing (computationally expensive)
            if "comprehensive test suite" in imp.title.lower():
                imp.context_relevance = 0.2  # ✅ Mark as low relevance
                imp.compatibility_notes = "For scientific code, focus on validation tests not full coverage"
                filtered.append(imp)
            # ✅ Keep numerical validation improvements
            elif imp.business_impact in ["reduces_numerical_errors", "validates_results"]:
                filtered.append(imp)
            else:
                filtered.append(imp)

        return filtered

    def _filter_by_capacity(
        self,
        improvements: List[ImprovementV2],
        context: ImprovementContext
    ) -> List[ImprovementV2]:
        """Filter suggestions by team capacity"""

        # Calculate total effort if all suggestions adopted
        total_effort = sum(imp.effort_hours for imp in improvements)

        if total_effort > context.team_hours_per_week:
            # ✅ Filter to fit within capacity
            filtered = []
            cumulative = 0

            for imp in sorted(improvements, key=lambda x: x.roi_estimate, reverse=True):
                if cumulative + imp.effort_hours <= context.team_hours_per_week:
                    filtered.append(imp)
                    cumulative += imp.effort_hours
                elif imp.effort_hours <= context.max_effort_per_suggestion:
                    # Can still fit if under max per item
                    if cumulative + imp.effort_hours <= context.team_hours_per_week * 2:
                        filtered.append(imp)
                        cumulative += imp.effort_hours

            # Warn if suggestions exceed capacity
            if cumulative > context.team_hours_per_week:
                logger.warning(
                    f"Suggestions ({cumulative}h) exceed weekly capacity "
                    f"({context.team_hours_per_week}h). "
                    f"Prioritize by ROI."
                )

            return filtered

        return improvements
```

### Usage Example

```python
# User specifies their context
context = ImprovementContext(
    business_priority=BusinessPriority.SPEED_TO_MARKET,
    team_size=TeamSize.SMALL,
    maturity_stage="mvp",
    industry_vertical="web",
    team_hours_per_week=20
)

# Get context-aware suggestions
suggester = ImprovementSuggesterV2()
improvements = suggester.suggest(profile, context)

# Display with context relevance
for imp in improvements:
    relevance_indicator = "✓" if imp.context_relevance >= 0.9 else "~" if imp.context_relevance >= 0.7 else "✗"
    print(f"{relevance_indicator} [{imp.priority}] {imp.title}")
    print(f"   Effort: {imp.effort_hours}h | ROI: {imp.roi_estimate:.1f}x")
    print(f"   Relevance: {int(imp.context_relevance * 100)}%")
    if imp.compatibility_notes:
        print(f"   Note: {imp.compatibility_notes}")
    print()
```

### Risk Mitigation Summary

| Risk | Mitigation |
|------|-----------|
| **Misaligned priorities** | Business priority routing to different suggestion sets |
| **Industry-specific gaps** | Industry vertical filtering and compatibility notes |
| **Team overwhelm** | Capacity-aware filtering (fit within weekly hours) |
| **Silent irrelevance** | Explicit relevance scoring and priority indicators |
| **No learning** | Foundation for tracking adoption and improving rules |

**Expected Impact**:
- Worst case now returns low-relevance suggestions with clear indicators instead of universally trusted noise
- Users can tune feature to their business context
- ROI-sorted suggestions optimize for impact per hour
- Regret gap reduced from "universally wrong" to "context-aware with transparency"

---

## Summary: Black Swan Resilience Roadmap

| Failure Mode | Current Risk | Mitigation | Expected Regret Gap |
|------|------|------|------|
| **Pathological Directories** | Crash/timeout, no fallback | Depth limiting, symlink detection, partial results | Large → Medium |
| **LLM Response Degradation** | Silent garbage analysis | Multi-strategy parsing, confidence scoring | Catastrophic → Medium |
| **Context-Blind Suggestions** | Feature irrelevance | Business context routing, capacity filtering, ROI scoring | Severe → Small |

### Implementation Priority

1. **High Priority** (addresses largest regret gap):
   - LLM Response Degradation → Multi-strategy parsing with confidence
   - Reason: Silent quality collapse is most dangerous (trusted garbage)

2. **Medium Priority** (addresses operational resilience):
   - Pathological Directories → Defensive scanning with partial results
   - Reason: DOS/timeout prevents feature usage entirely

3. **Medium-High Priority** (addresses feature value):
   - Context-Blind Suggestions → Business context routing
   - Reason: Feature irrelevance undermines entire feature adoption

### Testing Strategy for Improvements

```python
# Adversarial test cases to add:

def test_symlink_loop_detection():
    """Verify symlink loops don't cause infinite recursion"""
    # Create A → B → A symlink loop
    # Assert scan completes with partial results

def test_scanning_timeout():
    """Verify scan times out gracefully on huge monorepo"""
    # Create mock filesystem with 1M+ files
    # Assert timeout → partial results

def test_llm_response_parsing_strategies():
    """Verify multi-strategy parsing handles malformed responses"""
    # Test malformed JSON, partial response, format change
    # Assert confidence level reflects parsing strategy

def test_confidence_scoring():
    """Verify confidence levels accurately reflect quality"""
    # Compare high-confidence vs low-confidence profiles
    # Assert low-confidence has more heuristic fields

def test_business_context_routing():
    """Verify suggestions align with business priorities"""
    # Test speed_to_market vs quality_first contexts
    # Assert different suggestion sets for same codebase

def test_capacity_filtering():
    """Verify suggestions don't exceed team capacity"""
    # Test small team with 4h/week capacity
    # Assert suggestions total ≤ 4 hours
```

---

## Conclusion

The Codebase Enhancer POC provides excellent functionality for typical codebases but contains three **black swan vulnerabilities** with large minimax regret gaps. Each failure mode has a clear, specific mitigation strategy that trades complexity for resilience.

**Recommendation**: Implement in priority order (LLM degradation → DOS resilience → context awareness) to progressively close the regret gaps and transition from POC to production-grade feature.


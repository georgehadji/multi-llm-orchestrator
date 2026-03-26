# Αρχιτεκτονική Ανάλυση: Core vs Plugins

> **Αρχή Σχεδιασμού**: *Minimal Core, Maximal Extensibility*
> 
> Το core παρέχει μόνο το απαραίτητο 20% για το 80% των χρηστών. Τα plugins καλύπτουν τα edge cases και domain-specific ανάγκες.

---

## Επισκόπηση των 4 Συστημάτων

| Σύστημα | Τρέχουσα Κατάσταση | Πρόταση | Λόγος |
|---------|-------------------|---------|-------|
| **Plugin System** | Core | ✅ **Core** | Είναι το θεμέλιο για όλα τα άλλα |
| **Production Feedback Loop** | Core | 🟡 **Hybrid** | Core = storage + API, Plugins = processors |
| **Model Leaderboard** | Core | 🟡 **Core + Plugin Hooks** | Core = benchmarking engine, Plugins = custom benchmarks |
| **Outcome-Weighted Router** | Core | ✅ **Core** | Είναι το ανταγωνιστικό πλεονέκτημα (Nash stability) |

---

## 1. Plugin System → Core (100%)

**Γιατί Core:**
- Είναι η υποδομή που επιτρέπει όλα τα άλλα extensions
- Χωρίς αυτό, δεν μπορείς να προσθέσεις τίποτα άλλο
- Μικρό footprint (~600 lines), μεγάλη αξία

**Τι είναι Core:**
```python
# Core
- Plugin base classes (Plugin, ValidatorPlugin, etc.)
- PluginRegistry (discovery, lifecycle management)
- PluginMetadata / ValidationResult dataclasses
- get_plugin_registry() singleton

# Όχι Core (μένει ως παράδειγμα ή plugin)
- PythonTypeCheckerValidator → Μετακίνηση σε επίσημο plugin
- TeamsIntegration → Μετακίνηση σε επίσημο plugin
```

---

## 2. Production Feedback Loop → Hybrid

### Core (60%)

Αυτά ΠΡΕΠΕΙ να είναι core γιατί είναι το "νευρικό σύστημα":

```python
# Core Components
1. ProductionOutcome dataclass      # Το ίδιο το δεδομένο
2. FeedbackLoop.record_outcome()    # Η πύλη εισόδου
3. ModelPerformanceRecord           # Storage & aggregation
4. Storage layer (JSON/SQLite)      # Persistence
5. get_model_score()                # Query API
```

### Plugins (40%)

Αυτά είναι επεκτάσεις που όχι όλοι χρειάζονται:

```python
# Plugin Candidates

## A. FeedbackProcessor Plugins
class SentryFeedbackPlugin(FeedbackPlugin):
    """Επεξεργάζεται errors από Sentry"""
    pass

class DatadogFeedbackPlugin(FeedbackPlugin):
    """Τραβάει metrics από Datadog"""
    pass

class CustomWebhookFeedbackPlugin(FeedbackPlugin):
    """Στέλνει σε custom webhook"""
    pass

## B. CodebaseFingerprint Plugins  
class PythonFingerprintPlugin(FeedbackPlugin):
    """Αναλύει Python codebases για patterns"""
    pass

class ReactFingerprintPlugin(FeedbackPlugin):
    """Αναλύει React codebases για patterns"""
    pass

## C. Analytics Plugins
class PostHogAnalyticsPlugin(FeedbackPlugin):
    """Στέλνει events σε PostHog"""
    pass
```

**Architecture:**
```
Core Feedback Loop
    ↓ (calls)
Plugin Registry
    ↓ (dispatches to)
[Plugin A] [Plugin B] [Plugin C]
```

---

## 3. Model Leaderboard → Core + Plugin Hooks

### Core (70%)

Το benchmarking engine πρέπει να είναι core:

```python
# Core
1. BenchmarkTask dataclass
2. BenchmarkResult dataclass
3. ModelBenchmarkSummary
4. ModelLeaderboard.run_benchmarks()
5. get_leaderboard() / LeaderboardEntry
6. Routing weight updates
```

### Plugin Extensions (30%)

```python
# Plugin Candidates

## A. Custom Benchmark Tasks
class SecurityBenchmarkPlugin(Plugin):
    """Προσθέτει security-focused benchmark tasks"""
    def get_tasks(self) -> List[BenchmarkTask]:
        return [
            BenchmarkTask(
                name="SQL Injection Test",
                prompt="Generate code vulnerable to SQL injection...",
                expected_patterns=["parametrized query", "sanitization"],
            ),
            ...
        ]

## B. Domain-Specific Benchmarks
class MLCodeBenchmarkPlugin(Plugin):
    """Benchmark tasks για ML/AI code"""
    pass

class EmbeddedSystemsBenchmarkPlugin(Plugin):
    """Benchmark tasks για embedded C"""
    pass

## C. External Data Sources
class LMSYSDataPlugin(Plugin):
    """Τραβάει rankings από LMSYS Chatbot Arena"""
    pass
```

---

## 4. Outcome-Weighted Router → Core (90%)

**Γιατί σχεδόν όλο Core:**

Αυτό είναι το **κύριο ανταγωνιστικό πλεονέκτημα** (Nash stability). 
Δεν μπορείς να το έχεις ως optional plugin γιατί:

1. Χρειάζεται deep integration με το engine
2. Πρέπει να είναι πάντα ενεργό για να μαζεύει data
3. Είναι το moat που προστατεύει από ανταγωνιστές

```python
# Core (90%)
1. OutcomeWeightedRouter class
2. RoutingContext / ModelScore
3. RoutingStrategy enum
4. select_model() main logic
5. Production score weighting
6. Codebase fingerprint matching
7. Exploration vs Exploitation logic

# Plugin Interface (10%)
1. RouterPlugin base class (ήδη υπάρχει)
2. RoutingSuggestion dataclass
3. _blend_plugin_suggestions() method
```

Οι χρήστες μπορούν να γράψουν **custom router plugins** που:
- Προτείνουν models για ειδικές περιπτώσεις
- Override τη συμπεριφορά για specific domains
- Προσθέτουν constraints (π.χ. "μόνο EU-based models")

---

## Συνοπτικός Πίνακας

```
┌─────────────────────────────┬──────────────────┬──────────────────┐
│ Συστατικό                   │ Core             │ Plugin           │
├─────────────────────────────┼──────────────────┼──────────────────┤
│ Plugin Registry             │ ✅ 100%          │                  │
│ Plugin Base Classes         │ ✅ 100%          │                  │
│                             │                  │                  │
│ Feedback Storage            │ ✅ 60%           │                  │
│ Feedback API (record/get)   │ ✅ Core          │                  │
│ Feedback Processors         │                  │ 🟡 40%           │
│ Codebase Analyzers          │                  │ 🟡 40%           │
│ External Integrations       │                  │ 🟡 40%           │
│                             │                  │                  │
│ Benchmark Engine            │ ✅ 70%           │                  │
│ Standard Tasks              │ ✅ Core          │                  │
│ Custom Benchmark Tasks      │                  │ 🟡 30%           │
│ External Data Sources       │                  │ 🟡 30%           │
│                             │                  │                  │
│ Router Core Logic           │ ✅ 90%           │                  │
│ Routing Strategies          │ ✅ Core          │                  │
│ Custom Router Plugins       │                  │ 🟡 10%           │
│                             │                  │                  │
│ Built-in Validators         │                  │ ⚪ Official      │
│ Built-in Integrations       │                  │ ⚪ Official      │
└─────────────────────────────┴──────────────────┴──────────────────┘

⚪ Official = Επίσημα plugins (διατηρούνται από core team, αλλά optional)
```

---

## Επίσημα Plugins (Official Plugins)

Προτείνω να μετακινηθούν σε ξεχωριστό repo/package:

```
orchestrator-plugins/
├── validators/
│   ├── python-mypy/
│   ├── python-bandit/
│   ├── rust-cargo/
│   ├── go-vet/
│   └── javascript-eslint/
├── integrations/
│   ├── slack/
│   ├── teams/
│   ├── discord/
│   ├── sentry/
│   └── datadog/
├── benchmarks/
│   ├── security-suite/
│   ├── ml-code-suite/
│   └── mobile-suite/
└── routers/
    ├── cost-optimizer/
    └── domain-specific/
```

**Benefits:**
1. **Faster Core Releases** - Core updates χωρίς να σπάνε plugins
2. **Community Contributions** - Ο κόσμος γράφει plugins χωρίς PR στο core
3. **Optional Dependencies** - Κατεβάζεις μόνο ό,τι χρειάζεσαι
4. **Version Independence** - Plugin v2.0 μπορεί να δουλεύει με Core v5.x

---

## Πρακτική Εφαρμογή

### Τρέχουσα Κατάσταση
```python
# Τώρα (όλα στο core)
from orchestrator.plugins import TeamsIntegration
from orchestrator.feedback_loop import FeedbackLoop
from orchestrator.leaderboard import ModelLeaderboard
```

### Προτεινόμενη Κατάσταση
```python
# Μετά το refactoring

# Core (πάντα available)
from orchestrator.plugins import PluginRegistry, ValidatorPlugin
from orchestrator.feedback_loop import FeedbackLoop, ProductionOutcome
from orchestrator.leaderboard import ModelLeaderboard
from orchestrator.outcome_router import OutcomeWeightedRouter

# Official Plugins (optional, pip install)
from orchestrator_plugins.teams import TeamsIntegration
from orchestrator_plugins.sentry import SentryFeedbackPlugin
from orchestrator_plugins.security import SecurityBenchmarkPlugin

# Community Plugins
from mycompany_plugins import InternalRouterPlugin
```

---

## Migration Path

### Phase 1: Plugin Interface Stabilization (2 εβδομάδες)
1. Εξαγωγή των plugin interfaces σε stable API
2. Documentation για plugin developers
3. Test suite για plugin compatibility

### Phase 2: Core Extraction (2 εβδομάδες)
1. Μετακίνηση built-in validators σε `orchestrator-plugins-validators`
2. Μετακίνηση integrations σε `orchestrator-plugins-integrations`
3. Update documentation

### Phase 3: Optional Installation (1 εβδομάδα)
1. `pip install orchestrator` → Core μόνο
2. `pip install orchestrator[all]` → Core + Official Plugins
3. `pip install orchestrator[slack,sentry]` → Επιλεκτικά

---

## Τελική Σύσταση

| Ενέργεια | Προτεραιότητα |
|----------|---------------|
| **Outcome-Weighted Router** → Core | 🔴 Critical |
| **Plugin System** → Core | 🔴 Critical |
| **Feedback Storage/API** → Core | 🟡 High |
| **Benchmark Engine** → Core | 🟡 High |
| **Built-in Validators** → Official Plugins | 🟢 Medium |
| **Integrations** → Official Plugins | 🟢 Medium |
| **Custom Benchmarks** → Plugin Interface | 🟢 Medium |
| **Feedback Processors** → Plugin Interface | 🔵 Low |

**Core Size Target**: ~50% του τωρινού codebase
**Plugin Ecosystem**: ~50% μετακινείται σε επίσημα/community plugins

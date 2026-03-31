"""
Multi-LLM Orchestrator v6.0 — Optimized Paradigm
================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

MAJOR CHANGES v6.0:
- Dashboard consolidation: 7 dashboards → 1 core + plugins
- Event unification: 4 event systems → 1 unified bus
- Plugin extraction: Core-only + optional plugins

Quick Start:
    from orchestrator import Orchestrator, Budget

    # New unified dashboard
    from orchestrator import run_dashboard
    run_dashboard(view="mission-control")

    # New unified events
    from orchestrator import get_event_bus, ProjectStartedEvent
"""

__version__ = "6.0.0"

# ═══════════════════════════════════════════════════════════════════════════════
# Core Models & Types
# ═══════════════════════════════════════════════════════════════════════════════
from .api_clients import APIResponse, UnifiedClient
from .cache import DiskCache

# ═══════════════════════════════════════════════════════════════════════════════
# Core Engine & Clients
# ═══════════════════════════════════════════════════════════════════════════════
from .engine import Orchestrator
from .models import (
    COST_TABLE,
    FALLBACK_CHAIN,
    ROUTING_TABLE,
    AttemptRecord,
    Budget,
    Model,
    ProjectState,
    ProjectStatus,
    Task,
    TaskResult,
    TaskStatus,
    TaskType,
    build_default_profiles,
)
from .state import StateManager

# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Unified Dashboard Core (v6.0) — Replaces 7 dashboard implementations
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from .dashboard_core import (
        DashboardCore,
        DashboardView,
        MissionControlView,
        ViewContext,
        ViewRegistry,
        get_dashboard_core,
        run_dashboard,
    )

    HAS_UNIFIED_DASHBOARD = True
except ImportError as _e:
    HAS_UNIFIED_DASHBOARD = False
    DashboardCore = None
    get_dashboard_core = None
    DashboardView = None
    ViewContext = None
    ViewRegistry = None
    run_dashboard = None
    MissionControlView = None

# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Unified Events System (v6.0) — Replaces streaming/events/hooks/capability_logger
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from .unified_events import (
        BudgetWarningEvent,
        CapabilityCompletedEvent,
        CapabilityUsedEvent,
        DomainEvent,
        EventType,
        FallbackTriggeredEvent,
        ModelSelectedEvent,
        ProjectCompletedEvent,
        ProjectStartedEvent,
        TaskCompletedEvent,
        TaskFailedEvent,
        TaskProgressEvent,
        TaskStartedEvent,
        UnifiedEventBus,
        get_current_project,
        get_event_bus,
        log_capability_use,
        set_current_project,
    )

    HAS_UNIFIED_EVENTS = True
except ImportError as _e:
    HAS_UNIFIED_EVENTS = False
    UnifiedEventBus = None
    get_event_bus = None
    DomainEvent = None
    EventType = None
    ProjectStartedEvent = None
    ProjectCompletedEvent = None
    TaskStartedEvent = None
    TaskCompletedEvent = None
    TaskFailedEvent = None
    TaskProgressEvent = None
    CapabilityUsedEvent = None
    CapabilityCompletedEvent = None
    BudgetWarningEvent = None
    ModelSelectedEvent = None
    FallbackTriggeredEvent = None
    log_capability_use = None
    set_current_project = None
    get_current_project = None

# ═══════════════════════════════════════════════════════════════════════════════
# Policy & Planning
# ═══════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════==
# Codebase Enhancer
# ═════════════════════════════════════════════════════════════════════════════==
from .codebase_analyzer import CodebaseAnalyzer, CodebaseMap
from .openrouter_sync import OpenRouterSync
from .codebase_profile import CodebaseProfile
from .codebase_understanding import CodebaseUnderstanding
from .hybrid_search_pipeline import HybridSearchPipeline
from .improvement_suggester import Improvement, ImprovementSuggester
from .planner import ConstraintPlanner
from .policy import (
    EnforcementMode,
    JobSpec,
    ModelProfile,
    Policy,
    PolicyHierarchy,
    PolicySet,
    RateLimit,
    VALID_QUALITY_MODES,
)
from .policy_engine import PolicyEngine
from .query_expander import QueryExpander
from .rate_limiter import RateLimiter, RateLimitExceeded
from .resume_detector import ResumeDetector
from .session_lifecycle import SessionLifecycleManager

# ═══════════════════════════════════════════════════════════════════════════════
# Validation & Quality Assurance
# ═══════════════════════════════════════════════════════════════════════════════
from .validators import VALIDATORS, ValidationResult, async_run_validators, run_validators

try:
    from .model_routing import (
        PHASE_TO_TIER,
        TIER_ROUTING,
        ModelTier,
        get_tier_for_phase,
        select_model,
    )

    HAS_MODEL_ROUTING = True
except ImportError:
    HAS_MODEL_ROUTING = False

try:
    from .autonomy import (
        AUTONOMY_PRESETS,
        AutonomyConfig,
        AutonomyLevel,
        get_autonomy_config,
        requires_approval,
    )

    HAS_AUTONOMY = True
except ImportError:
    HAS_AUTONOMY = False

try:
    from .verification import REPLVerifier, VerificationLevel, VerificationResult, self_healing_loop

    HAS_VERIFICATION = True
except ImportError:
    HAS_VERIFICATION = False

# ═══════════════════════════════════════════════════════════════════════════════
# Observability & Telemetry
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# Advanced Features
# ═══════════════════════════════════════════════════════════════════════════════
from .adaptive_router import AdaptiveRouter, ModelState
from .aggregator import ProfileAggregator, RunRecord
from .audit import AuditLog, AuditRecord
from .cost import BudgetHierarchy, CostForecaster, CostPredictor, ForecastReport, RiskLevel
from .hooks import DashboardHookRegistry, HookRegistry
from .hooks import EventType as HookEventType
from .metrics import ConsoleExporter, JSONExporter, MetricsExporter, PrometheusExporter

# ═══════════════════════════════════════════════════════════════════════════════
# Optimization & Cost Management
# ═══════════════════════════════════════════════════════════════════════════════
from .optimization import GreedyBackend, OptimizationBackend, ParetoBackend, WeightedSumBackend
from .remediation import RemediationEngine, RemediationPlan, RemediationStrategy
from .semantic_cache import DuplicationDetector, SemanticCache
from .telemetry import TelemetryCollector
from .tracing import TracingConfig, configure_tracing, get_tracer

# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Multi-Level Cache Optimizer (v6.1) — Advanced caching with L1/L2/L3
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from .cache_optimizer import (
        CacheConfig,
        CacheOptimizer,
        CacheWarmer,
        L1MemoryCache,
        L2DiskCache,
        L3SemanticCache,
        SmartCacheKeyGenerator,
        get_cache_optimizer,
        reset_cache_optimizer,
    )

    HAS_CACHE_OPTIMIZER = True
except ImportError as _e:
    HAS_CACHE_OPTIMIZER = False
    CacheOptimizer = None
    CacheConfig = None
    L1MemoryCache = None
    L2DiskCache = None
    L3SemanticCache = None
    SmartCacheKeyGenerator = None
    CacheWarmer = None
    get_cache_optimizer = None
    reset_cache_optimizer = None

# ═══════════════════════════════════════════════════════════════════════════════
# Output & Progress
# ═══════════════════════════════════════════════════════════════════════════════
from .dry_run import DryRunRenderer, ExecutionPlan, TaskPlan
from .output_writer import write_output_dir
from .progress_writer import ProgressEntry, ProgressWriter

# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Backward Compatibility Layer (v6.0)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from .compat import (
        ProjectCompleted,
        ProjectEventBus,
        ProjectStarted,
        StreamEvent,
        TaskCompleted,
        TaskStarted,
        print_migration_guide,
        run_ant_design_dashboard,
        run_live_dashboard,
        run_mission_control,
    )

    HAS_COMPAT_LAYER = True
except ImportError:
    HAS_COMPAT_LAYER = False
    run_live_dashboard = None
    run_mission_control = None
    run_ant_design_dashboard = None
    ProjectEventBus = None
    StreamEvent = None
    ProjectStarted = None
    TaskStarted = None
    TaskCompleted = None
    ProjectCompleted = None
    print_migration_guide = None

# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY: Old Dashboards (DEPRECATED — will be removed in v7.0)
# ═══════════════════════════════════════════════════════════════════════════════
# These are kept for backward compatibility but will show deprecation warnings

try:
    from .dashboard import run_dashboard as _legacy_run_dashboard
except ImportError:
    _legacy_run_dashboard = None

try:
    from .dashboard_optimized import run_dashboard as _legacy_run_optimized
except ImportError:
    _legacy_run_optimized = None

try:
    from .dashboard_real import run_dashboard_realtime
except ImportError:
    run_dashboard_realtime = None

try:
    from .dashboard_enhanced import (
        ActiveTaskInfo,
        ArchitectureInfo,
        DashboardIntegration,
        EnhancedDashboardServer,
        EnhancedDataProvider,
        ModelStatusInfo,
        ProjectInfo,
    )
    from .dashboard_enhanced import (
        run_enhanced_dashboard as _legacy_run_enhanced,
    )
except ImportError:
    EnhancedDashboardServer = None
    EnhancedDataProvider = None
    DashboardIntegration = None
    ArchitectureInfo = None
    ProjectInfo = None
    ActiveTaskInfo = None
    ModelStatusInfo = None
    _legacy_run_enhanced = None

run_enhanced_dashboard = _legacy_run_enhanced

try:
    from .dashboard_antd import (
        AntDesignDashboardServer,
    )
    from .dashboard_antd import (
        run_ant_design_dashboard as _legacy_run_antd,
    )
except ImportError:
    AntDesignDashboardServer = None
    _legacy_run_antd = None

try:
    from .dashboard_mission_control import (
        MissionControlServer,
    )
    from .dashboard_mission_control import (
        run_mission_control as _legacy_run_mc,
    )
except ImportError:
    MissionControlServer = None
    _legacy_run_mc = None

run_mission_control = _legacy_run_mc

try:
    from .unified_dashboard import run_unified_dashboard
except ImportError:
    run_unified_dashboard = None

try:
    from .project_runner import ProjectRunner
except ImportError:
    ProjectRunner = None

try:
    from .dashboard_live import (
        DashboardLiveIntegration,
        LiveDashboardServer,
    )
    from .dashboard_live import (
        run_live_dashboard as _legacy_run_live,
    )
except ImportError:
    LiveDashboardServer = None
    DashboardLiveIntegration = None
    _legacy_run_live = None

# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY: Streaming & Events (DEPRECATED — use unified_events)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from .streaming import (
        ProjectCompleted as _legacy_project_completed,
    )
    from .streaming import (
        ProjectEventBus as _legacy_event_bus,
    )
    from .streaming import (
        ProjectStarted as _legacy_project_started,
    )
    from .streaming import (
        StreamEvent as _legacy_stream_event,
    )
    from .streaming import (
        TaskCompleted as _legacy_task_completed,
    )
    from .streaming import (
        TaskStarted as _legacy_task_started,
    )
except ImportError:
    _legacy_event_bus = None
    _legacy_project_started = None
    _legacy_task_started = None
    _legacy_task_completed = None
    _legacy_project_completed = None
    _legacy_stream_event = None

# ═══════════════════════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Core
    "Orchestrator",
    "AttemptRecord",
    "Budget",
    "Model",
    "Task",
    "TaskResult",
    "TaskType",
    "TaskStatus",
    "ProjectState",
    "ProjectStatus",
    # NEW: Unified Dashboard (v6.0)
    "DashboardCore",
    "get_dashboard_core",
    "DashboardView",
    "ViewContext",
    "ViewRegistry",
    "run_dashboard",
    "MissionControlView",
    "MissionControlServer",
    "HAS_UNIFIED_DASHBOARD",
    # NEW: Unified Events (v6.0)
    "UnifiedEventBus",
    "get_event_bus",
    "DomainEvent",
    "EventType",
    "ProjectStartedEvent",
    "ProjectCompletedEvent",
    "TaskStartedEvent",
    "TaskCompletedEvent",
    "TaskFailedEvent",
    "TaskProgressEvent",
    "CapabilityUsedEvent",
    "CapabilityCompletedEvent",
    "BudgetWarningEvent",
    "ModelSelectedEvent",
    "FallbackTriggeredEvent",
    "log_capability_use",
    "set_current_project",
    "get_current_project",
    "HAS_UNIFIED_EVENTS",
    # Backward Compatibility
    "run_live_dashboard",
    "run_mission_control",
    "run_ant_design_dashboard",
    "ProjectEventBus",
    "ProjectRunner",
    "StreamEvent",
    "ProjectStarted",
    "TaskStarted",
    "TaskCompleted",
    "ProjectCompleted",
    "print_migration_guide",
    "HAS_COMPAT_LAYER",
    # Legacy (deprecated)
    "AntDesignDashboardServer",
    "EnhancedDashboardServer",
    "EnhancedDataProvider",
    "DashboardIntegration",
    "ArchitectureInfo",
    "ProjectInfo",
    "ActiveTaskInfo",
    "ModelStatusInfo",
    "LiveDashboardServer",
    "DashboardLiveIntegration",
    "run_dashboard_realtime",
    "run_enhanced_dashboard",
    "run_unified_dashboard",
    # Routing & Cost
    "COST_TABLE",
    "ROUTING_TABLE",
    "FALLBACK_CHAIN",
    # App Building
    "AppBuilder",
    "AppBuildResult",
    "AppDetector",
    "AppProfile",
    "ArchitectureDecision",
    "ArchitectureAdvisor",
    # Validation
    "run_validators",
    "async_run_validators",
    "VALIDATORS",
    "ValidationResult",
    # Codebase Enhancer
    "CodebaseAnalyzer",
    "CodebaseMap",
    "OpenRouterSync",
    "CodebaseUnderstanding",
    "CodebaseProfile",
    "Improvement",
    "ImprovementSuggester",
    # Hybrid Search Pipeline
    "QueryExpander",
    "HybridSearchPipeline",
    # Rate Limiting
    "RateLimiter",
    "RateLimitExceeded",
    # Session Lifecycle
    "SessionLifecycleManager",
    # Model Routing Tiers
    "ModelTier",
    "TIER_ROUTING",
    "PHASE_TO_TIER",
    "select_model",
    "get_tier_for_phase",
    "HAS_MODEL_ROUTING",
    # Autonomy
    "AutonomyLevel",
    "AutonomyConfig",
    "AUTONOMY_PRESETS",
    "get_autonomy_config",
    "requires_approval",
    "HAS_AUTONOMY",
    # Verification
    "VerificationLevel",
    "VerificationResult",
    "REPLVerifier",
    "self_healing_loop",
    "HAS_VERIFICATION",
    # Policy
    "ModelProfile",
    "Policy",
    "PolicySet",
    "JobSpec",
    "PolicyEngine",
    "PolicyViolationError",
    "ConstraintPlanner",
    "EnforcementMode",
    "RateLimit",
    "PolicyHierarchy",
    # Optimization
    "OptimizationBackend",
    "GreedyBackend",
    "WeightedSumBackend",
    "ParetoBackend",
    # Audit & Telemetry
    "AuditLog",
    "AuditRecord",
    "TelemetryCollector",
    "TelemetryStore",
    "HookRegistry",
    "HookEventType",
    "DashboardHookRegistry",
    "MetricsExporter",
    "ConsoleExporter",
    "JSONExporter",
    "PrometheusExporter",
    "TracingConfig",
    "configure_tracing",
    "get_tracer",
    # Cost Management
    "BudgetHierarchy",
    "CostPredictor",
    "CostForecaster",
    "ForecastReport",
    "RiskLevel",
    # Advanced Features
    "AdaptiveRouter",
    "ModelState",
    "SemanticCache",
    "DuplicationDetector",
    "ProfileAggregator",
    "RunRecord",
    "RemediationEngine",
    "RemediationStrategy",
    "RemediationPlan",
    # Output & Progress
    "write_output_dir",
    "ProgressWriter",
    "ProgressEntry",
    "ExecutionPlan",
    "TaskPlan",
    "DryRunRenderer",
    # Exceptions
    "ApplicationError",
    "ConfigurationError",
    "MissingAPIKeyError",
    "OrchestratorError",
    "BudgetExceededError",
    "TimeoutError",
    "ModelError",
    "ModelUnavailableError",
    "RateLimitError",
    "TokenLimitError",
    "AuthenticationError",
    "TaskError",
    "TaskValidationError",
    "TaskTimeoutError",
    "TaskRetryExhaustedError",
    "PolicyError",
    "PolicyViolationError",
    "CacheError",
    "StateError",
    # Logging
    "get_logger",
    "set_correlation_id",
    "get_correlation_id",
    "configure_logging",
    "LogContext",
    "JSONFormatter",
    "TextFormatter",
    # Control Plane
    "SLAs",
    "InputSpec",
    "Constraints",
    "JobSpecV2",
    "RoutingHint",
    "ValidationRule",
    "EscalationRule",
    "PolicySpecV2",
    "Decision",
    "MonitorResult",
    "ReferenceMonitor",
    "ControlPlane",
    "RoutingPlan",
    "SpecValidationError",
    "PolicyViolation",
    "AgentDraft",
    "OrchestrationAgent",
    # Agents
    "AgentPool",
    "TaskChannel",
    # Project File
    "load_project_file",
    # Utility
    "build_default_profiles",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Import remaining optional modules
# ═══════════════════════════════════════════════════════════════════════════════

# Output Organization & Test Automation
try:
    from .output_organizer import (
        CacheMessageSuppressor,
        OrganizationReport,
        OutputOrganizer,
        organize_project_output,
        suppress_cache_messages,
    )
    from .output_organizer import (
        TestResult as OrgTestResult,
    )

    __all__.extend(
        [
            "OutputOrganizer",
            "organize_project_output",
            "OrganizationReport",
            "OrgTestResult",
            "suppress_cache_messages",
            "CacheMessageSuppressor",
        ]
    )
except ImportError:
    pass

# InDesign Plugin Rules
try:
    from .indesign_plugin_rules import (
        InDesignPluginRules,
        InDesignRulesConfig,
        generate_indesign_plugin_rules,
    )
except ImportError:
    InDesignPluginRules = None
    InDesignRulesConfig = None
    generate_indesign_plugin_rules = None

# Knowledge Management
try:
    from .knowledge_base import (
        KnowledgeArtifact,
        KnowledgeBase,
        KnowledgeType,
        Pattern,
        get_knowledge_base,
    )

    __all__.extend(
        [
            "KnowledgeBase",
            "KnowledgeArtifact",
            "KnowledgeType",
            "Pattern",
            "get_knowledge_base",
        ]
    )
except ImportError:
    pass

# Project Management
try:
    from .project_manager import (
        CriticalPathAnalyzer,
        Milestone,
        ProjectManager,
        Resource,
        ResourceScheduler,
        ResourceType,
        Risk,
        TaskPriority,
        TaskSchedule,
        get_project_manager,
    )

    __all__.extend(
        [
            "ProjectManager",
            "TaskSchedule",
            "Resource",
            "Milestone",
            "Risk",
            "CriticalPathAnalyzer",
            "ResourceScheduler",
            "TaskPriority",
            "ResourceType",
            "get_project_manager",
        ]
    )
except ImportError:
    pass

# Product Management
try:
    from .product_manager import (
        Feature,
        FeatureFlagManager,
        FeaturePriority,
        FeatureStatus,
        ProductManager,
        Release,
        RICEScore,
        SentimentAnalyzer,
        UserFeedback,
        get_product_manager,
    )

    __all__.extend(
        [
            "ProductManager",
            "Feature",
            "Release",
            "UserFeedback",
            "RICEScore",
            "FeatureStatus",
            "FeaturePriority",
            "FeatureFlagManager",
            "SentimentAnalyzer",
            "get_product_manager",
        ]
    )
except ImportError:
    pass

# Quality Control
try:
    from .quality_control import (
        CodeMetrics,
        QualityController,
        QualityIssue,
        QualityReport,
        QualitySeverity,
        StaticAnalyzer,
        TestLevel,
        TestResult,
        TestRunner,
        get_quality_controller,
    )

    __all__.extend(
        [
            "QualityController",
            "QualityReport",
            "TestResult",
            "QualityIssue",
            "CodeMetrics",
            "StaticAnalyzer",
            "TestRunner",
            "TestLevel",
            "QualitySeverity",
            "get_quality_controller",
        ]
    )
except ImportError:
    pass

# Architecture Rules
try:
    from .architecture_rules import (
        APIStyle,
        ArchitecturalStyle,
        ArchitectureDecision,
        ArchitectureRulesEngine,
        CodingStandard,
        DatabaseType,
        ProgrammingParadigm,
        ProjectRules,
        RulesGenerator,
        TechnologyStack,
        create_architecture_rules,
    )

    __all__.extend(
        [
            "ArchitectureRulesEngine",
            "RulesGenerator",
            "create_architecture_rules",
            "ProjectRules",
            "ArchitectureDecision",
            "TechnologyStack",
            "CodingStandard",
            "ArchitecturalStyle",
            "ProgrammingParadigm",
            "APIStyle",
            "DatabaseType",
            "InDesignPluginRules",
            "InDesignRulesConfig",
            "generate_indesign_plugin_rules",
        ]
    )
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
# NASH STABILITY FEATURES (v6.1) — Strategic Competitive Moat
# ═══════════════════════════════════════════════════════════════════════════════

# Model Performance Knowledge Graph
try:
    from .knowledge_graph import (
        Edge,
        EdgeType,
        Node,
        NodeType,
        PathResult,
        PerformanceKnowledgeGraph,
        SimilarityMatch,
        get_knowledge_graph,
    )

    __all__.extend(
        [
            "PerformanceKnowledgeGraph",
            "get_knowledge_graph",
            "NodeType",
            "EdgeType",
            "Node",
            "Edge",
            "PathResult",
            "SimilarityMatch",
        ]
    )
except ImportError:
    pass

# Adaptive Prompt Template System
try:
    from .adaptive_templates import (
        PYTHON_TEMPLATES,
        WEB_TEMPLATES,
        AdaptiveTemplateSystem,
        ContextProfile,
        TemplatePerformance,
        TemplateStyle,
        TemplateVariant,
        get_adaptive_template_system,
    )

    __all__.extend(
        [
            "AdaptiveTemplateSystem",
            "get_adaptive_template_system",
            "TemplateVariant",
            "TemplateStyle",
            "TemplatePerformance",
            "ContextProfile",
            "PYTHON_TEMPLATES",
            "WEB_TEMPLATES",
        ]
    )
except ImportError:
    pass

# Predictive Cost-Quality Frontier API
try:
    from .pareto_frontier import (
        CostQualityFrontier,
        FrontierRecommendation,
        ModelPrediction,
        Objective,
        OptimizationDirection,
        ParetoPoint,
        get_cost_quality_frontier,
    )

    __all__.extend(
        [
            "CostQualityFrontier",
            "get_cost_quality_frontier",
            "Objective",
            "OptimizationDirection",
            "ModelPrediction",
            "ParetoPoint",
            "FrontierRecommendation",
        ]
    )
except ImportError:
    pass

# Cross-Organization Federated Learning
try:
    from .federated_learning import (
        DifferentialPrivacyEngine,
        FederatedLearningOrchestrator,
        GlobalBaseline,
        LocalModel,
        ModelInsight,
        PrivacyBudget,
        PrivacyMechanism,
        get_federated_orchestrator,
    )

    __all__.extend(
        [
            "FederatedLearningOrchestrator",
            "get_federated_orchestrator",
            "DifferentialPrivacyEngine",
            "PrivacyBudget",
            "ModelInsight",
            "GlobalBaseline",
            "LocalModel",
            "PrivacyMechanism",
        ]
    )
except ImportError:
    pass

# Nash-Stable Orchestrator (Integration of all features)
try:
    from .nash_stable_orchestrator import (
        NashStabilityMetrics,
        NashStableOrchestrator,
        get_nash_stable_orchestrator,
    )

    __all__.extend(
        [
            "NashStableOrchestrator",
            "get_nash_stable_orchestrator",
            "NashStabilityMetrics",
        ]
    )
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
# NASH STABILITY INFRASTRUCTURE (v6.1.1) — Events, Backup, Auto-Tuning
# ═══════════════════════════════════════════════════════════════════════════════

# Event System
try:
    from .nash_events import (
        AutoTuningTriggeredEvent,
        BackupCreatedEvent,
        DriftDetectedEvent,
        EventType,
        FrontierComputedEvent,
        InsightContributedEvent,
        KnowledgeGraphUpdatedEvent,
        NashEvent,
        NashEventBus,
        NashEventHandlers,
        StabilityScoreUpdatedEvent,
        TemplateResultReportedEvent,
        TemplateSelectedEvent,
        get_event_bus,
        on_event,
    )

    __all__.extend(
        [
            "NashEventBus",
            "get_event_bus",
            "on_event",
            "EventType",
            "NashEvent",
            "KnowledgeGraphUpdatedEvent",
            "TemplateSelectedEvent",
            "TemplateResultReportedEvent",
            "FrontierComputedEvent",
            "DriftDetectedEvent",
            "InsightContributedEvent",
            "StabilityScoreUpdatedEvent",
            "AutoTuningTriggeredEvent",
            "BackupCreatedEvent",
            "NashEventHandlers",
        ]
    )
except ImportError:
    pass

# Backup System
try:
    from .nash_backup import (
        BackupComponent,
        BackupFormat,
        BackupManifest,
        NashBackupManager,
        RestoreResult,
        get_backup_manager,
    )

    __all__.extend(
        [
            "NashBackupManager",
            "get_backup_manager",
            "BackupManifest",
            "BackupComponent",
            "RestoreResult",
            "BackupFormat",
        ]
    )
except ImportError:
    pass

# Auto-Tuning System
try:
    from .nash_auto_tuning import (
        AutoTuner,
        MultiArmedBandit,
        OptimizationDirection,
        ParameterConfig,
        TuningResult,
        TuningStrategy,
        get_auto_tuner,
    )

    __all__.extend(
        [
            "AutoTuner",
            "get_auto_tuner",
            "ParameterConfig",
            "TuningResult",
            "OptimizationDirection",
            "TuningStrategy",
            "MultiArmedBandit",
        ]
    )
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Additional Feature Modules (Implemented in v6.x)
# ═══════════════════════════════════════════════════════════════════════════════

# Brain and Cognitive Layer
try:
    from .brain import Brain, CognitiveState, ReasoningStep

    __all__.extend(["Brain", "ReasoningStep", "CognitiveState"])
except ImportError:
    pass

# Evaluation Module
try:
    from .evaluation import EvaluationResult, Evaluator

    __all__.extend(["Evaluator", "EvaluationResult"])
except ImportError:
    pass

# Escalation Module
try:
    from .escalation import EscalationHandler, EscalationResult, EscalationRule

    __all__.extend(["EscalationHandler", "EscalationRule", "EscalationResult"])
except ImportError:
    pass

# Checkpoints Module
try:
    from .checkpoints import Checkpoint, CheckpointManager

    __all__.extend(["CheckpointManager", "Checkpoint"])
except ImportError:
    pass

# Prompt Enhancer Module
try:
    from .prompt_enhancer import PromptEnhancer

    __all__.extend(["PromptEnhancer"])
except ImportError:
    pass

# Cost Analytics Module
try:
    from .cost_analytics import CostAnalytics, CostBreakdown, CostForecast

    __all__.extend(["CostAnalytics", "CostBreakdown", "CostForecast"])
except ImportError:
    pass

# Competitive Intelligence Module
try:
    from .competitive import CompetitiveIntelligence, CompetitiveRecommendation, MarketDataPoint

    __all__.extend(["CompetitiveIntelligence", "CompetitiveRecommendation", "MarketDataPoint"])
except ImportError:
    pass

# Plan-Then-Build Module
try:
    from .plan_then_build import ExecutionPlan, PlanStep, PlanThenBuilder

    __all__.extend(["PlanThenBuilder", "PlanStep", "ExecutionPlan"])
except ImportError:
    pass

# Memory Bank Module
try:
    from .memory_bank import MemoryBank, MemoryEntry

    __all__.extend(["MemoryBank", "MemoryEntry"])
except ImportError:
    pass

# Context Condensing Module
try:
    from .context_condensing import ContextCondenser

    __all__.extend(["ContextCondenser"])
except ImportError:
    pass

# Hierarchy Module
try:
    from .hierarchy import HierarchyManager, Node, NodeType

    __all__.extend(["HierarchyManager", "Node", "NodeType"])
except ImportError:
    pass

# Triggers Module
try:
    from .triggers import Trigger, TriggerConditionOperator, TriggerManager

    __all__.extend(["TriggerManager", "Trigger", "TriggerConditionOperator"])
except ImportError:
    pass

# Workspace Module
try:
    from .workspace import Workspace, WorkspaceManager

    __all__.extend(["WorkspaceManager", "Workspace"])
except ImportError:
    pass

# Gateway Module
try:
    from .gateway import APIGateway, APIRequest, APIResponse, RateLimitInfo

    __all__.extend(["APIGateway", "APIRequest", "APIResponse", "RateLimitInfo"])
except ImportError:
    pass

# Connectors Module
try:
    from .connectors import (
        BaseConnector,
        ConnectorManager,
        DatabaseConnector,
        FileConnector,
        HTTPConnector,
    )

    __all__.extend(
        ["ConnectorManager", "BaseConnector", "DatabaseConnector", "HTTPConnector", "FileConnector"]
    )
except ImportError:
    pass

# Sandbox Module
try:
    from .sandbox import ExecutionResult, Sandbox

    __all__.extend(["Sandbox", "ExecutionResult"])
except ImportError:
    pass

# Context Sources Module
try:
    from .context_sources import BaseContextSource, ContextChunk, ContextSourceManager

    __all__.extend(["ContextSourceManager", "ContextChunk", "BaseContextSource"])
except ImportError:
    pass

# Skills Module
try:
    from .skills import SkillDefinition, SkillManager, get_global_skill_manager

    __all__.extend(["SkillManager", "SkillDefinition", "get_global_skill_manager"])
except ImportError:
    pass

# Drift Module
try:
    from .drift import (
        DriftDetectionResult,
        DriftDetector,
        ModelDriftMonitor,
        get_global_drift_detector,
    )

    __all__.extend(
        ["DriftDetector", "DriftDetectionResult", "ModelDriftMonitor", "get_global_drift_detector"]
    )
except ImportError:
    pass

# Browser Testing Module
try:
    from .browser_testing import BrowserTester, TestStep, get_global_browser_tester
    from .browser_testing import TestResult as BrowserTestResult

    __all__.extend(["BrowserTester", "TestStep", "BrowserTestResult", "get_global_browser_tester"])
except ImportError:
    pass

# Token Optimizer Module
try:
    from .token_optimizer import TokenOptimizer, get_global_token_optimizer

    __all__.extend(["TokenOptimizer", "get_global_token_optimizer"])
except ImportError:
    pass

# A2A Protocol Module
try:
    from .a2a_protocol import (
        A2AClient,
        A2ACoordinator,
        A2AManager,
        A2AMessage,
        A2AQueueManager,
        A2AResponse,
        A2ATask,
        AgentCard,
        AgentState,
        MessagePart,
        TaskResult,
        TaskSendRequest,
        TaskStatus,
        get_global_a2a_coordinator,
    )

    __all__.extend(
        [
            "A2AClient",
            "A2ACoordinator",
            "A2AMessage",
            "A2ATask",
            "A2AResponse",
            "get_global_a2a_coordinator",
            "A2AManager",
            "A2AQueueManager",
            "TaskSendRequest",
            "TaskResult",
            "AgentCard",
            "MessagePart",
            "TaskStatus",
            "AgentState",
        ]
    )
except ImportError:
    pass

# Persona Modes Module
try:
    from .persona_modes import (
        Persona,
        PersonaConfig,
        PersonaModeManager,
        get_global_persona_manager,
    )

    __all__.extend(["PersonaModeManager", "Persona", "PersonaConfig", "get_global_persona_manager"])
except ImportError:
    pass

# Learning Aggregator Module
try:
    from .learning_aggregator import (
        LearningAggregator,
        ModelPerformanceStats,
        RoutingRecommendation,
        TaskPerformanceRecord,
    )

    __all__.extend(
        [
            "LearningAggregator",
            "TaskPerformanceRecord",
            "ModelPerformanceStats",
            "RoutingRecommendation",
        ]
    )
except ImportError:
    pass

# Multi-Tenant Gateway Module
try:
    from .multi_tenant_gateway import MultiTenantGateway, QuotaUsage, Tenant, get_global_gateway

    __all__.extend(["MultiTenantGateway", "Tenant", "QuotaUsage", "get_global_gateway"])
except ImportError:
    pass

# Print optimization info on import
import logging

logger = logging.getLogger("orchestrator")
if HAS_UNIFIED_DASHBOARD and HAS_UNIFIED_EVENTS:
    logger.debug(
        "Orchestrator v6.0 optimizations loaded: "
        f"unified_dashboard={HAS_UNIFIED_DASHBOARD}, "
        f"unified_events={HAS_UNIFIED_EVENTS}"
    )

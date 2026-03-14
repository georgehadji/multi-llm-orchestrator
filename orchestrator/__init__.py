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
from .models import (
    AttemptRecord, Budget, Model, Task, TaskResult, TaskType, TaskStatus,
    ProjectState, ProjectStatus, build_default_profiles,
    COST_TABLE, ROUTING_TABLE, FALLBACK_CHAIN,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Core Engine & Clients
# ═══════════════════════════════════════════════════════════════════════════════
from .engine import Orchestrator
from .api_clients import UnifiedClient, APIResponse
from .cache import DiskCache
from .state import StateManager

# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Unified Dashboard Core (v6.0) — Replaces 7 dashboard implementations
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from .dashboard_core import (
        DashboardCore,
        get_dashboard_core,
        DashboardView,
        ViewContext,
        ViewRegistry,
        run_dashboard,
        MissionControlView,
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
        UnifiedEventBus,
        get_event_bus,
        DomainEvent,
        EventType,
        ProjectStartedEvent,
        ProjectCompletedEvent,
        TaskStartedEvent,
        TaskCompletedEvent,
        TaskFailedEvent,
        TaskProgressEvent,
        CapabilityUsedEvent,
        CapabilityCompletedEvent,
        BudgetWarningEvent,
        ModelSelectedEvent,
        FallbackTriggeredEvent,
        log_capability_use,
        set_current_project,
        get_current_project,
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
from .policy import (
    ModelProfile, Policy, PolicySet, JobSpec,
    EnforcementMode, RateLimit, PolicyHierarchy,
)
from .policy_engine import PolicyEngine
from .planner import ConstraintPlanner

# ═══════════════════════════════════════════════════════════════════════════════
# Validation & Quality Assurance
# ═══════════════════════════════════════════════════════════════════════════════
from .validators import run_validators, async_run_validators, VALIDATORS, ValidationResult
from .resume_detector import ResumeDetector

# ═════════════════════════════════════════════════════════════════════════════==
# Codebase Enhancer
# ═════════════════════════════════════════════════════════════════════════════==
from .codebase_analyzer import CodebaseAnalyzer, CodebaseMap
from .codebase_understanding import CodebaseUnderstanding
from .codebase_profile import CodebaseProfile
from .improvement_suggester import Improvement, ImprovementSuggester
from .query_expander import QueryExpander
from .hybrid_search_pipeline import HybridSearchPipeline
from .rate_limiter import RateLimiter, RateLimitExceeded
from .session_lifecycle import SessionLifecycleManager

# ═══════════════════════════════════════════════════════════════════════════════
# Observability & Telemetry
# ═══════════════════════════════════════════════════════════════════════════════
from .telemetry import TelemetryCollector
from .audit import AuditLog, AuditRecord
from .hooks import DashboardHookRegistry, HookRegistry, EventType as HookEventType
from .metrics import MetricsExporter, ConsoleExporter, JSONExporter, PrometheusExporter
from .tracing import TracingConfig, configure_tracing, get_tracer

# ═══════════════════════════════════════════════════════════════════════════════
# Optimization & Cost Management
# ═══════════════════════════════════════════════════════════════════════════════
from .optimization import OptimizationBackend, GreedyBackend, WeightedSumBackend, ParetoBackend
from .cost import BudgetHierarchy, CostPredictor, CostForecaster, ForecastReport, RiskLevel

# ═══════════════════════════════════════════════════════════════════════════════
# Advanced Features
# ═══════════════════════════════════════════════════════════════════════════════
from .adaptive_router import AdaptiveRouter, ModelState
from .semantic_cache import SemanticCache, DuplicationDetector
from .aggregator import ProfileAggregator, RunRecord
from .remediation import RemediationEngine, RemediationStrategy, RemediationPlan

# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Multi-Level Cache Optimizer (v6.1) — Advanced caching with L1/L2/L3
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from .cache_optimizer import (
        CacheOptimizer,
        CacheConfig,
        L1MemoryCache,
        L2DiskCache,
        L3SemanticCache,
        SmartCacheKeyGenerator,
        CacheWarmer,
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
from .output_writer import write_output_dir
from .progress_writer import ProgressWriter, ProgressEntry
from .dry_run import ExecutionPlan, TaskPlan, DryRunRenderer

# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Backward Compatibility Layer (v6.0)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from .compat import (
        run_live_dashboard,
        run_mission_control,
        run_ant_design_dashboard,
        ProjectEventBus,
        StreamEvent,
        ProjectStarted,
        TaskStarted,
        TaskCompleted,
        ProjectCompleted,
        print_migration_guide,
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
        EnhancedDashboardServer,
        EnhancedDataProvider,
        DashboardIntegration,
        ArchitectureInfo,
        ProjectInfo,
        ActiveTaskInfo,
        ModelStatusInfo,
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
        run_ant_design_dashboard as _legacy_run_antd,
    )
except ImportError:
    AntDesignDashboardServer = None
    _legacy_run_antd = None

try:
    from .dashboard_mission_control import (
        MissionControlServer,
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
        LiveDashboardServer,
        DashboardLiveIntegration,
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
        ProjectEventBus as _legacy_event_bus,
        ProjectStarted as _legacy_project_started,
        TaskStarted as _legacy_task_started,
        TaskCompleted as _legacy_task_completed,
        ProjectCompleted as _legacy_project_completed,
        StreamEvent as _legacy_stream_event,
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
    "Orchestrator", "AttemptRecord", "Budget", "Model", "Task", "TaskResult",
    "TaskType", "TaskStatus", "ProjectState", "ProjectStatus",
    
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
    "COST_TABLE", "ROUTING_TABLE", "FALLBACK_CHAIN",
    
    # App Building
    "AppBuilder", "AppBuildResult", "AppDetector", "AppProfile",
    "ArchitectureDecision", "ArchitectureAdvisor",
    
    # Validation
    "run_validators", "async_run_validators", "VALIDATORS", "ValidationResult",
    
    # Codebase Enhancer
    "CodebaseAnalyzer", "CodebaseMap",
    "CodebaseUnderstanding", "CodebaseProfile",
    "Improvement", "ImprovementSuggester",

    # Hybrid Search Pipeline
    "QueryExpander", "HybridSearchPipeline",

    # Rate Limiting
    "RateLimiter", "RateLimitExceeded",

    # Session Lifecycle
    "SessionLifecycleManager",
    
    # Policy
    "ModelProfile", "Policy", "PolicySet", "JobSpec",
    "PolicyEngine", "PolicyViolationError",
    "ConstraintPlanner", "EnforcementMode", "RateLimit", "PolicyHierarchy",
    
    # Optimization
    "OptimizationBackend", "GreedyBackend", "WeightedSumBackend", "ParetoBackend",
    
    # Audit & Telemetry
    "AuditLog", "AuditRecord",
    "TelemetryCollector", "TelemetryStore",
    "HookRegistry", "HookEventType", "DashboardHookRegistry",
    "MetricsExporter", "ConsoleExporter", "JSONExporter", "PrometheusExporter",
    "TracingConfig", "configure_tracing", "get_tracer",
    
    # Cost Management
    "BudgetHierarchy", "CostPredictor", "CostForecaster", "ForecastReport", "RiskLevel",
    
    # Advanced Features
    "AdaptiveRouter", "ModelState",
    "SemanticCache", "DuplicationDetector",
    "ProfileAggregator", "RunRecord",
    "RemediationEngine", "RemediationStrategy", "RemediationPlan",
    
    # Output & Progress
    "write_output_dir",
    "ProgressWriter", "ProgressEntry",
    "ExecutionPlan", "TaskPlan", "DryRunRenderer",
    
    # Exceptions
    "ApplicationError",
    "ConfigurationError", "MissingAPIKeyError",
    "OrchestratorError", "BudgetExceededError", "TimeoutError",
    "ModelError", "ModelUnavailableError", "RateLimitError", "TokenLimitError", "AuthenticationError",
    "TaskError", "TaskValidationError", "TaskTimeoutError", "TaskRetryExhaustedError",
    "PolicyError", "PolicyViolationError",
    "CacheError", "StateError",
    
    # Logging
    "get_logger", "set_correlation_id", "get_correlation_id", "configure_logging",
    "LogContext", "JSONFormatter", "TextFormatter",
    
    # Control Plane
    "SLAs", "InputSpec", "Constraints", "JobSpecV2",
    "RoutingHint", "ValidationRule", "EscalationRule", "PolicySpecV2",
    "Decision", "MonitorResult", "ReferenceMonitor",
    "ControlPlane", "RoutingPlan", "SpecValidationError", "PolicyViolation",
    "AgentDraft", "OrchestrationAgent",
    
    # Agents
    "AgentPool", "TaskChannel",
    
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
        OutputOrganizer,
        organize_project_output,
        OrganizationReport,
        TestResult as OrgTestResult,
        suppress_cache_messages,
        CacheMessageSuppressor,
    )
    __all__.extend([
        "OutputOrganizer", "organize_project_output", "OrganizationReport",
        "OrgTestResult", "suppress_cache_messages", "CacheMessageSuppressor",
    ])
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
        KnowledgeBase, KnowledgeArtifact, KnowledgeType, Pattern,
        get_knowledge_base,
    )
    __all__.extend([
        "KnowledgeBase", "KnowledgeArtifact", "KnowledgeType", "Pattern",
        "get_knowledge_base",
    ])
except ImportError:
    pass

# Project Management
try:
    from .project_manager import (
        ProjectManager, TaskSchedule, Resource, Milestone, Risk,
        CriticalPathAnalyzer, ResourceScheduler,
        TaskPriority, ResourceType,
        get_project_manager,
    )
    __all__.extend([
        "ProjectManager", "TaskSchedule", "Resource", "Milestone", "Risk",
        "CriticalPathAnalyzer", "ResourceScheduler",
        "TaskPriority", "ResourceType", "get_project_manager",
    ])
except ImportError:
    pass

# Product Management
try:
    from .product_manager import (
        ProductManager, Feature, Release, UserFeedback,
        RICEScore, FeatureStatus, FeaturePriority,
        FeatureFlagManager, SentimentAnalyzer,
        get_product_manager,
    )
    __all__.extend([
        "ProductManager", "Feature", "Release", "UserFeedback",
        "RICEScore", "FeatureStatus", "FeaturePriority",
        "FeatureFlagManager", "SentimentAnalyzer", "get_product_manager",
    ])
except ImportError:
    pass

# Quality Control
try:
    from .quality_control import (
        QualityController, QualityReport, TestResult, QualityIssue,
        CodeMetrics, StaticAnalyzer, TestRunner,
        TestLevel, QualitySeverity,
        get_quality_controller,
    )
    __all__.extend([
        "QualityController", "QualityReport", "TestResult", "QualityIssue",
        "CodeMetrics", "StaticAnalyzer", "TestRunner",
        "TestLevel", "QualitySeverity", "get_quality_controller",
    ])
except ImportError:
    pass

# Architecture Rules
try:
    from .architecture_rules import (
        ArchitectureRulesEngine, RulesGenerator,
        create_architecture_rules,
        ProjectRules, ArchitectureDecision,
        TechnologyStack, CodingStandard,
        ArchitecturalStyle, ProgrammingParadigm,
        APIStyle, DatabaseType,
    )
    __all__.extend([
        "ArchitectureRulesEngine", "RulesGenerator",
        "create_architecture_rules",
        "ProjectRules", "ArchitectureDecision",
        "TechnologyStack", "CodingStandard",
        "ArchitecturalStyle", "ProgrammingParadigm",
    "APIStyle", "DatabaseType",
    "InDesignPluginRules",
    "InDesignRulesConfig",
    "generate_indesign_plugin_rules",
    ])
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
# NASH STABILITY FEATURES (v6.1) — Strategic Competitive Moat
# ═══════════════════════════════════════════════════════════════════════════════

# Model Performance Knowledge Graph
try:
    from .knowledge_graph import (
        PerformanceKnowledgeGraph,
        get_knowledge_graph,
        NodeType,
        EdgeType,
        Node,
        Edge,
        PathResult,
        SimilarityMatch,
    )
    __all__.extend([
        "PerformanceKnowledgeGraph", "get_knowledge_graph",
        "NodeType", "EdgeType", "Node", "Edge",
        "PathResult", "SimilarityMatch",
    ])
except ImportError:
    pass

# Adaptive Prompt Template System
try:
    from .adaptive_templates import (
        AdaptiveTemplateSystem,
        get_adaptive_template_system,
        TemplateVariant,
        TemplateStyle,
        TemplatePerformance,
        ContextProfile,
        PYTHON_TEMPLATES,
        WEB_TEMPLATES,
    )
    __all__.extend([
        "AdaptiveTemplateSystem", "get_adaptive_template_system",
        "TemplateVariant", "TemplateStyle", "TemplatePerformance",
        "ContextProfile", "PYTHON_TEMPLATES", "WEB_TEMPLATES",
    ])
except ImportError:
    pass

# Predictive Cost-Quality Frontier API
try:
    from .pareto_frontier import (
        CostQualityFrontier,
        get_cost_quality_frontier,
        Objective,
        OptimizationDirection,
        ModelPrediction,
        ParetoPoint,
        FrontierRecommendation,
    )
    __all__.extend([
        "CostQualityFrontier", "get_cost_quality_frontier",
        "Objective", "OptimizationDirection",
        "ModelPrediction", "ParetoPoint", "FrontierRecommendation",
    ])
except ImportError:
    pass

# Cross-Organization Federated Learning
try:
    from .federated_learning import (
        FederatedLearningOrchestrator,
        get_federated_orchestrator,
        DifferentialPrivacyEngine,
        PrivacyBudget,
        ModelInsight,
        GlobalBaseline,
        LocalModel,
        PrivacyMechanism,
    )
    __all__.extend([
        "FederatedLearningOrchestrator", "get_federated_orchestrator",
        "DifferentialPrivacyEngine", "PrivacyBudget",
        "ModelInsight", "GlobalBaseline", "LocalModel",
        "PrivacyMechanism",
    ])
except ImportError:
    pass

# Nash-Stable Orchestrator (Integration of all features)
try:
    from .nash_stable_orchestrator import (
        NashStableOrchestrator,
        get_nash_stable_orchestrator,
        NashStabilityMetrics,
    )
    __all__.extend([
        "NashStableOrchestrator", "get_nash_stable_orchestrator",
        "NashStabilityMetrics",
    ])
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
# NASH STABILITY INFRASTRUCTURE (v6.1.1) — Events, Backup, Auto-Tuning
# ═══════════════════════════════════════════════════════════════════════════════

# Event System
try:
    from .nash_events import (
        NashEventBus,
        get_event_bus,
        on_event,
        EventType,
        NashEvent,
        KnowledgeGraphUpdatedEvent,
        TemplateSelectedEvent,
        TemplateResultReportedEvent,
        FrontierComputedEvent,
        DriftDetectedEvent,
        InsightContributedEvent,
        StabilityScoreUpdatedEvent,
        AutoTuningTriggeredEvent,
        BackupCreatedEvent,
        NashEventHandlers,
    )
    __all__.extend([
        "NashEventBus", "get_event_bus", "on_event",
        "EventType", "NashEvent",
        "KnowledgeGraphUpdatedEvent", "TemplateSelectedEvent",
        "TemplateResultReportedEvent", "FrontierComputedEvent",
        "DriftDetectedEvent", "InsightContributedEvent",
        "StabilityScoreUpdatedEvent", "AutoTuningTriggeredEvent",
        "BackupCreatedEvent", "NashEventHandlers",
    ])
except ImportError:
    pass

# Backup System
try:
    from .nash_backup import (
        NashBackupManager,
        get_backup_manager,
        BackupManifest,
        BackupComponent,
        RestoreResult,
        BackupFormat,
    )
    __all__.extend([
        "NashBackupManager", "get_backup_manager",
        "BackupManifest", "BackupComponent",
        "RestoreResult", "BackupFormat",
    ])
except ImportError:
    pass

# Auto-Tuning System
try:
    from .nash_auto_tuning import (
        AutoTuner,
        get_auto_tuner,
        ParameterConfig,
        TuningResult,
        OptimizationDirection,
        TuningStrategy,
        MultiArmedBandit,
    )
    __all__.extend([
        "AutoTuner", "get_auto_tuner",
        "ParameterConfig", "TuningResult",
        "OptimizationDirection", "TuningStrategy",
        "MultiArmedBandit",
    ])
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

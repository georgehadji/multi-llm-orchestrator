"""
Multi-LLM Orchestrator
======================
Author: Georgios-Chrysovalantis Chatzivantsidis
Local multi-model orchestration for autonomous project completion.
Supports: OpenAI GPT, Google Gemini, DeepSeek, Kimi K2.5, MiniMax, Zhipu.

Usage:
    from orchestrator import Orchestrator, Budget

    budget = Budget(max_usd=8.0, max_time_seconds=5400)
    orch = Orchestrator(budget=budget)
    state = asyncio.run(orch.run_project(
        project_description="...",
        success_criteria="...",
    ))
"""

__version__ = "1.2.0"

# ═══════════════════════════════════════════════════════════════════════════════
# Core Models & Types
# ═══════════════════════════════════════════════════════════════════════════════
from .models import (
    AttemptRecord, Budget, Model, Task, TaskResult, TaskType, TaskStatus,
    ProjectState, ProjectStatus, build_default_profiles,
    COST_TABLE, ROUTING_TABLE, FALLBACK_CHAIN,  # Export for inspection
)

# ═══════════════════════════════════════════════════════════════════════════════
# Core Engine & Clients
# ═══════════════════════════════════════════════════════════════════════════════
from .engine import Orchestrator
from .api_clients import UnifiedClient, APIResponse
from .cache import DiskCache
from .state import StateManager

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

# ═══════════════════════════════════════════════════════════════════════════════
# Observability & Telemetry
# ═══════════════════════════════════════════════════════════════════════════════
from .telemetry import TelemetryCollector
from .audit import AuditLog, AuditRecord
from .hooks import HookRegistry, EventType
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
# Output & Progress
# ═══════════════════════════════════════════════════════════════════════════════
from .output_writer import write_output_dir
from .progress_writer import ProgressWriter, ProgressEntry
from .streaming import (
    ProjectEventBus,
    ProjectStarted,
    TaskStarted,
    TaskProgressUpdate,
    TaskCompleted,
    TaskFailed,
    BudgetWarning,
    ProjectCompleted,
    StreamEvent,
)
from .progress import ProgressRenderer
from .visualization import DagRenderer

# ═══════════════════════════════════════════════════════════════════════════════
# Project Enhancement & Analysis
# ═══════════════════════════════════════════════════════════════════════════════
from .enhancer import Enhancement, ProjectEnhancer
from .codebase_analyzer import CodebaseAnalyzer, CodebaseMap
from .codebase_profile import CodebaseProfile
from .codebase_reader import CodebaseReader, CodebaseContext
from .codebase_understanding import CodebaseUnderstanding
from .improvement_suggester import ImprovementSuggester, Improvement

# ═══════════════════════════════════════════════════════════════════════════════
# App Building & Assembly
# ═══════════════════════════════════════════════════════════════════════════════
from .app_builder import AppBuildResult, AppBuilder
from .app_detector import AppDetector, AppProfile
from .architecture_advisor import ArchitectureDecision, ArchitectureAdvisor
from .project_assembler import ProjectAssembler, DependencyAnalyzer, ModuleInfo

# ═══════════════════════════════════════════════════════════════════════════════
# Execution Planning & Dry Run
# ═══════════════════════════════════════════════════════════════════════════════
from .dry_run import ExecutionPlan, TaskPlan, DryRunRenderer

# ═══════════════════════════════════════════════════════════════════════════════
# Production-Ready: Exception Hierarchy & Logging
# ═══════════════════════════════════════════════════════════════════════════════
from .exceptions import (
    ApplicationError,
    ConfigurationError,
    MissingAPIKeyError,
    OrchestratorError,
    BudgetExceededError,
    TimeoutError,
    ModelError,
    ModelUnavailableError,
    RateLimitError,
    TokenLimitError,
    AuthenticationError,
    TaskError,
    TaskValidationError,
    TaskTimeoutError,
    TaskRetryExhaustedError,
    PolicyError,
    PolicyViolationError,
    CacheError,
    StateError,
)
from .logging import (
    get_logger,
    set_correlation_id,
    get_correlation_id,
    configure_logging,
    LogContext,
    JSONFormatter,
    TextFormatter,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Control Plane & Governance (v2)
# ═══════════════════════════════════════════════════════════════════════════════
from .specs import (
    SLAs, InputSpec, Constraints,
    JobSpecV2,
    RoutingHint, ValidationRule, EscalationRule,
    PolicySpecV2,
)
from .reference_monitor import Decision, MonitorResult, ReferenceMonitor
from .control_plane import ControlPlane, RoutingPlan, SpecValidationError, PolicyViolation
from .orchestration_agent import AgentDraft, OrchestrationAgent

# ═══════════════════════════════════════════════════════════════════════════════
# Policy DSL
# ═══════════════════════════════════════════════════════════════════════════════
from .policy_dsl import load_policy_file, load_policy_dict, PolicyAnalyzer, AnalysisReport

# ═══════════════════════════════════════════════════════════════════════════════
# Agents & Channels
# ═══════════════════════════════════════════════════════════════════════════════
from .agents import AgentPool, TaskChannel

# ═══════════════════════════════════════════════════════════════════════════════
# Project File & Assembly
# ═══════════════════════════════════════════════════════════════════════════════
from .project_file import load_project_file

# ═══════════════════════════════════════════════════════════════════════════════
# Telemetry Store
# ═══════════════════════════════════════════════════════════════════════════════
from .telemetry_store import TelemetryStore

# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard & Mission Control
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from .dashboard import run_dashboard, DashboardServer
except ImportError:
    run_dashboard = None
    DashboardServer = None

# ═══════════════════════════════════════════════════════════════════════════════
# Performance Optimization (v5.0)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from .performance import (
        LRUCache, RedisCache, CacheManager,
        ConnectionPool, MetricsCollector, QueryOptimizer,
        cached, cache_invalidate, get_cache,
        PerformanceMonitor,
    )
except ImportError:
    LRUCache = None
    RedisCache = None
    CacheManager = None
    ConnectionPool = None
    MetricsCollector = None
    QueryOptimizer = None
    cached = None
    cache_invalidate = None
    get_cache = None
    PerformanceMonitor = None

try:
    from .monitoring import (
        MetricsRegistry, KPIReporter, KPIDefinition, KPITier, KPIThreshold,
        HealthChecker, health_checker, metrics, STANDARD_KPIS,
        monitor_endpoint, monitor_async_task,
    )
except ImportError:
    MetricsRegistry = None
    KPIReporter = None
    KPIDefinition = None
    KPITier = None
    KPIThreshold = None
    HealthChecker = None
    STANDARD_KPIS = None
    monitor_endpoint = None
    monitor_async_task = None

# Optimized dashboard (v5.0)
try:
    from .dashboard_optimized import (
        OptimizedDashboardServer,
        PerformanceConfig,
        DebouncedUpdater,
        run_dashboard as run_optimized_dashboard,
    )
except ImportError:
    OptimizedDashboardServer = None
    PerformanceConfig = None
    DebouncedUpdater = None
    run_optimized_dashboard = None

# Real-time dashboard with live data (v5.1)
try:
    from .dashboard_real import (
        DashboardServerRealtime,
        RealtimeDataProvider,
        run_dashboard_realtime,
    )
except ImportError:
    DashboardServerRealtime = None
    RealtimeDataProvider = None
    run_dashboard_realtime = None

# ═══════════════════════════════════════════════════════════════════════════════
# Management Systems (v5.1)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from .knowledge_base import (
        KnowledgeBase, KnowledgeArtifact, KnowledgeType, Pattern,
        get_knowledge_base,
    )
except ImportError:
    KnowledgeBase = None
    KnowledgeArtifact = None
    KnowledgeType = None
    Pattern = None
    get_knowledge_base = None

try:
    from .project_manager import (
        ProjectManager, TaskSchedule, Resource, Milestone, Risk,
        CriticalPathAnalyzer, ResourceScheduler,
        TaskPriority, ResourceType,
        get_project_manager,
    )
except ImportError:
    ProjectManager = None
    TaskSchedule = None
    Resource = None
    Milestone = None
    Risk = None
    CriticalPathAnalyzer = None
    ResourceScheduler = None
    TaskPriority = None
    ResourceType = None
    get_project_manager = None

try:
    from .product_manager import (
        ProductManager, Feature, Release, UserFeedback,
        RICEScore, FeatureStatus, FeaturePriority,
        FeatureFlagManager, SentimentAnalyzer,
        get_product_manager,
    )
except ImportError:
    ProductManager = None
    Feature = None
    Release = None
    UserFeedback = None
    RICEScore = None
    FeatureStatus = None
    FeaturePriority = None
    FeatureFlagManager = None
    SentimentAnalyzer = None
    get_product_manager = None

try:
    from .quality_control import (
        QualityController, QualityReport, TestResult, QualityIssue,
        CodeMetrics, StaticAnalyzer, TestRunner,
        TestLevel, QualitySeverity,
        get_quality_controller,
    )
except ImportError:
    QualityController = None
    QualityReport = None
    TestResult = None
    QualityIssue = None
    CodeMetrics = None
    StaticAnalyzer = None
    TestRunner = None
    TestLevel = None
    QualitySeverity = None
    get_quality_controller = None

# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics & Debugging (v5.1)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from .diagnostics import (
        SystemDiagnostic, ProjectDiagnostic,
        DiagnosticReport, Issue,
        HealthStatus, Severity,
        print_diagnostic_report,
    )
except ImportError:
    SystemDiagnostic = None
    ProjectDiagnostic = None
    DiagnosticReport = None
    Issue = None
    HealthStatus = None
    Severity = None
    print_diagnostic_report = None

# ═══════════════════════════════════════════════════════════════════════════════
# Project Analysis (v5.1)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from .project_analyzer import (
        ProjectAnalyzer, analyze_project,
        ProjectAnalysisReport, ImprovementSuggestion,
        SuggestionPriority, SuggestionCategory,
        CodeIssue, ArchitectureInsight,
    )
except ImportError:
    ProjectAnalyzer = None
    analyze_project = None
    ProjectAnalysisReport = None
    ImprovementSuggestion = None
    SuggestionPriority = None
    SuggestionCategory = None
    CodeIssue = None
    ArchitectureInsight = None


__all__ = [
    # Core
    "Orchestrator", "AttemptRecord", "Budget", "Model", "Task", "TaskResult",
    # Dashboard (optional - requires fastapi/uvicorn)
    "run_dashboard", "DashboardServer",
    # Performance Optimization (v5.0)
    "LRUCache", "RedisCache", "CacheManager", "ConnectionPool",
    "MetricsCollector", "QueryOptimizer", "cached", "cache_invalidate", "get_cache",
    "PerformanceMonitor",
    # Monitoring & KPIs (v5.0)
    "MetricsRegistry", "KPIReporter", "KPIDefinition", "KPITier", "KPIThreshold",
    "HealthChecker", "health_checker", "metrics", "STANDARD_KPIS",
    "monitor_endpoint", "monitor_async_task",
    # Optimized Dashboard (v5.0)
    "OptimizedDashboardServer", "PerformanceConfig", "DebouncedUpdater",
    "run_optimized_dashboard",
    # Real-time Dashboard (v5.1)
    "DashboardServerRealtime", "RealtimeDataProvider", "run_dashboard_realtime",
    # Knowledge Management (v5.1)
    "KnowledgeBase", "KnowledgeArtifact", "KnowledgeType", "Pattern",
    "get_knowledge_base",
    # Project Management (v5.1)
    "ProjectManager", "TaskSchedule", "Resource", "Milestone", "Risk",
    "CriticalPathAnalyzer", "ResourceScheduler",
    "TaskPriority", "ResourceType",
    "get_project_manager",
    # Product Management (v5.1)
    "ProductManager", "Feature", "Release", "UserFeedback",
    "RICEScore", "FeatureStatus", "FeaturePriority",
    "FeatureFlagManager", "SentimentAnalyzer",
    "get_product_manager",
    # Quality Control (v5.1)
    "QualityController", "QualityReport", "TestResult", "QualityIssue",
    "CodeMetrics", "StaticAnalyzer", "TestRunner",
    "TestLevel", "QualitySeverity",
    "get_quality_controller",
    # Diagnostics & Debugging (v5.1)
    "SystemDiagnostic", "ProjectDiagnostic",
    "DiagnosticReport", "Issue",
    "HealthStatus", "Severity",
    "print_diagnostic_report",
    # Project Analysis (v5.1)
    "ProjectAnalyzer", "analyze_project",
    "ProjectAnalysisReport", "ImprovementSuggestion",
    "SuggestionPriority", "SuggestionCategory",
    "CodeIssue", "ArchitectureInsight",
    "TaskType", "TaskStatus", "ProjectState", "ProjectStatus",
    "UnifiedClient", "APIResponse",
    "DiskCache", "StateManager", "ResumeDetector",
    # Routing & Cost Tables (exported for inspection/config)
    "COST_TABLE", "ROUTING_TABLE", "FALLBACK_CHAIN",
    # App Building
    "AppBuilder", "AppBuildResult", "AppDetector", "AppProfile",
    "ArchitectureDecision", "ArchitectureAdvisor",
    # Validation
    "run_validators", "async_run_validators", "VALIDATORS", "ValidationResult",
    # Policy
    "ModelProfile", "Policy", "PolicySet", "JobSpec",
    "PolicyEngine", "PolicyViolationError",
    "ConstraintPlanner", "EnforcementMode", "RateLimit", "PolicyHierarchy",
    "load_policy_file", "load_policy_dict", "PolicyAnalyzer", "AnalysisReport",
    # Optimization
    "OptimizationBackend", "GreedyBackend", "WeightedSumBackend", "ParetoBackend",
    # Audit & Telemetry
    "AuditLog", "AuditRecord",
    "TelemetryCollector", "TelemetryStore",
    "HookRegistry", "EventType",
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
    "ProjectEventBus", "ProjectStarted", "TaskStarted", "TaskProgressUpdate",
    "TaskCompleted", "TaskFailed", "BudgetWarning", "ProjectCompleted", "StreamEvent",
    "ProgressRenderer", "DagRenderer",
    # Project Enhancement
    "Enhancement", "ProjectEnhancer",
    "CodebaseAnalyzer", "CodebaseMap", "CodebaseProfile",
    "CodebaseReader", "CodebaseContext", "CodebaseUnderstanding",
    "ImprovementSuggester", "Improvement",
    # Project Assembly
    "ProjectAssembler", "DependencyAnalyzer", "ModuleInfo",
    # Execution Planning
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
    # Control Plane v2
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

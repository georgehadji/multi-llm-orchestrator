"""
Architecture Rules Engine
=========================
Automatically generates and enforces architecture decisions.

Features:
- Optimal architecture selection based on project requirements
- Technology stack recommendations
- Constraint enforcement throughout project
- Rules file generation (.orchestrator-rules.yml)

Usage:
    from orchestrator.architecture_rules import ArchitectureRulesEngine, RulesGenerator
    
    engine = ArchitectureRulesEngine()
    rules = await engine.generate_rules(project_description, criteria)
    
    # Save rules file
    RulesGenerator.save_rules(rules, output_dir)
"""
from __future__ import annotations

import yaml
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import asyncio

from .log_config import get_logger
from .models import TaskType

logger = get_logger(__name__)


class ArchitecturalStyle(Enum):
    """High-level architectural styles."""
    MICROSERVICES = "microservices"
    MONOLITH = "monolith"
    SERVERLESS = "serverless"
    LAYERED = "layered"
    HEXAGONAL = "hexagonal"
    EVENT_DRIVEN = "event_driven"
    CQRS = "cqrs"
    MODULAR_MONOLITH = "modular_monolith"


class ProgrammingParadigm(Enum):
    """Programming paradigms."""
    OBJECT_ORIENTED = "object_oriented"
    FUNCTIONAL = "functional"
    PROCEDURAL = "procedural"
    DECLARATIVE = "declarative"
    REACTIVE = "reactive"


class APIStyle(Enum):
    """API architectural styles."""
    REST = "rest"
    GRAPHQL = "graphql"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    WEBHOOK = "webhook"


class DatabaseType(Enum):
    """Database types."""
    RELATIONAL = "relational"
    DOCUMENT = "document"
    KEY_VALUE = "key_value"
    GRAPH = "graph"
    TIME_SERIES = "time_series"
    COLUMNAR = "columnar"
    NONE = "none"  # For projects without database (hardcoded data, static files)


@dataclass
class TechnologyStack:
    """Recommended technology stack."""
    primary_language: str
    secondary_languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    libraries: List[str] = field(default_factory=list)
    databases: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    infrastructure: List[str] = field(default_factory=list)


@dataclass
class ArchitectureDecision:
    """Complete architecture decision."""
    # Core decisions
    style: ArchitecturalStyle
    paradigm: ProgrammingParadigm
    
    # Structure
    api_style: APIStyle
    database_type: DatabaseType
    
    # Stack
    stack: TechnologyStack
    
    # Constraints
    constraints: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    anti_patterns: List[str] = field(default_factory=list)
    
    # Rationale
    rationale: str = ""
    tradeoffs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "style": self.style.value,
            "paradigm": self.paradigm.value,
            "api_style": self.api_style.value,
            "database_type": self.database_type.value,
            "stack": asdict(self.stack),
            "constraints": self.constraints,
            "patterns": self.patterns,
            "anti_patterns": self.anti_patterns,
            "rationale": self.rationale,
            "tradeoffs": self.tradeoffs,
        }


@dataclass
class CodingStandard:
    """Coding standards and conventions."""
    naming_conventions: Dict[str, str] = field(default_factory=dict)
    code_style: str = ""
    documentation_required: bool = True
    type_hints: bool = True
    max_line_length: int = 100
    max_complexity: int = 10
    test_coverage_min: float = 80.0


@dataclass
class ProjectRules:
    """Complete project rules."""
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    project_type: str = ""
    
    # Architecture
    architecture: ArchitectureDecision = field(default_factory=lambda: ArchitectureDecision(
        style=ArchitecturalStyle.LAYERED,
        paradigm=ProgrammingParadigm.OBJECT_ORIENTED,
        api_style=APIStyle.REST,
        database_type=DatabaseType.RELATIONAL,
        stack=TechnologyStack(primary_language="python")
    ))
    
    # Standards
    coding_standards: CodingStandard = field(default_factory=CodingStandard)
    
    # Rules
    allowed_imports: List[str] = field(default_factory=list)
    forbidden_imports: List[str] = field(default_factory=list)
    required_patterns: List[str] = field(default_factory=list)
    forbidden_patterns: List[str] = field(default_factory=list)
    
    # Quality gates
    quality_gates: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata (not serialized)
    _llm_generated: bool = field(default=False, repr=False)
    _llm_optimized: bool = field(default=False, repr=False)
    _llm_model: str = field(default="", repr=False)
    
    def to_yaml(self) -> str:
        """Convert to YAML format."""
        data = asdict(self)
        # Remove internal metadata fields
        data.pop('_llm_generated', None)
        data.pop('_llm_optimized', None)
        data.pop('_llm_model', None)
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Remove internal metadata fields
        data.pop('_llm_generated', None)
        data.pop('_llm_optimized', None)
        data.pop('_llm_model', None)
        return data


class ArchitectureAnalyzer:
    """Analyze project requirements to determine optimal architecture."""
    
    # Keywords that trigger specific architectures
    ARCHITECTURE_TRIGGERS = {
        ArchitecturalStyle.MICROSERVICES: [
            "microservice", "distributed", "scale", "independent deployment",
            "service mesh", "kubernetes", "docker swarm", "multiple teams"
        ],
        ArchitecturalStyle.SERVERLESS: [
            "serverless", "lambda", "function", "event-triggered", "pay-per-use",
            "no server management", "aws lambda", "azure functions"
        ],
        ArchitecturalStyle.EVENT_DRIVEN: [
            "event-driven", "kafka", "message queue", "message-queue", 
            "streaming", "event sourcing", "event-sourcing", "cqrs",
            "pub-sub", "pub/sub", "rabbitmq", "event bus"
        ],
        ArchitecturalStyle.HEXAGONAL: [
            "ports", "adapters", "testable", "ports and adapters",
            "dependency injection", "clean architecture"
        ],
        ArchitecturalStyle.CQRS: [
            "read model", "write model", "command", "query", "separate",
            "event sourcing", "complex queries"
        ],
    }
    
    # Technology recommendations based on project type
    TECH_STACKS = {
        "web_api": TechnologyStack(
            primary_language="python",
            frameworks=["fastapi", "pydantic"],
            libraries=["uvicorn", "httpx", "sqlalchemy"],
            databases=["postgresql"],
            tools=["pytest", "black", "ruff"],
            infrastructure=["docker", "nginx"]
        ),
        "web_frontend": TechnologyStack(
            primary_language="typescript",
            frameworks=["react", "next.js"],
            libraries=["tailwindcss", "zustand", "react-query"],
            tools=["vite", "eslint", "prettier"],
            infrastructure=["vercel", "netlify"]
        ),
        "cli_tool": TechnologyStack(
            primary_language="python",
            frameworks=["typer", "click"],
            libraries=["rich", "pydantic"],
            tools=["pytest", "mypy"],
            infrastructure=["pip", "homebrew"]
        ),
        "data_pipeline": TechnologyStack(
            primary_language="python",
            frameworks=["apache spark", "pandas"],
            libraries=["numpy", "polars", "dask"],
            databases=["postgresql", "clickhouse"],
            tools=["jupyter", "dbt"],
            infrastructure=["airflow", "kubernetes"]
        ),
        "machine_learning": TechnologyStack(
            primary_language="python",
            frameworks=["pytorch", "tensorflow", "scikit-learn"],
            libraries=["numpy", "pandas", "matplotlib"],
            tools=["jupyter", "mlflow", "wandb"],
            infrastructure=["docker", "kubernetes"]
        ),
    }
    
    def analyze(self, description: str, criteria: str) -> ArchitectureDecision:
        """Analyze project and recommend architecture."""
        text = f"{description} {criteria}".lower()
        
        # Determine architectural style
        style = self._detect_architectural_style(text)
        
        # Determine paradigm
        paradigm = self._detect_paradigm(text)
        
        # Determine API style
        api_style = self._detect_api_style(text)
        
        # Determine database type
        database_type = self._detect_database_type(text)
        
        # Determine project type and stack
        project_type = self._detect_project_type(text)
        stack = self.TECH_STACKS.get(project_type, self.TECH_STACKS["web_api"])
        
        # Generate constraints
        constraints = self._generate_constraints(style, paradigm)
        
        # Generate patterns
        patterns = self._generate_patterns(style, paradigm)
        
        # Generate rationale
        rationale = self._generate_rationale(style, paradigm, stack)
        
        return ArchitectureDecision(
            style=style,
            paradigm=paradigm,
            api_style=api_style,
            database_type=database_type,
            stack=stack,
            constraints=constraints,
            patterns=patterns,
            rationale=rationale,
            tradeoffs=self._generate_tradeoffs(style)
        )
    
    def _detect_architectural_style(self, text: str) -> ArchitecturalStyle:
        """Detect best architectural style from description."""
        scores = {style: 0 for style in ArchitecturalStyle}
        
        for style, triggers in self.ARCHITECTURE_TRIGGERS.items():
            for trigger in triggers:
                if trigger in text:
                    scores[style] += 1
        
        # Default to layered if no clear winner
        if max(scores.values()) == 0:
            return ArchitecturalStyle.LAYERED
        
        return max(scores, key=scores.get)
    
    def _detect_paradigm(self, text: str) -> ProgrammingParadigm:
        """Detect programming paradigm."""
        if any(word in text for word in ["functional", "immutable", "pure function"]):
            return ProgrammingParadigm.FUNCTIONAL
        elif any(word in text for word in ["reactive", "stream", "observable"]):
            return ProgrammingParadigm.REACTIVE
        elif any(word in text for word in ["declarative", "configuration"]):
            return ProgrammingParadigm.DECLARATIVE
        else:
            return ProgrammingParadigm.OBJECT_ORIENTED
    
    def _detect_api_style(self, text: str) -> APIStyle:
        """Detect API style with priority to explicit mentions."""
        # Priority 1: Explicit REST API mentions
        if "rest api" in text or "restful" in text:
            return APIStyle.REST
        
        # Priority 2: Other specific APIs
        if "graphql" in text:
            return APIStyle.GRAPHQL
        elif "grpc" in text or "protobuf" in text:
            return APIStyle.GRPC
        elif "websocket" in text or "real-time" in text or "realtime" in text:
            return APIStyle.WEBSOCKET
        else:
            return APIStyle.REST
    
    def _detect_database_type(self, text: str) -> DatabaseType:
        """Detect database type."""
        # Check for explicit "no database" indicators
        no_db_indicators = [
            "no database", "no db", "hardcoded", "in-memory", "mock data",
            "static files", "json file", "no persistence", "without database"
        ]
        if any(phrase in text for phrase in no_db_indicators):
            return DatabaseType.NONE
        
        if any(word in text for word in ["mongodb", "document", "json"]):
            return DatabaseType.DOCUMENT
        elif any(word in text for word in ["redis", "cache", "key-value"]):
            return DatabaseType.KEY_VALUE
        elif any(word in text for word in ["time series", "metrics", "influx"]):
            return DatabaseType.TIME_SERIES
        elif any(word in text for word in ["neo4j", "graph", "relationship"]):
            return DatabaseType.GRAPH
        else:
            return DatabaseType.RELATIONAL
    
    def _detect_project_type(self, text: str) -> str:
        """Detect project type with priority to explicit backend frameworks."""
        # Priority 1: Explicit backend frameworks (these should win over "frontend" keyword)
        backend_frameworks = ["fastapi", "django", "flask", "tornado", "sanic", "starlette"]
        if any(word in text for word in backend_frameworks):
            return "web_api"
        
        # Priority 2: Check for "backend api" phrase
        if "backend api" in text or "backend" in text and "api" in text:
            return "web_api"
        
        # Priority 3: Frontend frameworks (only if no backend framework mentioned)
        if any(word in text for word in ["react", "vue", "angular", "svelte", "solidjs"]):
            return "web_frontend"
        
        # Priority 4: Other types
        if any(word in text for word in ["cli", "command line", "terminal"]):
            return "cli_tool"
        elif any(word in text for word in ["data pipeline", "etl", "batch"]):
            return "data_pipeline"
        elif any(word in text for word in ["machine learning", "ml", "ai", "model"]):
            return "machine_learning"
        
        # Default: web_api for anything with API/server keywords
        if any(word in text for word in ["api", "rest", "server", "backend"]):
            return "web_api"
        
        # Last resort: frontend (only if explicitly mentioned without backend)
        if "frontend" in text or "ui" in text:
            return "web_frontend"
        
        return "web_api"
    
    def _generate_constraints(self, style: ArchitecturalStyle, paradigm: ProgrammingParadigm) -> List[str]:
        """Generate architecture constraints."""
        constraints = [
            "All code must be type-annotated",
            "Maximum cyclomatic complexity of 10 per function",
            "Minimum 80% test coverage",
            "All public APIs must be documented",
        ]
        
        if style == ArchitecturalStyle.MICROSERVICES:
            constraints.extend([
                "Each service must have its own database",
                "Services communicate via events or HTTP",
                "No shared databases between services",
                "Services must be independently deployable",
            ])
        elif style == ArchitecturalStyle.HEXAGONAL:
            constraints.extend([
                "Business logic must not depend on frameworks",
                "All external dependencies through ports/adapters",
                "Domain layer has no external dependencies",
            ])
        
        if paradigm == ProgrammingParadigm.FUNCTIONAL:
            constraints.extend([
                "Prefer pure functions",
                "Minimize mutable state",
                "Use immutable data structures",
            ])
        
        return constraints
    
    def _generate_patterns(self, style: ArchitecturalStyle, paradigm: ProgrammingParadigm) -> List[str]:
        """Generate recommended patterns."""
        patterns = [
            "Repository Pattern",
            "Dependency Injection",
            "Factory Pattern",
        ]
        
        if style == ArchitecturalStyle.MICROSERVICES:
            patterns.extend([
                "Circuit Breaker",
                "API Gateway",
                "Event Sourcing",
                "CQRS",
            ])
        elif style == ArchitecturalStyle.EVENT_DRIVEN:
            patterns.extend([
                "Event Sourcing",
                "Pub/Sub",
                "Message Queue",
            ])
        
        return patterns
    
    def _generate_rationale(self, style: ArchitecturalStyle, paradigm: ProgrammingParadigm, stack: TechnologyStack) -> str:
        """Generate decision rationale."""
        return f"""
Selected {style.value.replace('_', ' ').title()} architecture with {paradigm.value.replace('_', ' ').title()} paradigm.

This choice provides:
- Clear separation of concerns
- Testability and maintainability
- Scalability for expected load
- Alignment with team expertise ({stack.primary_language})

The {stack.primary_language} ecosystem was chosen for its mature tooling,
strong community support, and alignment with project requirements.
""".strip()
    
    def _generate_tradeoffs(self, style: ArchitecturalStyle) -> List[str]:
        """Generate tradeoff analysis."""
        tradeoffs = {
            ArchitecturalStyle.MICROSERVICES: [
                "Pros: Independent scaling, technology diversity, fault isolation",
                "Cons: Operational complexity, network latency, data consistency challenges",
            ],
            ArchitecturalStyle.MONOLITH: [
                "Pros: Simplicity, easier testing, lower operational overhead",
                "Cons: Limited scalability, technology lock-in, deployment coupling",
            ],
            ArchitecturalStyle.SERVERLESS: [
                "Pros: No server management, auto-scaling, pay-per-use",
                "Cons: Cold starts, vendor lock-in, execution limits",
            ],
        }
        return tradeoffs.get(style, ["Standard tradeoffs apply"])


class RulesGenerator:
    """Generate and manage project rules files."""
    
    RULES_FILENAME = ".orchestrator-rules.yml"
    
    def generate_rules(
        self,
        description: str,
        criteria: str,
        project_type: str = ""
    ) -> ProjectRules:
        """Generate complete rules for a project."""
        
        # Analyze architecture
        analyzer = ArchitectureAnalyzer()
        architecture = analyzer.analyze(description, criteria)
        
        # Generate coding standards
        standards = CodingStandard(
            naming_conventions={
                "classes": "PascalCase",
                "functions": "snake_case",
                "constants": "UPPER_SNAKE_CASE",
                "variables": "snake_case",
            },
            code_style=f"{architecture.stack.primary_language}_pep8" if architecture.stack.primary_language == "python" else "standard",
            documentation_required=True,
            type_hints=True,
            max_line_length=100,
            max_complexity=10,
            test_coverage_min=80.0,
        )
        
        # Generate quality gates
        quality_gates = {
            "syntax_check": True,
            "type_check": True,
            "lint_check": True,
            "test_coverage_min": 80.0,
            "max_complexity": 10,
            "security_scan": True,
        }
        
        return ProjectRules(
            version="1.0",
            project_type=project_type or architecture.stack.primary_language,
            architecture=architecture,
            coding_standards=standards,
            allowed_imports=[],
            forbidden_imports=["*deprecated*", "*insecure*"],
            required_patterns=architecture.patterns,
            forbidden_patterns=architecture.anti_patterns,
            quality_gates=quality_gates,
        )
    
    def save_rules(self, rules: ProjectRules, output_dir: Path) -> Path:
        """Save rules to YAML file."""
        rules_file = output_dir / self.RULES_FILENAME
        
        yaml_content = rules.to_yaml()
        
        # Add header comment
        header = f"""# Architecture Rules for Project
# Generated: {rules.created_at}
# Version: {rules.version}
# 
# This file defines the architecture decisions and constraints
# that must be followed throughout the project.
# 
# DO NOT MODIFY MANUALLY - Regenerate using orchestrator

"""
        
        rules_file.write_text(header + yaml_content, encoding='utf-8')
        logger.info(f"Saved architecture rules to: {rules_file}")
        
        return rules_file
    
    def load_rules(self, rules_file: Path) -> Optional[ProjectRules]:
        """Load rules from YAML file."""
        if not rules_file.exists():
            return None
        
        try:
            data = yaml.safe_load(rules_file.read_text(encoding='utf-8'))
            
            # Reconstruct objects
            stack_data = data.get("architecture", {}).get("stack", {})
            stack = TechnologyStack(**stack_data)
            
            arch_data = data.get("architecture", {})
            architecture = ArchitectureDecision(
                style=ArchitecturalStyle(arch_data.get("style", "layered")),
                paradigm=ProgrammingParadigm(arch_data.get("paradigm", "object_oriented")),
                api_style=APIStyle(arch_data.get("api_style", "rest")),
                database_type=DatabaseType(arch_data.get("database_type", "relational")),
                stack=stack,
                constraints=arch_data.get("constraints", []),
                patterns=arch_data.get("patterns", []),
                anti_patterns=arch_data.get("anti_patterns", []),
                rationale=arch_data.get("rationale", ""),
                tradeoffs=arch_data.get("tradeoffs", []),
            )
            
            standards_data = data.get("coding_standards", {})
            coding_standards = CodingStandard(**standards_data)
            
            return ProjectRules(
                version=data.get("version", "1.0"),
                created_at=data.get("created_at", datetime.now().isoformat()),
                project_type=data.get("project_type", ""),
                architecture=architecture,
                coding_standards=coding_standards,
                allowed_imports=data.get("allowed_imports", []),
                forbidden_imports=data.get("forbidden_imports", []),
                required_patterns=data.get("required_patterns", []),
                forbidden_patterns=data.get("forbidden_patterns", []),
                quality_gates=data.get("quality_gates", {}),
            )
            
        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            return None


class ArchitectureRulesEngine:
    """
    Main engine for architecture rules.
    
    Usage:
        engine = ArchitectureRulesEngine(client)
        
        # Generate rules for new project
        rules = await engine.generate_rules(
            description="Build a scalable e-commerce API",
            criteria="Handle 10k requests/sec, 99.9% uptime"
        )
        
        # Save to file
        engine.save_rules(rules, Path("./output"))
        
        # Enforce during development
        engine.enforce_rules(rules, code_file)
    """
    
    def __init__(self, client=None):
        self.analyzer = ArchitectureAnalyzer()
        self.generator = RulesGenerator()
        self.client = client
    
    async def generate_rules(
        self,
        description: str,
        criteria: str,
        project_type: str = ""
    ) -> ProjectRules:
        """Generate architecture rules for a project using LLM if available."""
        logger.info("Generating architecture rules...")
        
        if self.client is not None:
            # Use LLM for architecture decision
            try:
                rules = await self._generate_rules_with_llm(description, criteria, project_type)
                logger.info(f"Architecture (LLM): {rules.architecture.style.value}")
                logger.info(f"Paradigm (LLM): {rules.architecture.paradigm.value}")
                logger.info(f"Stack (LLM): {rules.architecture.stack.primary_language}")
                return rules
            except Exception as e:
                logger.warning(f"LLM architecture generation failed: {e}, falling back to rule-based")
        
        # Step 1: Generate rule-based architecture
        rules = self.generator.generate_rules(description, criteria, project_type)
        
        logger.info(f"Initial Architecture: {rules.architecture.style.value}")
        logger.info(f"Initial Paradigm: {rules.architecture.paradigm.value}")
        logger.info(f"Initial Stack: {rules.architecture.stack.primary_language}")
        
        # Step 2: Ask LLM to optimize (if client available)
        if self.client is not None:
            try:
                optimized_rules = await self._optimize_rules_with_llm(
                    rules, description, criteria
                )
                if optimized_rules:
                    logger.info(f"✨ Architecture optimized by LLM")
                    logger.info(f"Optimized Style: {optimized_rules.architecture.style.value}")
                    return optimized_rules
            except Exception as e:
                logger.warning(f"LLM optimization failed: {e}, using rule-based")
        
        return rules
    
    async def _generate_rules_with_llm(
        self,
        description: str,
        criteria: str,
        project_type: str = ""
    ) -> ProjectRules:
        """Generate architecture rules using LLM."""
        from .models import Model, TaskType
        
        # Build the prompt
        prompt = f"""You are an expert software architect. Analyze this project and recommend the optimal architecture.

PROJECT DESCRIPTION:
{description}

SUCCESS CRITERIA:
{criteria}

Respond with a JSON object in this exact format:
{{
    "style": "one of: microservices, monolith, serverless, layered, hexagonal, event_driven, cqrs, modular_monolith",
    "paradigm": "one of: object_oriented, functional, procedural, declarative, reactive",
    "api_style": "one of: rest, graphql, grpc, websocket, webhook",
    "database_type": "one of: relational, document, key_value, graph, time_series, columnar, none",
    "primary_language": "e.g., python, typescript, go, rust, java",
    "frameworks": ["framework1", "framework2"],
    "libraries": ["library1", "library2"],
    "databases": ["database1"],
    "constraints": ["constraint1", "constraint2"],
    "patterns": ["pattern1", "pattern2"],
    "rationale": "Brief explanation of why this architecture was chosen"
}}

Choose the best options based on the project requirements. Be specific and practical."""

        # Call the LLM
        from .api_clients import UnifiedClient
        
        # Select a capable model for architecture decisions
        model = Model.GPT_4O  # Use a capable model for architecture
        
        resp = await self.client.call(
            model, prompt, system="You are a principal software architect with 20 years of experience.",
            max_tokens=2000, temperature=0.3, timeout=60
        )
        
        # Parse the response (strip markdown if present)
        import json
        import re
        
        response_text = resp.text.strip() if resp.text else ""
        
        # Check for empty response
        if not response_text:
            logger.error("LLM returned empty response for architecture decision")
            raise ValueError("Empty LLM response")
        
        # Remove markdown code fences if present
        if response_text.startswith("```"):
            # Find the end of the opening fence
            lines = response_text.split("\n")
            if len(lines) > 1:
                # Remove opening fence (```json or ```)
                if lines[0].startswith("```"):
                    lines = lines[1:]
                # Remove closing fence (```)
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()
        
        # Try to find JSON in the response (in case there's extra text)
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            response_text = json_match.group(0)
        
        try:
            arch_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM architecture response: {e}")
            logger.error(f"Response text: {resp.text[:500]}...")
            raise
        
        # Build the ArchitectureDecision
        stack = TechnologyStack(
            primary_language=arch_data.get("primary_language", "python"),
            frameworks=arch_data.get("frameworks", []),
            libraries=arch_data.get("libraries", []),
            databases=arch_data.get("databases", []),
        )
        
        architecture = ArchitectureDecision(
            style=ArchitecturalStyle(arch_data.get("style", "layered")),
            paradigm=ProgrammingParadigm(arch_data.get("paradigm", "object_oriented")),
            api_style=APIStyle(arch_data.get("api_style", "rest")),
            database_type=DatabaseType(arch_data.get("database_type", "relational")),
            stack=stack,
            constraints=arch_data.get("constraints", []),
            patterns=arch_data.get("patterns", []),
            rationale=arch_data.get("rationale", ""),
            tradeoffs=[],
        )
        
        # Build coding standards
        standards = CodingStandard(
            naming_conventions={
                "classes": "PascalCase",
                "functions": "snake_case",
                "constants": "UPPER_SNAKE_CASE",
                "variables": "snake_case",
            },
            code_style=f"{stack.primary_language}_standard",
            documentation_required=True,
            type_hints=True,
            max_line_length=100,
            max_complexity=10,
            test_coverage_min=80.0,
        )
        
        # Build quality gates
        quality_gates = {
            "syntax_check": True,
            "type_check": True,
            "lint_check": True,
            "test_coverage_min": 80.0,
            "max_complexity": 10,
            "security_scan": True,
        }
        
        return ProjectRules(
            version="1.0",
            project_type=project_type or stack.primary_language,
            architecture=architecture,
            coding_standards=standards,
            allowed_imports=[],
            forbidden_imports=["*deprecated*", "*insecure*"],
            required_patterns=architecture.patterns,
            forbidden_patterns=[],
            quality_gates=quality_gates,
            _llm_generated=True,
            _llm_optimized=False,
            _llm_model=model.value,
        )
    
    async def _optimize_rules_with_llm(
        self,
        initial_rules: ProjectRules,
        description: str,
        criteria: str
    ) -> Optional[ProjectRules]:
        """
        Ask LLM to review and optimize the rule-based architecture decisions.
        
        Returns optimized rules if improvements suggested, None if no changes needed.
        """
        from .models import Model
        
        arch = initial_rules.architecture
        
        prompt = f"""You are an expert software architect. Review this architecture proposal and suggest optimizations.

PROJECT DESCRIPTION:
{description}

SUCCESS CRITERIA:
{criteria}

CURRENT ARCHITECTURE PROPOSAL:
- Style: {arch.style.value}
- Paradigm: {arch.paradigm.value}
- API: {arch.api_style.value}
- Database: {arch.database_type.value}
- Primary Language: {arch.stack.primary_language}
- Frameworks: {', '.join(arch.stack.frameworks) if arch.stack.frameworks else 'None'}
- Libraries: {', '.join(arch.stack.libraries) if arch.stack.libraries else 'None'}
- Constraints: {', '.join(arch.constraints[:5]) if arch.constraints else 'None'}
- Patterns: {', '.join(arch.patterns[:5]) if arch.patterns else 'None'}

TASK:
1. Review if the current architecture is optimal for this project
2. Consider: scalability, maintainability, team expertise, ecosystem maturity
3. Suggest improvements if any

Respond with a JSON object in this exact format:
{{
    "can_optimize": true or false,
    "reasoning": "Brief explanation of why optimization is or isn't needed",
    "changes": [
        {{
            "field": "style|paradigm|api_style|database_type|primary_language|frameworks|libraries|constraints|patterns",
            "from": "current value",
            "to": "suggested value",
            "reason": "Why this change improves the architecture"
        }}
    ],
    "optimized_architecture": {{
        "style": "one of: microservices, monolith, serverless, layered, hexagonal, event_driven, cqrs, modular_monolith",
        "paradigm": "one of: object_oriented, functional, procedural, declarative, reactive",
        "api_style": "one of: rest, graphql, grpc, websocket, webhook",
        "database_type": "one of: relational, document, key_value, graph, time_series, columnar, none",
        "primary_language": "e.g., python, typescript, go, rust, java",
        "frameworks": ["framework1", "framework2"],
        "libraries": ["library1", "library2"],
        "databases": ["database1"],
        "constraints": ["constraint1", "constraint2"],
        "patterns": ["pattern1", "pattern2"],
        "rationale": "Explanation of the optimized architecture"
    }}
}}}}

If can_optimize is false, set changes to empty array and optimized_architecture to null.
Be conservative - only suggest changes if they provide clear benefits."""

        try:
            # Use a capable model for optimization
            model = Model.GPT_4O
            
            resp = await self.client.call(
                model, prompt, 
                system="You are a principal software architect. Be conservative - only suggest changes if they provide clear architectural benefits.",
                max_tokens=2500, temperature=0.2, timeout=60
            )
            
            # Parse the response (strip markdown if present)
            import json
            import re
            
            response_text = resp.text.strip() if resp.text else ""
            
            # Check for empty response
            if not response_text:
                logger.warning("LLM returned empty response for optimization")
                return None
            
            # Remove markdown code fences if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                if len(lines) > 1:
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    response_text = "\n".join(lines).strip()
            
            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                response_text = json_match.group(0)
            
            try:
                opt_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse optimization response: {e}")
                return None
            
            # Check if optimization is needed
            if not opt_data.get("can_optimize", False):
                logger.info("LLM: No architecture optimizations suggested")
                return None
            
            # Log the changes
            changes = opt_data.get("changes", [])
            if changes:
                logger.info(f"LLM suggested {len(changes)} architecture improvements:")
                for change in changes:
                    logger.info(f"  • {change['field']}: {change['from']} → {change['to']}")
                    logger.info(f"    Reason: {change['reason']}")
            
            # Build optimized architecture
            opt_arch = opt_data.get("optimized_architecture", {})
            
            stack = TechnologyStack(
                primary_language=opt_arch.get("primary_language", arch.stack.primary_language),
                frameworks=opt_arch.get("frameworks", arch.stack.frameworks),
                libraries=opt_arch.get("libraries", arch.stack.libraries),
                databases=opt_arch.get("databases", arch.stack.databases),
            )
            
            optimized_architecture = ArchitectureDecision(
                style=ArchitecturalStyle(opt_arch.get("style", arch.style.value)),
                paradigm=ProgrammingParadigm(opt_arch.get("paradigm", arch.paradigm.value)),
                api_style=APIStyle(opt_arch.get("api_style", arch.api_style.value)),
                database_type=DatabaseType(opt_arch.get("database_type", arch.database_type.value)),
                stack=stack,
                constraints=opt_arch.get("constraints", arch.constraints),
                patterns=opt_arch.get("patterns", arch.patterns),
                rationale=opt_arch.get("rationale", arch.rationale),
                tradeoffs=arch.tradeoffs,
            )
            
            # Update coding standards based on new primary language
            primary_lang = stack.primary_language
            standards = CodingStandard(
                naming_conventions=initial_rules.coding_standards.naming_conventions,
                code_style=f"{primary_lang}_standard",
                documentation_required=initial_rules.coding_standards.documentation_required,
                type_hints=initial_rules.coding_standards.type_hints,
                max_line_length=initial_rules.coding_standards.max_line_length,
                max_complexity=initial_rules.coding_standards.max_complexity,
                test_coverage_min=initial_rules.coding_standards.test_coverage_min,
            )
            
            return ProjectRules(
                version="1.0",
                project_type=initial_rules.project_type,
                architecture=optimized_architecture,
                coding_standards=standards,
                allowed_imports=initial_rules.allowed_imports,
                forbidden_imports=initial_rules.forbidden_imports,
                required_patterns=optimized_architecture.patterns,
                forbidden_patterns=initial_rules.forbidden_patterns,
                quality_gates=initial_rules.quality_gates,
                _llm_generated=False,
                _llm_optimized=True,
                _llm_model=model.value,
            )
            
        except Exception as e:
            logger.warning(f"LLM optimization failed: {e}")
            return None
    
    def save_rules(self, rules: ProjectRules, output_dir: Path) -> Path:
        """Save rules to file."""
        return self.generator.save_rules(rules, output_dir)
    
    def load_rules(self, rules_file: Path) -> Optional[ProjectRules]:
        """Load rules from file."""
        return self.generator.load_rules(rules_file)
    
    def generate_summary(self, rules: ProjectRules) -> str:
        """Generate human-readable summary."""
        arch = rules.architecture
        
        # Determine decision label based on metadata
        llm_generated = getattr(rules, '_llm_generated', False)
        llm_optimized = getattr(rules, '_llm_optimized', False)
        llm_model = getattr(rules, '_llm_model', '')
        
        # Format model name for display
        model_display = f" via {llm_model}" if llm_model else ""
        
        if llm_generated:
            decision_label = f"LLM{model_display} (Generated from scratch)"
        elif llm_optimized:
            decision_label = f"LLM{model_display} (Rule-based → Optimized)"
        else:
            decision_label = "Rule-based"
        
        # Format database display
        db_display = arch.database_type.value.title()
        if arch.database_type == DatabaseType.NONE:
            db_display = "None (No Database)"
        
        lines = [
            "🏗️ ARCHITECTURE DECISION",
            "=" * 60,
            "",
            f"Decided by: {decision_label}",
            "",
            f"Style: {arch.style.value.replace('_', ' ').title()}",
            f"Paradigm: {arch.paradigm.value.replace('_', ' ').title()}",
            f"API: {arch.api_style.value.upper()}",
            f"Database: {db_display}",
            "",
            "Technology Stack:",
            f"  Primary: {arch.stack.primary_language}",
        ]
        
        if arch.stack.frameworks:
            lines.append(f"  Frameworks: {', '.join(arch.stack.frameworks)}")
        
        if arch.stack.databases:
            lines.append(f"  Databases: {', '.join(arch.stack.databases)}")
        
        lines.extend([
            "",
            "Key Constraints:",
        ])
        
        for constraint in arch.constraints[:3]:
            lines.append(f"  • {constraint}")
        
        lines.extend([
            "",
            "Recommended Patterns:",
        ])
        
        for pattern in arch.patterns[:3]:
            lines.append(f"  • {pattern}")
        
        lines.extend([
            "",
            f"Rules file: .orchestrator-rules.yml",
            "=" * 60,
        ])
        
        return "\n".join(lines)


# Convenience function
async def create_architecture_rules(
    description: str,
    criteria: str,
    output_dir: Path,
    project_type: str = "",
    client = None
) -> ProjectRules:
    """
    Quick function to create and save architecture rules.
    
    Usage:
        rules = await create_architecture_rules(
            description="Build a REST API",
            criteria="High performance",
            output_dir=Path("./output"),
            client=client  # Optional: for LLM-based architecture decisions
        )
    """
    engine = ArchitectureRulesEngine(client=client)
    
    rules = await engine.generate_rules(description, criteria, project_type)
    engine.save_rules(rules, output_dir)
    
    print(engine.generate_summary(rules))
    
    return rules

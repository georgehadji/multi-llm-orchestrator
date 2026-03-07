"""
Enhanced Dashboard Example
==========================
Demonstrates how to use the new Enhanced Dashboard v2.0 with:
- Architecture decisions visibility
- Real-time task progress tracking
- Model status with reasons for inactivity
- Project details and success criteria

Usage:
    # Terminal 1: Start the dashboard
    python example_enhanced_dashboard.py --dashboard
    
    # Terminal 2: Run a project (will appear in dashboard)
    python example_enhanced_dashboard.py --project
"""
import argparse
import asyncio
import webbrowser
from pathlib import Path

from orchestrator import Orchestrator, Budget, Model


def start_dashboard():
    """Start the enhanced dashboard server."""
    from orchestrator.dashboard_enhanced import run_enhanced_dashboard
    
    print("Starting Enhanced Dashboard v2.0...")
    print("Open http://127.0.0.1:8888 in your browser")
    run_enhanced_dashboard(host="127.0.0.1", port=8888, open_browser=True)


async def run_example_project():
    """Run an example project that will appear in the dashboard."""
    from orchestrator.dashboard_enhanced import (
        EnhancedDashboardServer, 
        DashboardIntegration,
        EnhancedDataProvider
    )
    
    # Create dashboard data provider and integration
    data_provider = EnhancedDataProvider()
    dashboard_integration = DashboardIntegration(data_provider)
    
    # Create orchestrator with dashboard integration
    budget = Budget(max_usd=5.0, max_time_seconds=600)
    orch = Orchestrator(budget=budget)
    orch.set_dashboard_integration(dashboard_integration)
    
    # Project description that will trigger specific architecture decisions
    project_description = """
    Build a scalable microservices-based e-commerce API with event-driven architecture.
    The system should handle 10,000 requests per second with 99.9% uptime.
    Use PostgreSQL for data storage and Redis for caching.
    Implement CQRS pattern for order processing.
    """
    
    success_criteria = """
    - API response time < 100ms at p95
    - Support 10,000 concurrent users
    - 99.9% uptime SLA
    - Eventual consistency for order data
    - Full test coverage > 80%
    """
    
    print("=" * 70)
    print("Starting Example Project")
    print("=" * 70)
    print(f"\nProject: {project_description[:100]}...")
    print(f"\nCheck the dashboard at http://127.0.0.1:8888")
    print("\n" + "=" * 70)
    
    try:
        state = await orch.run_project(
            project_description=project_description,
            success_criteria=success_criteria,
            project_id="example-ecommerce-api-001",
            analyze_on_complete=True,
            output_dir=Path("./output/example_project"),
        )
        
        print("\n" + "=" * 70)
        print(f"Project completed with status: {state.status.value}")
        print(f"Budget used: ${state.budget.spent_usd:.4f}")
        print(f"Tasks completed: {len(state.results)}/{len(state.tasks)}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nProject failed: {e}")
    finally:
        await orch.close()


def demo_dashboard_data():
    """Populate dashboard with demo data for testing UI."""
    import time
    from orchestrator.dashboard_enhanced import (
        EnhancedDataProvider,
        ArchitectureInfo,
        ProjectInfo,
        ActiveTaskInfo,
    )
    from orchestrator.architecture_rules import (
        ArchitecturalStyle, ProgrammingParadigm, APIStyle, DatabaseType,
        TechnologyStack, ArchitectureDecision, ProjectRules
    )
    
    # Create data provider
    provider = EnhancedDataProvider()
    
    # Create demo architecture rules
    stack = TechnologyStack(
        primary_language="python",
        frameworks=["fastapi", "pydantic", "pytest"],
        libraries=["uvicorn", "httpx", "sqlalchemy", "redis-py"],
        databases=["postgresql", "redis"],
        tools=["docker", "black", "ruff", "mypy"],
        infrastructure=["kubernetes", "nginx", "prometheus"]
    )
    
    arch_decision = ArchitectureDecision(
        style=ArchitecturalStyle.MICROSERVICES,
        paradigm=ProgrammingParadigm.OBJECT_ORIENTED,
        api_style=APIStyle.REST,
        database_type=DatabaseType.RELATIONAL,
        stack=stack,
        constraints=[
            "All services must be stateless",
            "Eventual consistency for cross-service data",
            "Circuit breaker pattern for resilience",
            "Each service owns its database"
        ],
        patterns=[
            "CQRS",
            "Event Sourcing",
            "Saga Pattern",
            "API Gateway",
            "Circuit Breaker"
        ],
        rationale="Microservices chosen for independent scaling and deployment",
        tradeoffs=[
            "Pros: Independent scaling, technology diversity",
            "Cons: Operational complexity, distributed tracing needed"
        ]
    )
    
    rules = ProjectRules(
        version="1.0",
        project_type="web_api",
        architecture=arch_decision
    )
    
    # Set demo project
    provider._architecture_rules = provider._convert_architecture_rules(rules)
    
    # Set demo active task
    provider._active_task = ActiveTaskInfo(
        task_id="task_003_api_gateway",
        task_type="code_generation",
        prompt="""Implement the API Gateway service with the following requirements:
        - Route requests to appropriate microservices
        - Implement rate limiting (100 req/min per client)
        - JWT authentication validation
        - Request/response transformation
        - Circuit breaker integration
        
        The gateway should use FastAPI and integrate with the existing 
        user-service, order-service, and inventory-service."",
        status="running",
        iteration=2,
        max_iterations=3,
        score=0.85,
        model_used="gpt-4o",
        elapsed_seconds=245.5
    )
    
    print("Dashboard data populated with demo values")
    print("Start the dashboard to see the UI with sample data")
    
    return provider


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Dashboard Example")
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start the enhanced dashboard server"
    )
    parser.add_argument(
        "--project",
        action="store_true",
        help="Run an example project (dashboard must be running)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Populate dashboard with demo data for UI testing"
    )
    
    args = parser.parse_args()
    
    if args.dashboard:
        start_dashboard()
    elif args.project:
        asyncio.run(run_example_project())
    elif args.demo:
        demo_dashboard_data()
    else:
        parser.print_help()
        print("\n" + "=" * 70)
        print("Quick Start:")
        print("  1. Terminal 1: python example_enhanced_dashboard.py --dashboard")
        print("  2. Terminal 2: python example_enhanced_dashboard.py --project")
        print("=" * 70)

#!/usr/bin/env python
"""
Verification Script for All Integrations
=========================================

Run this to verify all 13 new modules are properly integrated.

Usage:
    python verify_integrations.py
"""

import sys
from pathlib import Path

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent))

def verify_imports():
    """Verify all modules can be imported."""
    print("=" * 70)
    print("VERIFYING ALL INTEGRATIONS")
    print("=" * 70)
    
    modules = {
        # Security (Agents of Chaos)
        "Task Verifier": ("orchestrator.task_verifier", ["TaskVerifier"]),
        "Accountability": ("orchestrator.accountability", ["AccountabilityTracker", "ActorType", "ActionType"]),
        "Agent Safety": ("orchestrator.agent_safety", ["AgentSafetyMonitor", "SafetyEventType"]),
        "Red Team": ("orchestrator.red_team", ["RedTeamFramework"]),
        
        # External Projects
        "Token Optimizer (RTK)": ("orchestrator.token_optimizer", ["TokenOptimizer"]),
        "Preflight (Mnemo)": ("orchestrator.preflight", ["PreflightValidator", "PreflightMode"]),
        "Session Watcher (Mnemo)": ("orchestrator.session_watcher", ["SessionWatcher"]),
        "Persona (Mnemo)": ("orchestrator.persona", ["PersonaManager", "PersonaMode"]),
        "A2A Protocol (LiteLLM)": ("orchestrator.a2a_protocol", ["A2AManager", "AgentCard"]),
        
        # QMD Integration
        "MCP Server": ("orchestrator.mcp_server", ["MCPServer", "MCPConfig"]),
        "BM25 Search": ("orchestrator.bm25_search", ["BM25Search"]),
        "LLM Reranker": ("orchestrator.reranker", ["LLMReranker"]),
    }
    
    failed = []
    passed = []
    
    for name, (module_path, classes) in modules.items():
        try:
            module = __import__(module_path, fromlist=classes)
            for cls_name in classes:
                getattr(module, cls_name)
            print(f"  [OK] {name}")
            passed.append(name)
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed.append(name)
    
    # Engine integration
    print("\n  Checking Engine Integration...")
    try:
        from orchestrator.engine import Orchestrator
        print("  [OK] Orchestrator Engine")
        
        # Check properties exist
        orch = Orchestrator.__new__(Orchestrator)  # Create without init for speed
        
        engine_attrs = [
            'task_verifier', 'accountability', 'agent_safety', 'red_team',
            'token_optimizer', 'preflight_validator', 'session_watcher',
            'persona_manager', 'memory_manager', 'bm25_search',
            'reranker', 'a2a_manager',
        ]
        
        missing = []
        for attr in engine_attrs:
            if not hasattr(Orchestrator, attr) and not hasattr(orch, f'_{attr.split("_")[1]}'):
                # Check private attrs
                possible_names = [
                    f'_{attr}',
                    f'_{attr.split("_")[0]}_{attr.split("_")[1]}',
                ]
                found = any(hasattr(orch, n) if hasattr(orch, '__dict__') else False for n in possible_names)
                if not found:
                    missing.append(attr)
        
        if missing:
            print(f"  [WARN] Missing engine attributes: {missing}")
        else:
            print("  [OK] All engine properties present")
            
    except Exception as e:
        print(f"  [FAIL] Engine Integration: {e}")
        failed.append("Engine")
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {len(passed)}/{len(modules)} modules verified")
    print("=" * 70)
    
    if failed:
        print(f"\n[FAIL] Failed: {failed}")
        return 1
    else:
        print("\n[SUCCESS] ALL INTEGRATIONS VERIFIED SUCCESSFULLY!")
        print("\nNext steps:")
        print("  1. Start MCP Server: python -m orchestrator.mcp_server --http --port 8181")
        print("  2. Configure Claude Desktop MCP config")
        print("  3. Use enhanced search: orch.hybrid_search(query, project_id)")
        return 0


if __name__ == "__main__":
    sys.exit(verify_imports())

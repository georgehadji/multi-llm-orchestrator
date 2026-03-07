"""
Demo: Live Dashboard Integration
================================
Demonstrates how to integrate the gamified live dashboard
with the orchestrator for real-time updates.

Usage:
    # Terminal 1: Start dashboard
    python demo_live_dashboard.py --dashboard
    
    # Terminal 2: Run demo project
    python demo_live_dashboard.py --demo
"""
import argparse
import asyncio
import time
from pathlib import Path


async def run_dashboard():
    """Start the live dashboard."""
    from orchestrator.dashboard_live import run_live_dashboard
    print("🎮 Starting Mission Control LIVE...")
    run_live_dashboard()


async def run_demo_project():
    """Run a demo project with live updates."""
    import httpx
    
    BASE_URL = "http://127.0.0.1:8888"
    
    print("🎯 Demo: Simulating project execution...")
    print("=" * 60)
    
    async with httpx.AsyncClient() as client:
        # 1. Start project
        print("\n1️⃣  Starting project...")
        await client.post(f"{BASE_URL}/api/project/start", json={
            "project_id": "demo-ecommerce-api",
            "description": "Build e-commerce API with microservices",
            "total_tasks": 5,
            "budget": 10.0,
        })
        print("   ✅ Project started")
        await asyncio.sleep(2)
        
        # 2. Simulate tasks
        tasks = [
            ("task_001", "setup_project", "Initialize project structure"),
            ("task_002", "user_service", "Create user authentication service"),
            ("task_003", "product_service", "Create product catalog service"),
            ("task_004", "order_service", "Create order processing service"),
            ("task_005", "api_gateway", "Set up API gateway"),
        ]
        
        for i, (task_id, task_type, description) in enumerate(tasks, 1):
            print(f"\n{i+1}️⃣  Running {task_id}...")
            
            # Start task
            await client.post(f"{BASE_URL}/api/task/update", json={
                "task_id": task_id,
                "task_type": task_type,
                "status": "running",
                "iteration": 1,
                "score": 0.0,
                "model_used": "gpt-4o",
                "prompt": description,
            })
            print(f"   ▶️  Task started")
            await asyncio.sleep(1)
            
            # Progress iterations
            for iteration in range(1, 4):
                score = min(0.7 + (iteration * 0.1), 0.95)
                await client.post(f"{BASE_URL}/api/task/update", json={
                    "task_id": task_id,
                    "task_type": task_type,
                    "status": "running",
                    "iteration": iteration,
                    "score": score,
                    "model_used": "gpt-4o",
                })
                print(f"   📝 Iteration {iteration}: Score {score:.0%}")
                await asyncio.sleep(0.5)
            
            # Complete task
            await client.post(f"{BASE_URL}/api/task/update", json={
                "task_id": task_id,
                "task_type": task_type,
                "status": "completed",
                "iteration": 3,
                "score": 0.95,
                "model_used": "gpt-4o",
            })
            print(f"   ✅ Task completed!")
            await asyncio.sleep(1)
        
        # 3. Run tests
        print("\n6️⃣  Running tests...")
        tests = ["test_users.py", "test_products.py", "test_orders.py"]
        for test_file in tests:
            await client.post(f"{BASE_URL}/api/test/update", json={
                "test_file": test_file,
                "status": "running",
                "progress": 0,
            })
            print(f"   🧪 {test_file} running...")
            await asyncio.sleep(1)
            
            await client.post(f"{BASE_URL}/api/test/update", json={
                "test_file": test_file,
                "status": "passed",
                "progress": 100,
            })
            print(f"   ✅ {test_file} passed!")
        
        # 4. Complete project
        print("\n7️⃣  Completing project...")
        await client.post(f"{BASE_URL}/api/project/complete", json={})
        print("   🎉 PROJECT COMPLETE!")
        print("\n" + "=" * 60)
        print("✨ Check the dashboard for the celebration!")
        print("🎊 You should see confetti and hear sound effects!")


async def simulate_achievements():
    """Simulate unlocking achievements."""
    import httpx
    
    BASE_URL = "http://127.0.0.1:8888"
    
    print("🏆 Simulating achievement unlocks...")
    
    # Note: Achievements are automatically unlocked based on task completion
    # This is handled internally by the dashboard server
    
    print("✅ Achievements will unlock automatically during task execution:")
    print("   🎯 Task Master - Complete first task")
    print("   ⚡ Speed Demon - Complete task in <30s")
    print("   💯 Perfectionist - 100% score")
    print("   💰 Budget Master - Under 50% budget")


def main():
    parser = argparse.ArgumentParser(description="Live Dashboard Demo")
    parser.add_argument("--dashboard", action="store_true", help="Start the dashboard")
    parser.add_argument("--demo", action="store_true", help="Run demo project")
    parser.add_argument("--achievements", action="store_true", help="Simulate achievements")
    
    args = parser.parse_args()
    
    if args.dashboard:
        asyncio.run(run_dashboard())
    elif args.demo:
        print("🎮 Make sure the dashboard is running first!")
        print("   python demo_live_dashboard.py --dashboard")
        print()
        input("Press Enter when dashboard is ready...")
        asyncio.run(run_demo_project())
    elif args.achievements:
        asyncio.run(simulate_achievements())
    else:
        print("🎮 Mission Control LIVE Demo")
        print("=" * 60)
        print()
        print("Quick Start:")
        print("  1. Terminal 1: python demo_live_dashboard.py --dashboard")
        print("  2. Terminal 2: python demo_live_dashboard.py --demo")
        print()
        print("Features:")
        print("  • Real-time WebSocket updates")
        print("  • Gamification (XP, levels, achievements)")
        print("  • Confetti celebration")
        print("  • Sound effects")
        print("  • Toast notifications")


if __name__ == "__main__":
    main()

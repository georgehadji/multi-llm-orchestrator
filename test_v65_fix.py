"""Quick test to verify v6.5 hook fix."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator.dashboard_mission_control import MissionControlServer, ActiveProject
from orchestrator.models import Budget
from orchestrator.engine import Orchestrator
from orchestrator.hooks import EventType

# Test that Orchestrator has add_hook method
print("Testing Orchestrator.add_hook()...")
orchestrator = Orchestrator(budget=Budget(max_usd=1.0))

# Test that add_hook exists
assert hasattr(orchestrator, 'add_hook'), "Orchestrator should have add_hook method"
print("✅ Orchestrator.add_hook() exists")

# Test that we can add a hook
def test_callback(**kwargs):
    pass

try:
    orchestrator.add_hook(EventType.TASK_STARTED, test_callback)
    print("✅ Can add hooks using add_hook()")
except Exception as e:
    print(f"❌ Failed to add hook: {e}")
    sys.exit(1)

# Test MissionControlServer initialization
print("\nTesting MissionControlServer...")
try:
    server = MissionControlServer(host="127.0.0.1", port=8889)
    print("✅ MissionControlServer initialized")
except Exception as e:
    print(f"❌ Failed to initialize server: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("✅ All tests passed! v6.5 hook fix is working.")
print("="*50)

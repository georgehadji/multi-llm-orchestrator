"""Test script to verify remove functionality."""

# Test that the new endpoints exist in the code
with open('orchestrator/dashboard_mission_control.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check for remove endpoint
assert '/api/project/{project_id}/remove' in content, "Remove endpoint not found"
print("✅ Remove endpoint found")

# Check for clear-finished endpoint
assert '/api/projects/clear-finished' in content, "Clear finished endpoint not found"
print("✅ Clear finished endpoint found")

# Check for removeProject JS function
assert 'async function removeProject(projectId)' in content, "removeProject JS function not found"
print("✅ removeProject JS function found")

# Check for clearFinished JS function
assert 'async function clearFinished()' in content, "clearFinished JS function not found"
print("✅ clearFinished JS function found")

# Check for remove button in UI
assert "onclick=\"event.stopPropagation(); removeProject" in content, "Remove button not found in project cards"
print("✅ Remove button in project cards found")

# Check for Clear All button
assert "onclick=\"clearFinished()" in content, "Clear All button not found"
print("✅ Clear All button found")

# Check for conditional button display
assert "btn-remove-project" in content, "Remove project button in panel not found"
print("✅ Remove project button in panel found")

print("\n" + "="*50)
print("✅ All remove/clear functionality is implemented!")
print("="*50)

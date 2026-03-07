"""Test dashboard HTML generation"""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

from orchestrator.dashboard_live import LiveDashboardServer

server = LiveDashboardServer()
html = server._get_html()

print(f"HTML length: {len(html)} characters")
print(f"HTML starts with: {html[:100]}...")
print(f"\n✅ HTML generated successfully!")
print(f"\nCheck if HTML contains key elements:")
print(f"  - React: {'react' in html.lower()}")
print(f"  - WebSocket: {'websocket' in html.lower()}")
print(f"  - Ant Design: {'antd' in html.lower()}")

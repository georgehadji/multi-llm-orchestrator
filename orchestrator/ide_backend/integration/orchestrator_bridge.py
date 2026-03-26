"""
Orchestrator Bridge - Connects IDE to AI Orchestrator core
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from ..log_config import get_logger
from ..websocket_manager import ConnectionManager

logger = get_logger(__name__)


class OrchestratorBridge:
    """Bridges AI Orchestrator events to IDE WebSocket."""
    
    def __init__(self, orchestrator: Any, connection_manager: ConnectionManager):
        self.orchestrator = orchestrator
        self.connection_manager = connection_manager
        self._session_mapping: Dict[str, str] = {}  # orchestrator_project_id -> ide_session_id
        
        # Subscribe to orchestrator events
        self._subscribe_to_events()
    
    def _subscribe_to_events(self):
        """Subscribe to orchestrator unified events."""
        try:
            from ..unified_events import (
                EventType,
                TaskStartedEvent,
                TaskCompletedEvent,
                TaskProgressEvent,
                CapabilityUsedEvent,
                CapabilityCompletedEvent,
                BudgetWarningEvent,
                ModelSelectedEvent,
            )
            
            # Try to get event bus
            event_bus = getattr(self.orchestrator, 'event_bus', None)
            if event_bus and hasattr(event_bus, 'subscribe'):
                event_bus.subscribe(self._on_event)
                logger.info("Subscribed to orchestrator events")
            else:
                logger.warning("Orchestrator event bus not available")
                
        except ImportError as e:
            logger.warning(f"Could not import unified events: {e}")
    
    def _on_event(self, event: Any):
        """Handle orchestrator event."""
        try:
            event_type = getattr(event, 'event_type', None) or getattr(event, 'type', None)
            event_data = self._convert_event(event)
            
            if event_type:
                # Broadcast to all sessions or specific session
                asyncio.create_task(
                    self.connection_manager.broadcast(str(event_type), event_data)
                )
        except Exception as e:
            logger.error(f"Error handling orchestrator event: {e}")
    
    def _convert_event(self, event: Any) -> Dict[str, Any]:
        """Convert orchestrator event to IDE format."""
        event_dict = {}
        
        # Extract common fields
        for attr in ['task_id', 'project_id', 'model', 'cost', 'quality', 'status']:
            if hasattr(event, attr):
                event_dict[attr] = getattr(event, attr)
        
        # Convert to snake_case for frontend
        return event_dict
    
    def map_session(self, orchestrator_project_id: str, ide_session_id: str):
        """Map orchestrator project ID to IDE session ID."""
        self._session_mapping[orchestrator_project_id] = ide_session_id
        logger.info(f"Mapped session: {orchestrator_project_id} -> {ide_session_id}")
    
    async def send_to_project(self, project_id: str, event: str, data: Dict[str, Any]):
        """Send event to specific project session."""
        ide_session_id = self._session_mapping.get(project_id)
        if ide_session_id:
            await self.connection_manager.send_to_session(ide_session_id, event, data)
        else:
            logger.warning(f"No IDE session mapped for project {project_id}")
    
    async def execute_task(self, session_id: str, task_spec: Dict[str, Any]):
        """Execute a task through the orchestrator."""
        try:
            # Get orchestrator method
            if hasattr(self.orchestrator, 'execute'):
                result = await self.orchestrator.execute(task_spec)
                return result
            elif hasattr(self.orchestrator, 'run'):
                result = await self.orchestrator.run(task_spec)
                return result
            else:
                logger.error("Orchestrator has no execute or run method")
                return None
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return None
    
    async def chat(self, session_id: str, message: str, context: Optional[Dict[str, Any]] = None):
        """Send chat message to orchestrator."""
        try:
            # This would integrate with the orchestrator's chat/reasoning capabilities
            # For now, it's a placeholder
            logger.info(f"Chat message for session {session_id}: {message[:100]}")
            return {"response": "Message received", "status": "processing"}
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {"error": str(e)}


# Global bridge instance
_bridge: Optional[OrchestratorBridge] = None


def get_orchestrator_bridge(
    orchestrator: Any,
    connection_manager: ConnectionManager,
) -> OrchestratorBridge:
    """Get or create the orchestrator bridge."""
    global _bridge
    if _bridge is None:
        _bridge = OrchestratorBridge(orchestrator, connection_manager)
    return _bridge


def reset_orchestrator_bridge():
    """Reset the bridge (for testing)."""
    global _bridge
    _bridge = None

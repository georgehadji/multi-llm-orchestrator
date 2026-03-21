"""
Triggers — Event-driven triggers
===============================
Module for managing event-driven triggers that can initiate actions based on conditions.

Pattern: Observer
Async: Yes — for I/O-bound trigger evaluations
Layer: L5 Events

Usage:
    from orchestrator.triggers import TriggerManager
    trigger_manager = TriggerManager()
    trigger = trigger_manager.create_trigger(
        name="High Error Rate",
        condition="error_rate > 0.05",
        action="alert_team",
        context={"team": "engineering"}
    )
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("orchestrator.triggers")


class TriggerConditionOperator(Enum):
    """Operators for trigger conditions."""
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_EQUAL = "ge"
    LESS_EQUAL = "le"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"


@dataclass
class Trigger:
    """Represents a single trigger."""
    
    id: str
    name: str
    condition: str  # Expression to evaluate
    action: str  # Action to take when condition is met
    context: Dict[str, Any]  # Context for the action
    enabled: bool = True
    last_evaluated: Optional[float] = None  # Timestamp of last evaluation
    last_triggered: Optional[float] = None  # Timestamp of last trigger
    cooldown_period: float = 300.0  # Cooldown period in seconds (default 5 minutes)
    eval_frequency: float = 60.0  # How often to evaluate in seconds (default 1 minute)


class TriggerManager:
    """Manages event-driven triggers that can initiate actions based on conditions."""

    def __init__(self):
        """Initialize the trigger manager."""
        self.triggers: Dict[str, Trigger] = {}
        self.eval_queue: List[str] = []  # Queue of trigger IDs to evaluate
        self.running = False
        self.eval_task: Optional[asyncio.Task] = None
        self.custom_conditions: Dict[str, Callable] = {}
        self.custom_actions: Dict[str, Callable] = {}
    
    def create_trigger(self, name: str, condition: str, action: str, 
                      context: Dict[str, Any], cooldown_period: float = 300.0,
                      eval_frequency: float = 60.0) -> Trigger:
        """
        Create a new trigger.
        
        Args:
            name: Name of the trigger
            condition: Expression to evaluate
            action: Action to take when condition is met
            context: Context for the action
            cooldown_period: Cooldown period in seconds
            eval_frequency: How often to evaluate in seconds
            
        Returns:
            Trigger: The created trigger
        """
        import time
        
        trigger_id = f"trigger_{len(self.triggers)}"
        trigger = Trigger(
            id=trigger_id,
            name=name,
            condition=condition,
            action=action,
            context=context,
            cooldown_period=cooldown_period,
            eval_frequency=eval_frequency
        )
        
        self.triggers[trigger_id] = trigger
        self.eval_queue.append(trigger_id)
        
        logger.info(f"Created trigger: {name} (ID: {trigger_id})")
        return trigger
    
    def remove_trigger(self, trigger_id: str) -> bool:
        """
        Remove a trigger.
        
        Args:
            trigger_id: ID of the trigger to remove
            
        Returns:
            bool: True if removed successfully, False otherwise
        """
        if trigger_id in self.triggers:
            del self.triggers[trigger_id]
            if trigger_id in self.eval_queue:
                self.eval_queue.remove(trigger_id)
            logger.info(f"Removed trigger: {trigger_id}")
            return True
        return False
    
    def enable_trigger(self, trigger_id: str) -> bool:
        """Enable a trigger."""
        if trigger_id in self.triggers:
            self.triggers[trigger_id].enabled = True
            logger.info(f"Enabled trigger: {trigger_id}")
            return True
        return False
    
    def disable_trigger(self, trigger_id: str) -> bool:
        """Disable a trigger."""
        if trigger_id in self.triggers:
            self.triggers[trigger_id].enabled = False
            logger.info(f"Disabled trigger: {trigger_id}")
            return True
        return False
    
    def register_custom_condition(self, name: str, func: Callable) -> bool:
        """
        Register a custom condition function.
        
        Args:
            name: Name of the condition
            func: Function that takes context and returns bool
            
        Returns:
            bool: True if registered successfully
        """
        self.custom_conditions[name] = func
        logger.info(f"Registered custom condition: {name}")
        return True
    
    def register_custom_action(self, name: str, func: Callable) -> bool:
        """
        Register a custom action function.
        
        Args:
            name: Name of the action
            func: Function that takes context and performs an action
            
        Returns:
            bool: True if registered successfully
        """
        self.custom_actions[name] = func
        logger.info(f"Registered custom action: {name}")
        return True
    
    async def evaluate_trigger(self, trigger_id: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate a single trigger with the provided context.
        
        Args:
            trigger_id: ID of the trigger to evaluate
            context: Context to evaluate the trigger against
            
        Returns:
            bool: True if the trigger fired, False otherwise
        """
        import time
        
        if trigger_id not in self.triggers:
            logger.error(f"Trigger {trigger_id} does not exist")
            return False
        
        trigger = self.triggers[trigger_id]
        
        if not trigger.enabled:
            return False
        
        # Check cooldown period
        current_time = time.time()
        if (trigger.last_triggered and 
            current_time - trigger.last_triggered < trigger.cooldown_period):
            return False
        
        # Evaluate the condition
        try:
            condition_met = await self._evaluate_condition(trigger.condition, context)
        except Exception as e:
            logger.error(f"Error evaluating condition for trigger {trigger_id}: {e}")
            return False
        
        if condition_met:
            # Execute the action
            try:
                await self._execute_action(trigger.action, {**trigger.context, **context})
                trigger.last_triggered = current_time
                logger.info(f"Trigger fired: {trigger.name} (ID: {trigger_id})")
                return True
            except Exception as e:
                logger.error(f"Error executing action for trigger {trigger_id}: {e}")
                return False
        
        # Update last evaluated time
        trigger.last_evaluated = current_time
        return False
    
    async def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition expression against the context."""
        import ast
        import operator
        
        # For security, only allow safe operations
        ops = {
            ast.Gt: operator.gt,
            ast.Lt: operator.lt,
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.GtE: operator.ge,
            ast.LtE: operator.le
        }
        
        try:
            # Parse the condition expression
            tree = ast.parse(condition, mode='eval')
            return self._eval_expr(tree.body, context, ops)
        except SyntaxError:
            # If it's not a valid expression, check for custom conditions
            if condition in self.custom_conditions:
                return await self.custom_conditions[condition](context)
            else:
                logger.error(f"Invalid condition expression: {condition}")
                return False
    
    def _eval_expr(self, node, context: Dict[str, Any], ops: Dict):
        """Recursively evaluate an AST expression."""
        import ast
        
        if isinstance(node, ast.Constant):  # Numbers, strings, booleans
            return node.value
        elif isinstance(node, ast.Name):  # Variables
            return context.get(node.id, 0)
        elif isinstance(node, ast.BinOp):  # Binary operations (e.g., x > y)
            left = self._eval_expr(node.left, context, ops)
            right = self._eval_expr(node.right, context, ops)
            return ops[type(node.op)](left, right)
        elif isinstance(node, ast.BoolOp):  # Boolean operations (and, or)
            if isinstance(node.op, ast.And):
                return all(self._eval_expr(value, context, ops) for value in node.values)
            elif isinstance(node.op, ast.Or):
                return any(self._eval_expr(value, context, ops) for value in node.values)
        elif isinstance(node, ast.Compare):  # Comparisons (e.g., x > y < z)
            left = self._eval_expr(node.left, context, ops)
            result = True
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_expr(comparator, context, ops)
                result = result and ops[type(op)](left, right)
                left = right
            return result
        else:
            raise TypeError(node)
    
    async def _execute_action(self, action: str, context: Dict[str, Any]):
        """Execute an action with the provided context."""
        if action in self.custom_actions:
            await self.custom_actions[action](context)
        else:
            # Default actions
            if action == "log":
                logger.info(f"Trigger action: {context}")
            elif action == "alert":
                logger.warning(f"Alert triggered: {context}")
            else:
                logger.warning(f"Unknown action: {action}, context: {context}")
    
    async def evaluate_all_triggers(self, context: Dict[str, Any] = None) -> Dict[str, bool]:
        """
        Evaluate all triggers with the provided context.
        
        Args:
            context: Context to evaluate all triggers against
            
        Returns:
            Dict mapping trigger IDs to whether they fired
        """
        if context is None:
            context = {}
        
        results = {}
        
        for trigger_id in list(self.triggers.keys()):  # Copy to avoid modification during iteration
            fired = await self.evaluate_trigger(trigger_id, context)
            results[trigger_id] = fired
        
        return results
    
    def start_evaluation_loop(self):
        """Start the background evaluation loop."""
        if not self.running:
            self.running = True
            self.eval_task = asyncio.create_task(self._evaluation_loop())
            logger.info("Started trigger evaluation loop")
    
    def stop_evaluation_loop(self):
        """Stop the background evaluation loop."""
        if self.running and self.eval_task:
            self.running = False
            self.eval_task.cancel()
            logger.info("Stopped trigger evaluation loop")
    
    async def _evaluation_loop(self):
        """Background loop to periodically evaluate triggers."""
        import time
        
        while self.running:
            try:
                # Evaluate triggers based on their frequency
                current_time = time.time()
                
                for trigger_id, trigger in self.triggers.items():
                    if (trigger.enabled and 
                        (trigger.last_evaluated is None or 
                         current_time - trigger.last_evaluated >= trigger.eval_frequency)):
                        # Evaluate this trigger with empty context (should come from system)
                        await self.evaluate_trigger(trigger_id, {})
                
                # Sleep for a short time to prevent busy waiting
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                logger.info("Trigger evaluation loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in trigger evaluation loop: {e}")
                await asyncio.sleep(5.0)  # Wait before retrying
    
    def get_trigger_stats(self) -> Dict[str, Any]:
        """
        Get statistics about triggers.
        
        Returns:
            Dict with trigger statistics
        """
        import time
        
        current_time = time.time()
        active_triggers = sum(1 for t in self.triggers.values() if t.enabled)
        total_firings = 0
        avg_evaluation_freq = 0.0
        
        if self.triggers:
            avg_evaluation_freq = sum(t.eval_frequency for t in self.triggers.values()) / len(self.triggers)
        
        # Count total firings by looking at last_triggered times
        for trigger in self.triggers.values():
            if trigger.last_triggered:
                total_firings += 1
        
        return {
            "total_triggers": len(self.triggers),
            "active_triggers": active_triggers,
            "disabled_triggers": len(self.triggers) - active_triggers,
            "total_firings": total_firings,
            "average_evaluation_frequency": avg_evaluation_freq,
            "running": self.running
        }
    
    async def trigger_manual_evaluation(self, trigger_id: str, context: Dict[str, Any]) -> bool:
        """
        Manually evaluate a specific trigger with provided context.
        
        Args:
            trigger_id: ID of the trigger to evaluate
            context: Context to evaluate the trigger against
            
        Returns:
            bool: True if the trigger fired, False otherwise
        """
        return await self.evaluate_trigger(trigger_id, context)
    
    def update_trigger(self, trigger_id: str, **updates) -> bool:
        """
        Update properties of an existing trigger.
        
        Args:
            trigger_id: ID of the trigger to update
            **updates: Properties to update
            
        Returns:
            bool: True if updated successfully, False otherwise
        """
        if trigger_id not in self.triggers:
            return False
        
        trigger = self.triggers[trigger_id]
        
        # Update allowed properties
        updatable_fields = {
            'name', 'condition', 'action', 'context', 'enabled',
            'cooldown_period', 'eval_frequency'
        }
        
        for field, value in updates.items():
            if field in updatable_fields:
                setattr(trigger, field, value)
        
        logger.info(f"Updated trigger: {trigger_id}")
        return True
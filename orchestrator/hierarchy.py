"""
Hierarchy — Multi-level org/team hierarchy
=========================================
Module for managing multi-level organizational and team hierarchies with budget allocation
and access controls.

Pattern: Composite
Async: No — pure data operations
Layer: L1 Infrastructure

Usage:
    from orchestrator.hierarchy import HierarchyManager
    hierarchy = HierarchyManager()
    org = hierarchy.create_org("Acme Corp", budget=10000.0)
    team = hierarchy.create_team("Engineering", org_id=org.id, budget=5000.0)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger("orchestrator.hierarchy")


class NodeType(Enum):
    """Type of node in the hierarchy."""
    ORGANIZATION = "organization"
    TEAM = "team"
    PROJECT = "project"
    USER = "user"


@dataclass
class Node:
    """Represents a node in the hierarchy (org, team, project, user)."""
    
    id: str
    name: str
    node_type: NodeType
    parent_id: Optional[str] = None
    budget: float = 0.0
    allocated_budget: float = 0.0
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def available_budget(self) -> float:
        """Get the available budget (total - allocated)."""
        return self.budget - self.allocated_budget


class HierarchyManager:
    """Manages multi-level organizational and team hierarchies."""

    def __init__(self):
        """Initialize the hierarchy manager."""
        self.nodes: Dict[str, Node] = {}
        self.children_map: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        self.parent_map: Dict[str, str] = {}  # child_id -> parent_id
    
    def create_org(self, name: str, budget: float = 0.0, 
                   metadata: Optional[dict] = None) -> Node:
        """
        Create a new organization.
        
        Args:
            name: Name of the organization
            budget: Budget allocated to the organization
            metadata: Additional metadata
            
        Returns:
            Node: The created organization node
        """
        org_id = f"org_{len(self.nodes)}"
        org_node = Node(
            id=org_id,
            name=name,
            node_type=NodeType.ORGANIZATION,
            budget=budget,
            metadata=metadata or {}
        )
        
        self.nodes[org_id] = org_node
        self.children_map[org_id] = []
        
        logger.info(f"Created organization: {name} (ID: {org_id})")
        return org_node
    
    def create_team(self, name: str, org_id: str, budget: float = 0.0,
                    metadata: Optional[dict] = None) -> Node:
        """
        Create a new team under an organization.
        
        Args:
            name: Name of the team
            org_id: ID of the parent organization
            budget: Budget allocated to the team
            metadata: Additional metadata
            
        Returns:
            Node: The created team node
        """
        if org_id not in self.nodes:
            raise ValueError(f"Organization with ID {org_id} does not exist")
        
        if self.nodes[org_id].node_type != NodeType.ORGANIZATION:
            raise ValueError(f"Parent node {org_id} is not an organization")
        
        # Check if organization has enough budget
        org_node = self.nodes[org_id]
        if org_node.allocated_budget + budget > org_node.budget > 0:
            raise ValueError(f"Insufficient budget in organization {org_id}")
        
        team_id = f"team_{len(self.nodes)}"
        team_node = Node(
            id=team_id,
            name=name,
            node_type=NodeType.TEAM,
            parent_id=org_id,
            budget=budget,
            metadata=metadata or {}
        )
        
        self.nodes[team_id] = team_node
        self.children_map[team_id] = []
        self.children_map[org_id].append(team_id)
        self.parent_map[team_id] = org_id
        
        # Update organization's allocated budget
        org_node.allocated_budget += budget
        
        logger.info(f"Created team: {name} (ID: {team_id}) under org {org_id}")
        return team_node
    
    def create_project(self, name: str, team_id: str, budget: float = 0.0,
                      metadata: Optional[dict] = None) -> Node:
        """
        Create a new project under a team.
        
        Args:
            name: Name of the project
            team_id: ID of the parent team
            budget: Budget allocated to the project
            metadata: Additional metadata
            
        Returns:
            Node: The created project node
        """
        if team_id not in self.nodes:
            raise ValueError(f"Team with ID {team_id} does not exist")
        
        if self.nodes[team_id].node_type != NodeType.TEAM:
            raise ValueError(f"Parent node {team_id} is not a team")
        
        # Check if team has enough budget
        team_node = self.nodes[team_id]
        if team_node.allocated_budget + budget > team_node.budget > 0:
            raise ValueError(f"Insufficient budget in team {team_id}")
        
        project_id = f"proj_{len(self.nodes)}"
        project_node = Node(
            id=project_id,
            name=name,
            node_type=NodeType.PROJECT,
            parent_id=team_id,
            budget=budget,
            metadata=metadata or {}
        )
        
        self.nodes[project_id] = project_node
        self.children_map[project_id] = []
        self.children_map[team_id].append(project_id)
        self.parent_map[project_id] = team_id
        
        # Update team's allocated budget
        team_node.allocated_budget += budget
        
        logger.info(f"Created project: {name} (ID: {project_id}) under team {team_id}")
        return project_node
    
    def create_user(self, name: str, team_id: str, budget: float = 0.0,
                   metadata: Optional[dict] = None) -> Node:
        """
        Create a new user under a team.
        
        Args:
            name: Name of the user
            team_id: ID of the parent team
            budget: Budget allocated to the user
            metadata: Additional metadata
            
        Returns:
            Node: The created user node
        """
        if team_id not in self.nodes:
            raise ValueError(f"Team with ID {team_id} does not exist")
        
        if self.nodes[team_id].node_type != NodeType.TEAM:
            raise ValueError(f"Parent node {team_id} is not a team")
        
        # Check if team has enough budget
        team_node = self.nodes[team_id]
        if team_node.allocated_budget + budget > team_node.budget > 0:
            raise ValueError(f"Insufficient budget in team {team_id}")
        
        user_id = f"user_{len(self.nodes)}"
        user_node = Node(
            id=user_id,
            name=name,
            node_type=NodeType.USER,
            parent_id=team_id,
            budget=budget,
            metadata=metadata or {}
        )
        
        self.nodes[user_id] = user_node
        self.children_map[user_id] = []  # Users don't have children
        self.children_map[team_id].append(user_id)
        self.parent_map[user_id] = team_id
        
        # Update team's allocated budget
        team_node.allocated_budget += budget
        
        logger.info(f"Created user: {name} (ID: {user_id}) under team {team_id}")
        return user_node
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)
    
    def get_children(self, node_id: str) -> List[Node]:
        """Get all children of a node."""
        child_ids = self.children_map.get(node_id, [])
        return [self.nodes[child_id] for child_id in child_ids]
    
    def get_parent(self, node_id: str) -> Optional[Node]:
        """Get the parent of a node."""
        parent_id = self.parent_map.get(node_id)
        if parent_id:
            return self.nodes.get(parent_id)
        return None
    
    def get_path_to_root(self, node_id: str) -> List[Node]:
        """Get the path from a node to the root organization."""
        path = []
        current_id = node_id
        
        while current_id:
            node = self.nodes.get(current_id)
            if node:
                path.append(node)
                current_id = self.parent_map.get(current_id)
            else:
                break
        
        return list(reversed(path))  # Root to leaf order
    
    def allocate_budget(self, node_id: str, amount: float) -> bool:
        """
        Allocate budget to a node, updating all ancestors.
        
        Args:
            node_id: ID of the node to allocate budget to
            amount: Amount to allocate
            
        Returns:
            bool: True if allocation successful, False otherwise
        """
        node = self.nodes.get(node_id)
        if not node:
            logger.error(f"Node {node_id} does not exist")
            return False
        
        # Check if the node has enough available budget
        if node.available_budget() < amount:
            logger.error(f"Insufficient budget in node {node_id}")
            return False
        
        # Update the node's allocated budget
        node.allocated_budget += amount
        
        # Propagate allocation up the hierarchy
        current_id = self.parent_map.get(node_id)
        while current_id:
            parent_node = self.nodes[current_id]
            parent_node.allocated_budget += amount
            current_id = self.parent_map.get(current_id)
        
        logger.info(f"Allocated ${amount} to node {node_id}")
        return True
    
    def release_budget(self, node_id: str, amount: float) -> bool:
        """
        Release budget from a node, updating all ancestors.
        
        Args:
            node_id: ID of the node to release budget from
            amount: Amount to release
            
        Returns:
            bool: True if release successful, False otherwise
        """
        node = self.nodes.get(node_id)
        if not node:
            logger.error(f"Node {node_id} does not exist")
            return False
        
        # Check if the node has allocated enough budget
        if node.allocated_budget < amount:
            logger.error(f"Not enough allocated budget in node {node_id}")
            return False
        
        # Update the node's allocated budget
        node.allocated_budget -= amount
        
        # Propagate release up the hierarchy
        current_id = self.parent_map.get(node_id)
        while current_id:
            parent_node = self.nodes[current_id]
            parent_node.allocated_budget -= amount
            current_id = self.parent_map.get(current_id)
        
        logger.info(f"Released ${amount} from node {node_id}")
        return True
    
    def get_budget_utilization(self, node_id: str) -> Dict[str, float]:
        """
        Get budget utilization for a node and its descendants.
        
        Args:
            node_id: ID of the node to get utilization for
            
        Returns:
            Dict with budget utilization information
        """
        node = self.nodes.get(node_id)
        if not node:
            return {}
        
        total_budget = node.budget
        allocated = node.allocated_budget
        available = node.available_budget()
        utilization = (allocated / total_budget * 100) if total_budget > 0 else 0
        
        return {
            "total_budget": total_budget,
            "allocated": allocated,
            "available": available,
            "utilization_percent": utilization
        }
    
    def get_subtree_budget_utilization(self, node_id: str) -> Dict[str, any]:
        """
        Get budget utilization for a node and its entire subtree.
        
        Args:
            node_id: ID of the root node of the subtree
            
        Returns:
            Dict with subtree budget utilization information
        """
        node = self.nodes.get(node_id)
        if not node:
            return {}
        
        # Calculate total budget and allocation for the subtree
        total_budget = 0
        total_allocated = 0
        
        def traverse_subtree(current_id: str):
            nonlocal total_budget, total_allocated
            current_node = self.nodes[current_id]
            total_budget += current_node.budget
            total_allocated += current_node.allocated_budget
            
            for child_id in self.children_map.get(current_id, []):
                traverse_subtree(child_id)
        
        traverse_subtree(node_id)
        
        utilization = (total_allocated / total_budget * 100) if total_budget > 0 else 0
        
        return {
            "node_id": node_id,
            "total_budget_in_subtree": total_budget,
            "total_allocated_in_subtree": total_allocated,
            "utilization_percent": utilization,
            "node_count": self._count_nodes_in_subtree(node_id)
        }
    
    def _count_nodes_in_subtree(self, node_id: str) -> int:
        """Count the number of nodes in a subtree."""
        count = 1  # Count the current node
        for child_id in self.children_map.get(node_id, []):
            count += self._count_nodes_in_subtree(child_id)
        return count
    
    def validate_hierarchy(self) -> List[str]:
        """
        Validate the hierarchy for inconsistencies.
        
        Returns:
            List of validation errors found
        """
        errors = []
        
        # Check for budget overruns
        for node_id, node in self.nodes.items():
            if node.allocated_budget > node.budget > 0:
                errors.append(f"Budget overrun in node {node_id}: allocated ${node.allocated_budget} > budget ${node.budget}")
        
        # Check for orphaned nodes (nodes with parent_id but no corresponding parent)
        for node_id, node in self.nodes.items():
            if node.parent_id and node.parent_id not in self.nodes:
                errors.append(f"Orphaned node {node_id}: parent {node.parent_id} does not exist")
        
        # Check for cycles in parent-child relationships
        for node_id in self.nodes:
            path = set()
            current = node_id
            while current and current in self.parent_map:
                if current in path:
                    errors.append(f"Cycle detected involving node {current}")
                    break
                path.add(current)
                current = self.parent_map[current]
        
        return errors
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the hierarchy.
        
        Returns:
            Dict with hierarchy statistics
        """
        org_count = sum(1 for node in self.nodes.values() if node.node_type == NodeType.ORGANIZATION)
        team_count = sum(1 for node in self.nodes.values() if node.node_type == NodeType.TEAM)
        project_count = sum(1 for node in self.nodes.values() if node.node_type == NodeType.PROJECT)
        user_count = sum(1 for node in self.nodes.values() if node.node_type == NodeType.USER)
        
        total_budget = sum(node.budget for node in self.nodes.values())
        total_allocated = sum(node.allocated_budget for node in self.nodes.values())
        avg_utilization = (total_allocated / total_budget * 100) if total_budget > 0 else 0
        
        return {
            "node_count": len(self.nodes),
            "org_count": org_count,
            "team_count": team_count,
            "project_count": project_count,
            "user_count": user_count,
            "total_budget": total_budget,
            "total_allocated": total_allocated,
            "average_utilization": avg_utilization
        }
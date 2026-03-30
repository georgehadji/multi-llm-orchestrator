"""
Workspace — Workspace isolation
==============================
Module for managing isolated workspaces with separate configurations, data, and access controls.

Pattern: Repository
Async: Yes — for I/O-bound operations
Layer: L1 Infrastructure

Usage:
    from orchestrator.workspace import WorkspaceManager
    ws_manager = WorkspaceManager(base_dir="./workspaces")
    workspace = ws_manager.create_workspace("project_alpha", owner="user123")
    ws_manager.activate_workspace(workspace.id)
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("orchestrator.workspace")


class Workspace:
    """Represents a single workspace."""

    def __init__(self, id: str, name: str, owner: str, description: str = "",
                 created_at: datetime = None, metadata: dict = None):
        self.id = id
        self.name = name
        self.owner = owner
        self.description = description
        self.created_at = created_at or datetime.now()
        self.metadata = metadata or {}
        self.members: list[str] = []
        self.settings: dict[str, any] = {}
        self.stats = {
            "tasks_completed": 0,
            "tokens_used": 0,
            "storage_used": 0
        }

    def to_dict(self) -> dict[str, any]:
        """Convert workspace to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "owner": self.owner,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "members": self.members,
            "settings": self.settings,
            "stats": self.stats
        }

    @classmethod
    def from_dict(cls, data: dict[str, any]) -> Workspace:
        """Create workspace from dictionary."""
        workspace = cls(
            id=data["id"],
            name=data["name"],
            owner=data["owner"],
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {})
        )
        workspace.members = data.get("members", [])
        workspace.settings = data.get("settings", {})
        workspace.stats = data.get("stats", {"tasks_completed": 0, "tokens_used": 0, "storage_used": 0})
        return workspace


class WorkspaceManager:
    """Manages isolated workspaces with separate configurations and data."""

    def __init__(self, base_dir: str = "./workspaces"):
        """Initialize the workspace manager."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.workspaces: dict[str, Workspace] = {}
        self.active_workspace_id: str | None = None
        self.workspace_dirs: dict[str, Path] = {}

        # Load existing workspaces
        self._load_workspaces()

    def _load_workspaces(self):
        """Load existing workspaces from the base directory."""
        for workspace_dir in self.base_dir.iterdir():
            if workspace_dir.is_dir():
                config_file = workspace_dir / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file, encoding='utf-8') as f:
                            data = json.load(f)

                        workspace = Workspace.from_dict(data)
                        self.workspaces[workspace.id] = workspace
                        self.workspace_dirs[workspace.id] = workspace_dir

                        logger.info(f"Loaded workspace: {workspace.name} (ID: {workspace.id})")
                    except Exception as e:
                        logger.error(f"Failed to load workspace from {config_file}: {e}")

    def create_workspace(self, name: str, owner: str, description: str = "",
                        metadata: dict = None) -> Workspace:
        """
        Create a new workspace.

        Args:
            name: Name of the workspace
            owner: Owner of the workspace
            description: Description of the workspace
            metadata: Additional metadata

        Returns:
            Workspace: The created workspace
        """
        # Generate unique ID based on name and owner
        id_source = f"{name}_{owner}_{datetime.now().isoformat()}"
        workspace_id = hashlib.sha256(id_source.encode()).hexdigest()[:12]

        # Create workspace directory
        workspace_dir = self.base_dir / workspace_id
        workspace_dir.mkdir(exist_ok=True)

        # Create subdirectories for different types of data
        (workspace_dir / "data").mkdir(exist_ok=True)
        (workspace_dir / "models").mkdir(exist_ok=True)
        (workspace_dir / "cache").mkdir(exist_ok=True)
        (workspace_dir / "logs").mkdir(exist_ok=True)
        (workspace_dir / "temp").mkdir(exist_ok=True)

        # Create workspace object
        workspace = Workspace(
            id=workspace_id,
            name=name,
            owner=owner,
            description=description,
            metadata=metadata or {}
        )

        # Add owner as the first member
        workspace.members.append(owner)

        # Save workspace configuration
        config_file = workspace_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(workspace.to_dict(), f, indent=2, default=str)

        # Store in memory
        self.workspaces[workspace_id] = workspace
        self.workspace_dirs[workspace_id] = workspace_dir

        logger.info(f"Created workspace: {name} (ID: {workspace_id})")
        return workspace

    def get_workspace(self, workspace_id: str) -> Workspace | None:
        """Get a workspace by its ID."""
        return self.workspaces.get(workspace_id)

    def list_workspaces(self, owner: str | None = None) -> list[Workspace]:
        """
        List all workspaces, optionally filtered by owner.

        Args:
            owner: Optional owner to filter by

        Returns:
            List of workspaces
        """
        if owner:
            return [ws for ws in self.workspaces.values() if ws.owner == owner]
        return list(self.workspaces.values())

    def activate_workspace(self, workspace_id: str) -> bool:
        """
        Activate a workspace, making it the current one for operations.

        Args:
            workspace_id: ID of the workspace to activate

        Returns:
            bool: True if activation successful, False otherwise
        """
        if workspace_id in self.workspaces:
            self.active_workspace_id = workspace_id
            logger.info(f"Activated workspace: {workspace_id}")
            return True
        else:
            logger.error(f"Workspace {workspace_id} does not exist")
            return False

    def deactivate_workspace(self):
        """Deactivate the current workspace."""
        self.active_workspace_id = None
        logger.info("Deactivated workspace")

    def get_active_workspace(self) -> Workspace | None:
        """Get the currently active workspace."""
        if self.active_workspace_id:
            return self.workspaces.get(self.active_workspace_id)
        return None

    def add_member(self, workspace_id: str, user_id: str) -> bool:
        """
        Add a member to a workspace.

        Args:
            workspace_id: ID of the workspace
            user_id: ID of the user to add

        Returns:
            bool: True if addition successful, False otherwise
        """
        workspace = self.workspaces.get(workspace_id)
        if not workspace:
            logger.error(f"Workspace {workspace_id} does not exist")
            return False

        if user_id not in workspace.members:
            workspace.members.append(user_id)

            # Update config file
            config_file = self.workspace_dirs[workspace_id] / "config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(workspace.to_dict(), f, indent=2, default=str)

            logger.info(f"Added member {user_id} to workspace {workspace_id}")
            return True

        return False

    def remove_member(self, workspace_id: str, user_id: str) -> bool:
        """
        Remove a member from a workspace.

        Args:
            workspace_id: ID of the workspace
            user_id: ID of the user to remove

        Returns:
            bool: True if removal successful, False otherwise
        """
        workspace = self.workspaces.get(workspace_id)
        if not workspace:
            logger.error(f"Workspace {workspace_id} does not exist")
            return False

        if user_id in workspace.members and user_id != workspace.owner:
            workspace.members.remove(user_id)

            # Update config file
            config_file = self.workspace_dirs[workspace_id] / "config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(workspace.to_dict(), f, indent=2, default=str)

            logger.info(f"Removed member {user_id} from workspace {workspace_id}")
            return True

        return False

    def update_settings(self, workspace_id: str, settings: dict[str, any]) -> bool:
        """
        Update settings for a workspace.

        Args:
            workspace_id: ID of the workspace
            settings: New settings to apply

        Returns:
            bool: True if update successful, False otherwise
        """
        workspace = self.workspaces.get(workspace_id)
        if not workspace:
            logger.error(f"Workspace {workspace_id} does not exist")
            return False

        workspace.settings.update(settings)

        # Update config file
        config_file = self.workspace_dirs[workspace_id] / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(workspace.to_dict(), f, indent=2, default=str)

        logger.info(f"Updated settings for workspace {workspace_id}")
        return True

    def get_workspace_path(self, workspace_id: str, subfolder: str = "") -> Path | None:
        """
        Get the path for a workspace, optionally with a subfolder.

        Args:
            workspace_id: ID of the workspace
            subfolder: Optional subfolder within the workspace

        Returns:
            Path to the workspace or subfolder, or None if workspace doesn't exist
        """
        if workspace_id not in self.workspace_dirs:
            return None

        workspace_dir = self.workspace_dirs[workspace_id]
        if subfolder:
            return workspace_dir / subfolder
        return workspace_dir

    def update_workspace_stats(self, workspace_id: str, **stats) -> bool:
        """
        Update statistics for a workspace.

        Args:
            workspace_id: ID of the workspace
            **stats: Statistics to update

        Returns:
            bool: True if update successful, False otherwise
        """
        workspace = self.workspaces.get(workspace_id)
        if not workspace:
            logger.error(f"Workspace {workspace_id} does not exist")
            return False

        for stat, value in stats.items():
            if stat in workspace.stats:
                if isinstance(workspace.stats[stat], (int, float)):
                    workspace.stats[stat] += value
                else:
                    workspace.stats[stat] = value

        # Update config file
        config_file = self.workspace_dirs[workspace_id] / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(workspace.to_dict(), f, indent=2, default=str)

        logger.info(f"Updated stats for workspace {workspace_id}: {stats}")
        return True

    async def cleanup_workspace(self, workspace_id: str,
                               preserve_recent_days: int = 30) -> bool:
        """
        Clean up a workspace by removing old temporary files and logs.

        Args:
            workspace_id: ID of the workspace to clean up
            preserve_recent_days: Number of days to preserve recent files

        Returns:
            bool: True if cleanup successful, False otherwise
        """
        import time

        workspace_dir = self.workspace_dirs.get(workspace_id)
        if not workspace_dir:
            logger.error(f"Workspace {workspace_id} does not exist")
            return False

        try:
            cutoff_time = time.time() - (preserve_recent_days * 24 * 60 * 60)

            # Clean up temp directory
            temp_dir = workspace_dir / "temp"
            if temp_dir.exists():
                for file_path in temp_dir.iterdir():
                    if file_path.stat().st_mtime < cutoff_time:
                        if file_path.is_file():
                            file_path.unlink()
                            logger.info(f"Deleted old temp file: {file_path}")
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                            logger.info(f"Deleted old temp directory: {file_path}")

            # Clean up logs directory
            logs_dir = workspace_dir / "logs"
            if logs_dir.exists():
                for file_path in logs_dir.iterdir():
                    if file_path.stat().st_mtime < cutoff_time:
                        if file_path.is_file():
                            file_path.unlink()
                            logger.info(f"Deleted old log file: {file_path}")

            logger.info(f"Cleaned up workspace {workspace_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clean up workspace {workspace_id}: {e}")
            return False

    def delete_workspace(self, workspace_id: str, delete_files: bool = True) -> bool:
        """
        Delete a workspace and optionally its files.

        Args:
            workspace_id: ID of the workspace to delete
            delete_files: Whether to delete the workspace files from disk

        Returns:
            bool: True if deletion successful, False otherwise
        """
        if workspace_id not in self.workspaces:
            logger.error(f"Workspace {workspace_id} does not exist")
            return False

        # Remove from memory
        workspace = self.workspaces.pop(workspace_id)
        workspace_dir = self.workspace_dirs.pop(workspace_id)

        # Delete files if requested
        if delete_files and workspace_dir.exists():
            try:
                shutil.rmtree(workspace_dir)
                logger.info(f"Deleted workspace files: {workspace_dir}")
            except Exception as e:
                logger.error(f"Failed to delete workspace files {workspace_dir}: {e}")
                # Still return True as the workspace is removed from memory
        elif not delete_files:
            logger.info(f"Kept workspace files at: {workspace_dir}")

        # Update active workspace if this was the active one
        if self.active_workspace_id == workspace_id:
            self.active_workspace_id = None

        logger.info(f"Deleted workspace: {workspace.name} (ID: {workspace_id})")
        return True

    def get_workspace_usage(self, workspace_id: str) -> dict[str, any]:
        """
        Get usage statistics for a workspace.

        Args:
            workspace_id: ID of the workspace

        Returns:
            Dict with usage statistics
        """
        workspace = self.workspaces.get(workspace_id)
        if not workspace:
            return {}

        workspace_dir = self.workspace_dirs.get(workspace_id)
        if not workspace_dir:
            return {}

        # Calculate disk usage
        total_size = 0
        for dirpath, _dirnames, filenames in walk(workspace_dir):
            for filename in filenames:
                filepath = Path(dirpath) / filename
                total_size += filepath.stat().st_size

        return {
            "workspace_id": workspace_id,
            "name": workspace.name,
            "owner": workspace.owner,
            "member_count": len(workspace.members),
            "disk_usage_bytes": total_size,
            "disk_usage_mb": round(total_size / (1024 * 1024), 2),
            "stats": workspace.stats
        }


import os


# Helper function to walk directories (since os.walk isn't async)
def walk(path):
    """Helper to walk directories synchronously."""
    yield from os.walk(path)

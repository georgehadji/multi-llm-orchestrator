"""
EntityRLS - Row-level and field-level security rule generation.
=================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 12, Phase B5 (Base44-inspired).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import json
import logging

logger = logging.getLogger(__name__)


class RLSLevel(str):
    ROW = "row"
    FIELD = "field"
    TENANT = "tenant"


@dataclass
class RLSPolicy:
    entity: str  # Table/model name
    level: str = RLSLevel.ROW
    condition: str = ""  # SQL WHERE clause or policy expression
    fields_visible: list = field(default_factory=list)  # For field-level
    fields_hidden: list = field(default_factory=list)  # For field-level
    roles: list = field(default_factory=list)  # Which roles this applies to (empty = all)
    description: str = ""

    def to_dict(self):
        return {
            "entity": self.entity,
            "level": self.level,
            "condition": self.condition,
            "fields_visible": self.fields_visible,
            "fields_hidden": self.fields_hidden,
            "roles": self.roles,
            "description": self.description,
        }


class EntityRLSManager:
    """Generates row-level and field-level security policies for entities."""

    def __init__(self):
        self._policies: list[RLSPolicy] = []

    def add_policy(self, policy):
        self._policies.append(policy)

    def generate_row_level(self, entity, owner_field="user_id", auth_context="auth.uid()"):
        """Generate a standard row-level security policy.

        Creates: "user can only see/update their own records"
        """
        condition = f"{owner_field} = {auth_context}"
        policy = RLSPolicy(
            entity=entity,
            level=RLSLevel.ROW,
            condition=condition,
            description=f"Users can only access their own {entity} records",
        )
        self._policies.append(policy)
        return policy

    def generate_tenant_isolation(
        self, entity, tenant_field="tenant_id", auth_context="auth.tenant_id()"
    ):
        """Generate tenant-level isolation policy."""
        policy = RLSPolicy(
            entity=entity,
            level=RLSLevel.TENANT,
            condition=f"{tenant_field} = {auth_context}",
            description=f"Tenant isolation for {entity}",
        )
        self._policies.append(policy)
        return policy

    def generate_field_level(self, entity, public_fields, hidden_fields=None, roles=None):
        """Generate field-level visibility policy."""
        policy = RLSPolicy(
            entity=entity,
            level=RLSLevel.FIELD,
            fields_visible=public_fields,
            fields_hidden=hidden_fields or [],
            roles=roles or ["public"],
            description=f"Field-level access for {entity}",
        )
        self._policies.append(policy)
        return policy

    def generate_role_based(self, entity, role_conditions):
        """Generate role-based policies for an entity.

        Args:
            entity: Entity name
            role_conditions: Dict of {role: condition} e.g. {"admin": "true", "user": "user_id = auth.uid()"}
        """
        policies = []
        for role, condition in role_conditions.items():
            policy = RLSPolicy(
                entity=entity,
                level=RLSLevel.ROW,
                condition=condition,
                roles=[role],
                description=f"{role} access to {entity}",
            )
            self._policies.append(policy)
            policies.append(policy)
        return policies

    def to_sql(self, entity):
        """Generate SQL RLS statements for an entity."""
        entity_policies = [p for p in self._policies if p.entity == entity]
        statements = [
            f"-- RLS Policies for {entity}",
            f"ALTER TABLE {entity} ENABLE ROW LEVEL SECURITY;",
            "",
        ]
        for p in entity_policies:
            if p.level in (RLSLevel.ROW, RLSLevel.TENANT):
                role_filter = ""
                if p.roles:
                    role_list = ", ".join(f"'{r}'" for r in p.roles)
                    role_filter = f" TO {role_list}"
                stmt = f'CREATE POLICY "{p.description}" ON {entity} FOR ALL{role_filter} USING ({p.condition});'
                statements.append(stmt)
            elif p.level == RLSLevel.FIELD:
                cols = ", ".join(p.fields_visible)
                statements.append(f"-- Field-level: visible columns for {entity}: {cols}")
        return "\n".join(statements)

    def validate(self, entity, user_context=None):
        """Validate that RLS policies are consistent for an entity."""
        issues = []
        entity_policies = [p for p in self._policies if p.entity == entity]
        if not entity_policies:
            issues.append(f"No RLS policies defined for {entity}")
        for p in entity_policies:
            if p.level in (RLSLevel.ROW, RLSLevel.TENANT) and not p.condition:
                issues.append(f"Row-level policy for {entity} has empty condition")
        return issues

    def list_policies(self):
        return [p.to_dict() for p in self._policies]
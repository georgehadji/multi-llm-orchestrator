"""
Database Generator — Schema and migration generation
=====================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Generate database schemas, migrations, and ORM models from descriptions.
Supports PostgreSQL, SQLite, MySQL, and MongoDB.

Features:
- Schema generation from description
- Migration file generation
- ORM model generation (SQLAlchemy, Prisma)
- Relationship detection
- Index optimization

USAGE:
    from orchestrator.database_generator import DatabaseSchemaGenerator
    
    generator = DatabaseSchemaGenerator()
    
    # Generate schema from description
    schema = await generator.from_description(
        "Task management app with users, projects, and comments",
        db_type="postgresql",
    )
    
    # Generate migrations
    migrations = await generator.generate_migrations(schema)
    
    # Generate ORM models
    models = generator.generate_models(schema, orm="sqlalchemy")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Any

logger = logging.getLogger("orchestrator.database_generator")


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class DatabaseType(str, Enum):
    """Database types."""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    MYSQL = "mysql"
    MONGODB = "mongodb"


class ORMType(str, Enum):
    """ORM types."""
    SQLALCHEMY = "sqlalchemy"
    PRISMA = "prisma"
    DJANGO = "django"
    NONE = "none"


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class Column:
    """Database column."""
    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False
    unique: bool = False
    default: Optional[str] = None
    foreign_key: Optional[str] = None
    index: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "nullable": self.nullable,
            "primary_key": self.primary_key,
            "unique": self.unique,
            "default": self.default,
            "foreign_key": self.foreign_key,
            "index": self.index,
        }


@dataclass
class Table:
    """Database table."""
    name: str
    columns: List[Column] = field(default_factory=list)
    indexes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "columns": [c.to_dict() for c in self.columns],
            "indexes": self.indexes,
        }


@dataclass
class DatabaseSchema:
    """Database schema."""
    tables: List[Table] = field(default_factory=list)
    relationships: List[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "tables": [t.to_dict() for t in self.tables],
            "relationships": self.relationships,
        }


# ─────────────────────────────────────────────
# Database Schema Generator
# ─────────────────────────────────────────────

class DatabaseSchemaGenerator:
    """
    Generate database schemas from descriptions.
    
    Analyzes project descriptions to infer tables, columns,
    and relationships.
    """
    
    def __init__(self):
        self._total_generations = 0
    
    async def from_description(
        self,
        description: str,
        db_type: DatabaseType = DatabaseType.POSTGRESQL,
    ) -> DatabaseSchema:
        """
        Generate schema from description.
        
        Args:
            description: Project description
            db_type: Database type
        
        Returns:
            Database schema
        """
        schema = DatabaseSchema()
        
        # Always include users table
        schema.tables.append(self._create_users_table())
        
        # Analyze description for additional tables
        desc_lower = description.lower()
        
        # Task/Todo tables
        if any(word in desc_lower for word in ["task", "todo", "project"]):
            schema.tables.append(self._create_tasks_table())
        
        # Comment tables
        if any(word in desc_lower for word in ["comment", "feedback", "review"]):
            schema.tables.append(self._create_comments_table())
        
        # Product/E-commerce tables
        if any(word in desc_lower for word in ["product", "order", "cart", "shop"]):
            schema.tables.extend([
                self._create_products_table(),
                self._create_orders_table(),
            ])
        
        # Blog/Content tables
        if any(word in desc_lower for word in ["blog", "post", "article", "content"]):
            schema.tables.append(self._create_posts_table())
        
        # Generate relationships
        schema.relationships = self._generate_relationships(schema.tables)
        
        self._total_generations += 1
        
        logger.info(
            f"Generated schema with {len(schema.tables)} tables for {db_type.value}"
        )
        
        return schema
    
    async def generate_migrations(
        self,
        schema: DatabaseSchema,
        db_type: DatabaseType = DatabaseType.POSTGRESQL,
    ) -> List[str]:
        """
        Generate migration files.
        
        Args:
            schema: Database schema
            db_type: Database type
        
        Returns:
            List of migration SQL files
        """
        migrations = []
        
        # Initial migration
        migration = f"""-- Initial migration
-- Generated by AI Orchestrator Database Generator
-- Database: {db_type.value}

"""
        
        for table in schema.tables:
            migration += self._generate_create_table(table, db_type)
            migration += "\n"
        
        migrations.append(migration)
        
        return migrations
    
    def generate_models(
        self,
        schema: DatabaseSchema,
        orm: ORMType = ORMType.SQLALCHEMY,
    ) -> Dict[str, str]:
        """
        Generate ORM models.
        
        Args:
            schema: Database schema
            orm: ORM type
        
        Returns:
            Dictionary of model files
        """
        if orm == ORMType.SQLALCHEMY:
            return self._generate_sqlalchemy_models(schema)
        elif orm == ORMType.PRISMA:
            return self._generate_prisma_schema(schema)
        elif orm == ORMType.DJANGO:
            return self._generate_django_models(schema)
        else:
            return {}
    
    def _create_users_table(self) -> Table:
        """Create users table."""
        return Table(
            name="users",
            columns=[
                Column("id", "UUID", primary_key=True, default="gen_random_uuid()"),
                Column("email", "VARCHAR(255)", nullable=False, unique=True),
                Column("password_hash", "VARCHAR(255)", nullable=False),
                Column("name", "VARCHAR(255)"),
                Column("created_at", "TIMESTAMP", default="NOW()"),
                Column("updated_at", "TIMESTAMP", default="NOW()"),
            ],
            indexes=["idx_users_email"],
        )
    
    def _create_tasks_table(self) -> Table:
        """Create tasks table."""
        return Table(
            name="tasks",
            columns=[
                Column("id", "UUID", primary_key=True, default="gen_random_uuid()"),
                Column("title", "VARCHAR(255)", nullable=False),
                Column("description", "TEXT"),
                Column("status", "VARCHAR(50)", default="'pending'"),
                Column("priority", "INTEGER", default="0"),
                Column("user_id", "UUID", foreign_key="users.id"),
                Column("created_at", "TIMESTAMP", default="NOW()"),
                Column("updated_at", "TIMESTAMP", default="NOW()"),
            ],
            indexes=["idx_tasks_user_id", "idx_tasks_status"],
        )
    
    def _create_comments_table(self) -> Table:
        """Create comments table."""
        return Table(
            name="comments",
            columns=[
                Column("id", "UUID", primary_key=True, default="gen_random_uuid()"),
                Column("content", "TEXT", nullable=False),
                Column("user_id", "UUID", foreign_key="users.id"),
                Column("task_id", "UUID", foreign_key="tasks.id"),
                Column("created_at", "TIMESTAMP", default="NOW()"),
            ],
            indexes=["idx_comments_user_id", "idx_comments_task_id"],
        )
    
    def _create_products_table(self) -> Table:
        """Create products table."""
        return Table(
            name="products",
            columns=[
                Column("id", "UUID", primary_key=True, default="gen_random_uuid()"),
                Column("name", "VARCHAR(255)", nullable=False),
                Column("description", "TEXT"),
                Column("price", "DECIMAL(10,2)", nullable=False),
                Column("stock", "INTEGER", default="0"),
                Column("created_at", "TIMESTAMP", default="NOW()"),
            ],
        )
    
    def _create_orders_table(self) -> Table:
        """Create orders table."""
        return Table(
            name="orders",
            columns=[
                Column("id", "UUID", primary_key=True, default="gen_random_uuid()"),
                Column("user_id", "UUID", foreign_key="users.id"),
                Column("total", "DECIMAL(10,2)", nullable=False),
                Column("status", "VARCHAR(50)", default="'pending'"),
                Column("created_at", "TIMESTAMP", default="NOW()"),
            ],
        )
    
    def _create_posts_table(self) -> Table:
        """Create posts table."""
        return Table(
            name="posts",
            columns=[
                Column("id", "UUID", primary_key=True, default="gen_random_uuid()"),
                Column("title", "VARCHAR(255)", nullable=False),
                Column("content", "TEXT", nullable=False),
                Column("user_id", "UUID", foreign_key="users.id"),
                Column("published", "BOOLEAN", default="FALSE"),
                Column("created_at", "TIMESTAMP", default="NOW()"),
            ],
        )
    
    def _generate_relationships(self, tables: List[Table]) -> List[dict]:
        """Generate relationships from tables."""
        relationships = []
        
        for table in tables:
            for column in table.columns:
                if column.foreign_key:
                    relationships.append({
                        "type": "many_to_one",
                        "from_table": table.name,
                        "from_column": column.name,
                        "to_table": column.foreign_key.split(".")[0],
                        "to_column": column.foreign_key.split(".")[1],
                    })
        
        return relationships
    
    def _generate_create_table(
        self,
        table: Table,
        db_type: DatabaseType,
    ) -> str:
        """Generate CREATE TABLE SQL."""
        sql = f"CREATE TABLE IF NOT EXISTS {table.name} (\n"
        
        column_defs = []
        for column in table.columns:
            col_sql = f"    {column.name} {column.type}"
            
            if column.primary_key:
                col_sql += " PRIMARY KEY"
            
            if column.default:
                col_sql += f" DEFAULT {column.default}"
            
            if not column.nullable and not column.primary_key:
                col_sql += " NOT NULL"
            
            if column.unique:
                col_sql += " UNIQUE"
            
            if column.foreign_key:
                col_sql += f" REFERENCES {column.foreign_key}"
            
            column_defs.append(col_sql)
        
        sql += ",\n".join(column_defs)
        sql += "\n);\n"
        
        # Add indexes
        for index in table.indexes:
            sql += f"CREATE INDEX IF NOT EXISTS {index} ON {table.name} ({index.replace('idx_', '')});\n"
        
        return sql
    
    def _generate_sqlalchemy_models(
        self,
        schema: DatabaseSchema,
    ) -> Dict[str, str]:
        """Generate SQLAlchemy models."""
        files = {}
        
        models_code = '''"""
Database Models
Generated by AI Orchestrator Database Generator
"""

from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, ForeignKey, DECIMAL
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

'''
        
        for table in schema.tables:
            models_code += self._generate_sqlalchemy_model(table)
            models_code += "\n\n"
        
        files["models.py"] = models_code
        
        return files
    
    def _generate_sqlalchemy_model(self, table: Table) -> str:
        """Generate single SQLAlchemy model."""
        class_name = table.name.title().replace("_", "")
        
        code = f"class {class_name}(Base):\n"
        code += f'    """{table.name} table."""\n\n'
        code += f"    __tablename__ = '{table.name}'\n\n"
        
        for column in table.columns:
            col_type = self._sqlalchemy_type(column.type)
            code += f"    {column.name} = Column({col_type}"
            
            if column.primary_key:
                code += ", primary_key=True, default=uuid.uuid4"
            
            if column.nullable:
                code += ", nullable=True"
            else:
                code += ", nullable=False"
            
            if column.default and column.default != "NOW()" and column.default != "gen_random_uuid()":
                code += f", default={column.default}"
            
            if column.foreign_key:
                code += f", ForeignKey('{column.foreign_key}')"
            
            code += ")\n"
        
        return code
    
    def _generate_prisma_schema(
        self,
        schema: DatabaseSchema,
    ) -> Dict[str, str]:
        """Generate Prisma schema."""
        files = {}
        
        prisma_code = '''// Prisma Schema
// Generated by AI Orchestrator Database Generator

generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

'''
        
        for table in schema.tables:
            prisma_code += self._generate_prisma_model(table)
            prisma_code += "\n\n"
        
        files["schema.prisma"] = prisma_code
        
        return files
    
    def _generate_prisma_model(self, table: Table) -> str:
        """Generate single Prisma model."""
        model_name = table.name.title().replace("_", "")
        
        code = f"model {model_name} {{\n"
        
        for column in table.columns:
            prisma_type = self._prisma_type(column.type)
            code += f"  {column.name} {prisma_type}"
            
            if column.primary_key:
                code += " @id @default(uuid())"
            
            if column.unique:
                code += " @unique"
            
            if column.foreign_key:
                ref_table = column.foreign_key.split(".")[0].title().replace("_", "")
                code += f" @relation(references: [id])"
            
            code += "\n"
        
        code += "}\n"
        
        return code
    
    def _generate_django_models(
        self,
        schema: DatabaseSchema,
    ) -> Dict[str, str]:
        """Generate Django models."""
        files = {}
        
        models_code = '''"""
Django Models
Generated by AI Orchestrator Database Generator
"""

from django.db import models
from django.utils import timezone
import uuid

'''
        
        for table in schema.tables:
            models_code += self._generate_django_model(table)
            models_code += "\n\n"
        
        files["models.py"] = models_code
        
        return files
    
    def _generate_django_model(self, table: Table) -> str:
        """Generate single Django model."""
        class_name = table.name.title().replace("_", "")
        
        code = f"class {class_name}(models.Model):\n"
        code += f'    """{table.name} table."""\n\n'
        
        for column in table.columns:
            django_type = self._django_type(column.type)
            code += f"    {column.name} = {django_type}"
            
            if column.nullable:
                code += "(null=True, blank=True)"
            else:
                code += "()"
            
            code += "\n"
        
        code += "\n    class Meta:\n"
        code += f"        db_table = '{table.name}'\n"
        
        return code
    
    def _sqlalchemy_type(self, db_type: str) -> str:
        """Convert DB type to SQLAlchemy type."""
        type_map = {
            "UUID": "UUID(as_uuid=True)",
            "VARCHAR(255)": "String(255)",
            "TEXT": "Text",
            "INTEGER": "Integer",
            "BOOLEAN": "Boolean",
            "TIMESTAMP": "DateTime",
            "DECIMAL(10,2)": "DECIMAL(10, 2)",
        }
        return type_map.get(db_type, "String")
    
    def _prisma_type(self, db_type: str) -> str:
        """Convert DB type to Prisma type."""
        type_map = {
            "UUID": "String",
            "VARCHAR(255)": "String",
            "TEXT": "String",
            "INTEGER": "Int",
            "BOOLEAN": "Boolean",
            "TIMESTAMP": "DateTime",
            "DECIMAL(10,2)": "Decimal",
        }
        return type_map.get(db_type, "String")
    
    def _django_type(self, db_type: str) -> str:
        """Convert DB type to Django field."""
        type_map = {
            "UUID": "UUIDField",
            "VARCHAR(255)": "CharField(max_length=255)",
            "TEXT": "TextField",
            "INTEGER": "IntegerField",
            "BOOLEAN": "BooleanField",
            "TIMESTAMP": "DateTimeField",
            "DECIMAL(10,2)": "DecimalField(max_digits=10, decimal_places=2)",
        }
        return type_map.get(db_type, "CharField")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "total_generations": self._total_generations,
        }


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

_default_generator: Optional[DatabaseSchemaGenerator] = None


def get_database_generator() -> DatabaseSchemaGenerator:
    """Get or create default database generator."""
    global _default_generator
    if _default_generator is None:
        _default_generator = DatabaseSchemaGenerator()
    return _default_generator


def reset_database_generator() -> None:
    """Reset default generator (for testing)."""
    global _default_generator
    _default_generator = None


async def generate_database_schema(
    description: str,
    db_type: str = "postgresql",
) -> DatabaseSchema:
    """Generate database schema from description."""
    generator = get_database_generator()
    return await generator.from_description(description, DatabaseType(db_type))

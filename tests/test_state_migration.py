"""
Tests for database schema migration (Task 2: Database Persistence).

Tests cover:
- Idempotent migration on empty databases
- Migration with existing projects
- Helper function for keyword extraction and JSON storage
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from orchestrator.state import (
    extract_and_store_keywords,
    migrate_add_resume_fields,
)


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def empty_db(temp_db):
    """Create an empty database with the projects table but no data."""
    conn = sqlite3.connect(str(temp_db))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            project_id TEXT PRIMARY KEY,
            state      TEXT NOT NULL,
            status     TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    return temp_db


@pytest.fixture
def db_with_projects(temp_db):
    """Create a database with 3 existing projects."""
    conn = sqlite3.connect(str(temp_db))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            project_id TEXT PRIMARY KEY,
            state      TEXT NOT NULL,
            status     TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    """)

    # Insert 3 test projects
    test_projects = [
        ("proj1", '{"test": "data"}', "COMPLETED", 1.0, 2.0),
        ("proj2", '{"test": "data"}', "PARTIAL_SUCCESS", 3.0, 4.0),
        ("proj3", '{"test": "data"}', "FAILED", 5.0, 6.0),
    ]

    cursor.executemany(
        "INSERT INTO projects (project_id, state, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        test_projects,
    )
    conn.commit()
    conn.close()
    return temp_db


class TestMigration:
    """Tests for migrate_add_resume_fields function."""

    def test_migration_on_empty_database(self, empty_db):
        """Test 1: Migration on Empty Database"""
        # Call migration
        result = migrate_add_resume_fields(empty_db)

        # Verify returns True
        assert result is True

        # Verify columns exist
        conn = sqlite3.connect(str(empty_db))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(projects)")
        columns = {row[1] for row in cursor.fetchall()}

        assert "project_description" in columns
        assert "keywords_json" in columns
        conn.close()

    def test_migration_on_database_with_existing_projects(self, db_with_projects):
        """Test 2: Migration on Database with Existing Projects"""
        # Call migration
        result = migrate_add_resume_fields(db_with_projects)

        # Verify returns True
        assert result is True

        # Verify columns added
        conn = sqlite3.connect(str(db_with_projects))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(projects)")
        columns = {row[1] for row in cursor.fetchall()}

        assert "project_description" in columns
        assert "keywords_json" in columns

        # Verify existing projects have NULL for new columns
        cursor.execute("SELECT project_id, project_description, keywords_json FROM projects")
        rows = cursor.fetchall()

        assert len(rows) == 3
        for project_id, description, keywords in rows:
            assert description is None
            assert keywords is None

        conn.close()

    def test_idempotent_migration_safe_to_call_twice(self, empty_db):
        """Test 3: Idempotent Migration (safe to call twice)"""
        # Call migration first time
        result1 = migrate_add_resume_fields(empty_db)
        assert result1 is True

        # Call migration second time
        result2 = migrate_add_resume_fields(empty_db)
        assert result2 is True

        # Verify no errors and columns still exist
        conn = sqlite3.connect(str(empty_db))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(projects)")
        columns = {row[1] for row in cursor.fetchall()}

        assert "project_description" in columns
        assert "keywords_json" in columns
        conn.close()


class TestExtractAndStoreKeywords:
    """Tests for extract_and_store_keywords helper function."""

    def test_extract_and_store_keywords_with_valid_description(self):
        """Test 4: extract_and_store_keywords() with Valid Description"""
        description = "Build a REST API with FastAPI for production"

        result = extract_and_store_keywords(description)

        # Verify returns JSON string
        assert result is not None
        assert isinstance(result, str)

        # Verify can be parsed back to list
        keywords = json.loads(result)
        assert isinstance(keywords, list)

        # Verify keywords are sorted and expected
        expected = ["api", "build", "fastapi", "production", "rest"]
        assert keywords == expected

    def test_extract_and_store_keywords_with_none_input(self):
        """Test 5a: extract_and_store_keywords() with None input"""
        result = extract_and_store_keywords(None)

        # Verify returns None
        assert result is None

    def test_extract_and_store_keywords_with_empty_string(self):
        """Test 5b: extract_and_store_keywords() with Empty String"""
        result = extract_and_store_keywords("")

        # Verify returns None
        assert result is None

    def test_extract_and_store_keywords_with_stopwords_only(self):
        """Test 5c: extract_and_store_keywords() filters stopwords correctly"""
        # Text with only stopwords and short words
        description = "the a an is are and or to for of in on at"

        result = extract_and_store_keywords(description)

        # Should return None since all words are stopwords or too short
        assert result is None

    def test_extract_and_store_keywords_json_format(self):
        """Test 5d: Verify JSON format is valid and sorted"""
        description = "Python Django REST API backend"

        result = extract_and_store_keywords(description)

        # Verify it's valid JSON
        keywords = json.loads(result)

        # Verify it's sorted alphabetically
        assert keywords == sorted(keywords)
        # Verify format is array
        assert isinstance(keywords, list)

"""
Tests for config_as_code.py + type_generator.py — Entity schemas, type generation.
"""

from __future__ import annotations


import pytest

from orchestrator.config_as_code import (
    AppConfig,
    EntitySchema,
    EntityField,
    FieldType,
    AuthMethod,
)
from orchestrator.generators.type_generator import (
    DynamicTypeGenerator,
    TargetLanguage,
)


class TestEntityField:
    """Tests for EntityField dataclass."""

    def test_field_defaults(self):
        """Field must have sensible defaults."""
        f = EntityField(name="email", type=FieldType.EMAIL, required=True)
        assert f.name == "email"
        assert f.type == FieldType.EMAIL
        assert f.required
        assert not f.unique

    @pytest.mark.parametrize("ftype", list(FieldType))
    def test_all_field_types(self, ftype):
        """All FieldTypes must be valid."""
        f = EntityField(name="test", type=ftype)
        assert f.type == ftype


class TestEntitySchema:
    """Tests for EntitySchema."""

    def test_pydantic_generation(self):
        """Entity must generate valid Pydantic model."""
        entity = EntitySchema(
            name="User",
            fields=[
                EntityField(name="email", type=FieldType.EMAIL, required=True),
                EntityField(name="name", type=FieldType.STRING, description="Full name"),
                EntityField(name="age", type=FieldType.INTEGER),
            ],
        )
        code = entity.to_pydantic()
        assert "class User(BaseModel):" in code
        assert "email: str" in code
        assert "Optional[str] = None" in code
        assert "Optional[int] = None" in code

    def test_typescript_generation(self):
        """Entity must generate valid TypeScript interface."""
        entity = EntitySchema(
            name="Product",
            fields=[
                EntityField(name="price", type=FieldType.FLOAT, required=True),
            ],
        )
        code = entity.to_typescript()
        assert "export interface Product" in code
        assert "price: number" in code

    def test_empty_fields(self):
        """Entity with no fields must still generate code."""
        entity = EntitySchema(name="Empty")
        py = entity.to_pydantic()
        assert "class Empty(BaseModel):" in py
        ts = entity.to_typescript()
        assert "export interface Empty" in ts


class TestAppConfig:
    """Tests for AppConfig."""

    def test_save_load_json(self, tmp_path):
        """AppConfig must survive JSON roundtrip."""
        cfg = AppConfig(app_name="TestApp", api_prefix="/api/v1")
        entity = EntitySchema(
            name="User",
            fields=[
                EntityField(name="email", type=FieldType.EMAIL, required=True),
            ],
        )
        cfg.entities = [entity]

        fp = tmp_path / "config.json"
        cfg.save(str(fp))
        assert fp.exists()

        loaded = AppConfig.load(str(fp))
        assert loaded.app_name == "TestApp"
        assert len(loaded.entities) == 1
        assert loaded.entities[0].name == "User"

    def test_generate_all_types(self):
        """AppConfig must generate all entity types."""
        cfg = AppConfig(app_name="Test")
        cfg.entities = [
            EntitySchema(
                name="User", fields=[EntityField(name="email", type=FieldType.EMAIL, required=True)]
            ),
            EntitySchema(
                name="Post",
                fields=[EntityField(name="title", type=FieldType.STRING, required=True)],
            ),
        ]
        types = cfg.generate_all_types()
        assert "User_pydantic.py" in types
        assert "User.ts" in types
        assert "Post_pydantic.py" in types
        assert "Post.ts" in types

    def test_default_config(self):
        """Default AppConfig must be valid."""
        cfg = AppConfig()
        assert cfg.app_name == ""
        assert cfg.version == "0.1.0"
        assert cfg.auth.method == AuthMethod.JWT


class TestDynamicTypeGenerator:
    """Tests for DynamicTypeGenerator."""

    def _make_entity(self, name, fields):
        return type(
            "E",
            (),
            {
                "name": name,
                "fields": [
                    type("F", (), {"name": n, "type": t, "required": r}) for n, t, r in fields
                ],
            },
        )()

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (TargetLanguage.PYTHON, ".py"),
            (TargetLanguage.TYPESCRIPT, ".ts"),
            (TargetLanguage.SQL, ".sql"),
            (TargetLanguage.GRAPHQL, ".graphql"),
        ],
    )
    def test_all_languages(self, lang, ext):
        """All target languages must produce valid output."""
        gen = DynamicTypeGenerator()
        entity = self._make_entity("User", [("email", "email", True)])
        output = gen.generate(entity, lang)
        assert output.language == lang
        assert len(output.code) > 0
        assert output.filename.endswith(ext)

    def test_sql_generation_includes_create_table(self):
        """SQL generation must include CREATE TABLE."""
        gen = DynamicTypeGenerator()
        entity = self._make_entity(
            "Order",
            [
                ("amount", "float", True),
                ("status", "string", False),
            ],
        )
        output = gen.generate(entity, TargetLanguage.SQL)
        assert "CREATE TABLE" in output.code

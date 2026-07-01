"""
Input Validation Generator — Builder + Visitor Pattern
=======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Input validation schema generation using Builder Pattern for schema construction
and Visitor Pattern for different validation library outputs.

Paradigm: OOP with Functional utilities
Patterns: Builder, Visitor, Factory Method

Usage:
    from orchestrator.safety.input_validation import SchemaBuilder, ZodSchemaVisitor

    schema = (SchemaBuilder()
        .add_string("email", min_length=1, max_length=255, pattern=r'^[^@]+@[^@]+[.][^@]+$')
        .add_string("password", min_length=8)
        .add_number("age", min_value=18, max_value=120)
        .build(ZodSchemaVisitor()))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

# ═══════════════════════════════════════════════════════════════════
# SCHEMA ELEMENTS (AST Nodes)
# ═══════════════════════════════════════════════════════════════════


class SchemaElementType(str, Enum):
    """Schema element types."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    EMAIL = "email"
    URL = "url"
    DATE = "date"
    ARRAY = "array"
    OBJECT = "object"
    ENUM = "enum"


@dataclass(frozen=True)
class SchemaElement(ABC):
    """Abstract base class for schema elements."""

    name: str
    element_type: SchemaElementType
    required: bool = True
    description: Optional[str] = None


@dataclass(frozen=True)
class StringField(SchemaElement):
    """String field schema element."""

    name: str
    element_type: SchemaElementType = SchemaElementType.STRING
    min_length: int = 0
    max_length: int = 255
    pattern: Optional[str] = None
    enum_values: Optional[List[str]] = None
    trim: bool = True
    lowercase: bool = False
    uppercase: bool = False


@dataclass(frozen=True)
class NumberField(SchemaElement):
    """Number field schema element."""

    name: str
    element_type: SchemaElementType = SchemaElementType.NUMBER
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    integer_only: bool = False
    multiple_of: Optional[float] = None


@dataclass(frozen=True)
class EmailField(SchemaElement):
    """Email field schema element."""

    name: str
    element_type: SchemaElementType = SchemaElementType.EMAIL
    min_length: int = 5
    max_length: int = 255


@dataclass(frozen=True)
class URLField(SchemaElement):
    """URL field schema element."""

    name: str
    element_type: SchemaElementType = SchemaElementType.URL
    protocols: List[str] = field(default_factory=lambda: ["http", "https"])


@dataclass(frozen=True)
class BooleanField(SchemaElement):
    """Boolean field schema element."""

    name: str
    element_type: SchemaElementType = SchemaElementType.BOOLEAN
    default: bool = False


@dataclass(frozen=True)
class DateField(SchemaElement):
    """Date field schema element."""

    name: str
    element_type: SchemaElementType = SchemaElementType.DATE
    min_date: Optional[str] = None
    max_date: Optional[str] = None


@dataclass(frozen=True)
class ArrayField(SchemaElement):
    """Array field schema element."""

    name: str
    element_type: SchemaElementType = SchemaElementType.ARRAY
    item_type: SchemaElementType = SchemaElementType.STRING
    min_items: int = 0
    max_items: Optional[int] = None
    unique_items: bool = False


@dataclass(frozen=True)
class ObjectField(SchemaElement):
    """Object field schema element."""

    name: str
    element_type: SchemaElementType = SchemaElementType.OBJECT
    properties: List[SchemaElement] = field(default_factory=list)


@dataclass(frozen=True)
class EnumField(SchemaElement):
    """Enum field schema element."""

    name: str
    element_type: SchemaElementType = SchemaElementType.ENUM
    values: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# VISITOR PATTERN — SCHEMA VISITOR INTERFACE
# ═══════════════════════════════════════════════════════════════════


class SchemaVisitor(ABC):
    """
    Visitor Pattern for different validation libraries.

    Subclasses implement schema generation for specific libraries.
    """

    @abstractmethod
    def visit_string(self, field: StringField) -> str:
        """Visit string field."""
        pass

    @abstractmethod
    def visit_number(self, field: NumberField) -> str:
        """Visit number field."""
        pass

    @abstractmethod
    def visit_email(self, field: EmailField) -> str:
        """Visit email field."""
        pass

    @abstractmethod
    def visit_url(self, field: URLField) -> str:
        """Visit URL field."""
        pass

    @abstractmethod
    def visit_boolean(self, field: BooleanField) -> str:
        """Visit boolean field."""
        pass

    @abstractmethod
    def visit_date(self, field: DateField) -> str:
        """Visit date field."""
        pass

    @abstractmethod
    def visit_array(self, field: ArrayField) -> str:
        """Visit array field."""
        pass

    @abstractmethod
    def visit_object(self, field: ObjectField) -> str:
        """Visit object field."""
        pass

    @abstractmethod
    def visit_enum(self, field: EnumField) -> str:
        """Visit enum field."""
        pass


# ═══════════════════════════════════════════════════════════════════
# CONCRETE VISITORS
# ═══════════════════════════════════════════════════════════════════


class ZodSchemaVisitor(SchemaVisitor):
    """
    Visitor for Zod (TypeScript) schemas.

    Generates Zod validation schemas.
    """

    def visit_string(self, field: StringField) -> str:
        """Generate Zod string schema."""
        schema = "z.string()"

        if field.min_length > 0:
            schema += f".min({field.min_length})"
        if field.max_length < 255:
            schema += f".max({field.max_length})"
        if field.pattern:
            # Escape backslashes for JavaScript regex
            escaped_pattern = field.pattern.replace("\\", "\\\\")
            schema += f".regex({escaped_pattern})"
        if field.enum_values:
            values_str = ", ".join(f'"{v}"' for v in field.enum_values)
            schema = f"z.enum([{values_str}])"

        if field.trim:
            schema += ".trim()"
        if field.lowercase:
            schema += ".toLowerCase()"
        if field.uppercase:
            schema += ".toUpperCase()"

        if not field.required:
            schema += ".optional()"

        return schema

    def visit_number(self, field: NumberField) -> str:
        """Generate Zod number schema."""
        schema = "z.number()"

        if field.min_value is not None:
            schema += f".min({field.min_value})"
        if field.field.max_value is not None:
            schema += f".max({field.max_value})"
        if field.integer_only:
            schema += ".int()"
        if field.multiple_of is not None:
            schema += f".multipleOf({field.multiple_of})"

        if not field.required:
            schema += ".optional()"

        return schema

    def visit_email(self, field: EmailField) -> str:
        """Generate Zod email schema."""
        schema = "z.string().email()"

        if field.min_length > 0:
            schema += f".min({field.min_length})"
        if field.max_length < 255:
            schema += f".max({field.max_length})"

        if not field.required:
            schema += ".optional()"

        return schema

    def visit_url(self, field: URLField) -> str:
        """Generate Zod URL schema."""
        schema = "z.string().url()"

        if field.protocols:
            # Zod doesn't have built-in protocol filtering
            # Would need custom refinement
            pass

        if not field.required:
            schema += ".optional()"

        return schema

    def visit_boolean(self, field: BooleanField) -> str:
        """Generate Zod boolean schema."""
        schema = "z.boolean()"

        if field.default is not None:
            schema += f".default({str(field.default).lower()})"

        if not field.required:
            schema += ".optional()"

        return schema

    def visit_date(self, field: DateField) -> str:
        """Generate Zod date schema."""
        schema = "z.date()"

        if field.min_date:
            schema += f".min(new Date('{field.min_date}'))"
        if field.max_date:
            schema += f".max(new Date('{field.max_date}'))"

        if not field.required:
            schema += ".optional()"

        return schema

    def visit_array(self, field: ArrayField) -> str:
        """Generate Zod array schema."""
        item_schema = f"z.{field.item_type.value}()"
        schema = f"z.array({item_schema})"

        if field.min_items > 0:
            schema += f".min({field.min_items})"
        if field.max_items is not None:
            schema += f".max({field.max_items})"
        if field.unique_items:
            # Custom refinement for unique items
            pass

        if not field.required:
            schema += ".optional()"

        return schema

    def visit_object(self, field: ObjectField) -> str:
        """Generate Zod object schema."""
        properties = {}
        for prop in field.properties:
            visitor = ZodSchemaVisitor()
            properties[prop.name] = getattr(visitor, f"visit_{prop.element_type.value}")(prop)

        # Build object schema string
        props_str = ", ".join(f'"{k}": {v}' for k, v in properties.items())
        schema = f"z.object({{{props_str}}})"

        if not field.required:
            schema += ".optional()"

        return schema

    def visit_enum(self, field: EnumField) -> str:
        """Generate Zod enum schema."""
        values_str = ", ".join(f'"{v}"' for v in field.values)
        return f"z.enum([{values_str}])"


class JoiSchemaVisitor(SchemaVisitor):
    """
    Visitor for Joi (JavaScript) schemas.

    Generates Joi validation schemas.
    """

    def visit_string(self, field: StringField) -> str:
        """Generate Joi string schema."""
        schema = "Joi.string()"

        if field.min_length > 0:
            schema += f".min({field.min_length})"
        if field.max_length < 255:
            schema += f".max({field.max_length})"
        if field.pattern:
            schema += f".pattern({field.pattern})"
        if field.enum_values:
            values_str = ", ".join(f'"{v}"' for v in field.enum_values)
            schema += f".valid({values_str})"

        if not field.required:
            schema += ".optional()"
        else:
            schema += ".required()"

        return schema

    def visit_number(self, field: NumberField) -> str:
        """Generate Joi number schema."""
        schema = "Joi.number()"

        if field.min_value is not None:
            schema += f".min({field.min_value})"
        if field.max_value is not None:
            schema += f".max({field.max_value})"
        if field.integer_only:
            schema += ".integer()"
        if field.multiple_of is not None:
            schema += f".multiple({field.multiple_of})"

        if not field.required:
            schema += ".optional()"
        else:
            schema += ".required()"

        return schema

    def visit_email(self, field: EmailField) -> str:
        """Generate Joi email schema."""
        schema = "Joi.string().email()"

        if field.min_length > 0:
            schema += f".min({field.min_length})"
        if field.max_length < 255:
            schema += f".max({field.max_length})"

        if not field.required:
            schema += ".optional()"
        else:
            schema += ".required()"

        return schema

    def visit_url(self, field: URLField) -> str:
        """Generate Joi URL schema."""
        schema = "Joi.string().uri()"

        if field.protocols:
            # Joi supports scheme option
            schema += f".domain({{ scheme: {field.protocols} }})"

        if not field.required:
            schema += ".optional()"
        else:
            schema += ".required()"

        return schema

    def visit_boolean(self, field: BooleanField) -> str:
        """Generate Joi boolean schema."""
        schema = "Joi.boolean()"

        if field.default is not None:
            schema += f".default({str(field.default).lower()})"

        if not field.required:
            schema += ".optional()"
        else:
            schema += ".required()"

        return schema

    def visit_date(self, field: DateField) -> str:
        """Generate Joi date schema."""
        schema = "Joi.date()"

        if field.min_date:
            schema += f".min('{field.min_date}')"
        if field.max_date:
            schema += f".max('{field.max_date}')"

        if not field.required:
            schema += ".optional()"
        else:
            schema += ".required()"

        return schema

    def visit_array(self, field: ArrayField) -> str:
        """Generate Joi array schema."""
        item_schema = f"Joi.{field.item_type.value}()"
        schema = f"Joi.array().items({item_schema})"

        if field.min_items > 0:
            schema += f".min({field.min_items})"
        if field.max_items is not None:
            schema += f".max({field.max_items})"
        if field.unique_items:
            schema += ".unique()"

        if not field.required:
            schema += ".optional()"
        else:
            schema += ".required()"

        return schema

    def visit_object(self, field: ObjectField) -> str:
        """Generate Joi object schema."""
        properties = {}
        for prop in field.properties:
            visitor = JoiSchemaVisitor()
            properties[prop.name] = getattr(visitor, f"visit_{prop.element_type.value}")(prop)

        props_str = ", ".join(f"{k}: {v}" for k, v in properties.items())
        schema = f"Joi.object({{{props_str}}})"

        if not field.required:
            schema += ".optional()"
        else:
            schema += ".required()"

        return schema

    def visit_enum(self, field: EnumField) -> str:
        """Generate Joi enum schema."""
        values_str = ", ".join(f'"{v}"' for v in field.values)
        schema = f"Joi.string().valid({values_str})"

        if not field.required:
            schema += ".optional()"
        else:
            schema += ".required()"

        return schema


class PydanticSchemaVisitor(SchemaVisitor):
    """
    Visitor for Pydantic (Python) schemas.

    Generates Pydantic model definitions.
    """

    def visit_string(self, field: StringField) -> str:
        """Generate Pydantic string field."""
        type_hint = "str"

        if not field.required:
            type_hint = f"Optional[{type_hint}]"

        # Add validators as comments
        validators = []
        if field.min_length > 0:
            validators.append(f"min_length={field.min_length}")
        if field.max_length < 255:
            validators.append(f"max_length={field.max_length}")
        if field.pattern:
            validators.append(f"pattern=r'{field.pattern}'")

        validator_str = f" # Field({', '.join(validators)})" if validators else ""

        return f"{field.name}: {type_hint}{validator_str}"

    def visit_number(self, field: NumberField) -> str:
        """Generate Pydantic number field."""
        type_hint = "int" if field.integer_only else "float"

        if not field.required:
            type_hint = f"Optional[{type_hint}]"

        validators = []
        if field.min_value is not None:
            validators.append(f"ge={field.min_value}")
        if field.max_value is not None:
            validators.append(f"le={field.max_value}")
        if field.multiple_of is not None:
            validators.append(f"multiple_of={field.multiple_of}")

        validator_str = f" # Field({', '.join(validators)})" if validators else ""

        return f"{field.name}: {type_hint}{validator_str}"

    def visit_email(self, field: EmailField) -> str:
        """Generate Pydantic email field."""
        type_hint = "EmailStr"

        if not field.required:
            type_hint = f"Optional[{type_hint}]"

        return f"{field.name}: {type_hint}"

    def visit_url(self, field: URLField) -> str:
        """Generate Pydantic URL field."""
        type_hint = "HttpUrl"

        if not field.required:
            type_hint = f"Optional[{type_hint}]"

        return f"{field.name}: {type_hint}"

    def visit_boolean(self, field: BooleanField) -> str:
        """Generate Pydantic boolean field."""
        type_hint = "bool"

        if not field.required:
            type_hint = f"Optional[{type_hint}]"

        default_str = f" = {str(field.default).lower()}" if field.default is not None else ""

        return f"{field.name}: {type_hint}{default_str}"

    def visit_date(self, field: DateField) -> str:
        """Generate Pydantic date field."""
        type_hint = "date"

        if not field.required:
            type_hint = f"Optional[{type_hint}]"

        return f"{field.name}: {type_hint}"

    def visit_array(self, field: ArrayField) -> str:
        """Generate Pydantic array field."""
        item_type = field.item_type.value.capitalize()
        type_hint = f"List[{item_type}]"

        if not field.required:
            type_hint = f"Optional[{type_hint}]"

        validators = []
        if field.min_items > 0:
            validators.append(f"min_items={field.min_items}")
        if field.max_items is not None:
            validators.append(f"max_items={field.max_items}")

        validator_str = f" # Field({', '.join(validators)})" if validators else ""

        return f"{field.name}: {type_hint}{validator_str}"

    def visit_object(self, field: ObjectField) -> str:
        """Generate Pydantic object field."""
        # Generate nested model
        nested_model_name = f"{field.name.capitalize()}Model"

        properties = []
        for prop in field.properties:
            visitor = PydanticSchemaVisitor()
            properties.append(getattr(visitor, f"visit_{prop.element_type.value}")(prop))

        model_str = f"class {nested_model_name}(BaseModel):\n"
        model_str += "\n".join(f"    {p}" for p in properties)

        type_hint = nested_model_name

        if not field.required:
            type_hint = f"Optional[{type_hint}]"

        return f"{field.name}: {type_hint}  # {model_str}"

    def visit_enum(self, field: EnumField) -> str:
        """Generate Pydantic enum field."""
        enum_name = f"{field.name.capitalize()}Enum"

        enum_str = f"class {enum_name}(str, Enum):\n"
        for value in field.values:
            enum_str += f"    {value.upper()} = '{value}'\n"

        type_hint = enum_name

        if not field.required:
            type_hint = f"Optional[{type_hint}]"

        return f"{field.name}: {type_hint}  # {enum_str}"


# ═══════════════════════════════════════════════════════════════════
# BUILDER PATTERN — SCHEMA BUILDER
# ═══════════════════════════════════════════════════════════════════


class SchemaBuilder:
    """
    Builder Pattern for constructing validation schemas.

    Fluent interface for adding schema elements.

    Usage:
        builder = SchemaBuilder()
        builder.add_string("email", min_length=1, max_length=255)
        builder.add_number("age", min_value=18)
        schema = builder.build(ZodSchemaVisitor())
    """

    def __init__(self):
        """Initialize schema builder."""
        self._elements: List[SchemaElement] = []
        self._model_name: str = "Schema"

    def set_model_name(self, name: str) -> "SchemaBuilder":
        """Set model/class name (fluent interface)."""
        self._model_name = name
        return self

    def add_string(
        self,
        name: str,
        min_length: int = 0,
        max_length: int = 255,
        pattern: str = None,
        enum_values: List[str] = None,
        trim: bool = True,
        lowercase: bool = False,
        uppercase: bool = False,
        required: bool = True,
        description: str = None,
    ) -> "SchemaBuilder":
        """Add string field (fluent interface)."""
        self._elements.append(
            StringField(
                name=name,
                min_length=min_length,
                max_length=max_length,
                pattern=pattern,
                enum_values=enum_values,
                trim=trim,
                lowercase=lowercase,
                uppercase=uppercase,
                required=required,
                description=description,
            )
        )
        return self

    def add_number(
        self,
        name: str,
        min_value: float = None,
        max_value: float = None,
        integer_only: bool = False,
        multiple_of: float = None,
        required: bool = True,
        description: str = None,
    ) -> "SchemaBuilder":
        """Add number field (fluent interface)."""
        self._elements.append(
            NumberField(
                name=name,
                min_value=min_value,
                max_value=max_value,
                integer_only=integer_only,
                multiple_of=multiple_of,
                required=required,
                description=description,
            )
        )
        return self

    def add_email(
        self,
        name: str,
        min_length: int = 5,
        max_length: int = 255,
        required: bool = True,
        description: str = None,
    ) -> "SchemaBuilder":
        """Add email field (fluent interface)."""
        self._elements.append(
            EmailField(
                name=name,
                min_length=min_length,
                max_length=max_length,
                required=required,
                description=description,
            )
        )
        return self

    def add_url(
        self,
        name: str,
        protocols: List[str] = None,
        required: bool = True,
        description: str = None,
    ) -> "SchemaBuilder":
        """Add URL field (fluent interface)."""
        self._elements.append(
            URLField(
                name=name,
                protocols=protocols or ["http", "https"],
                required=required,
                description=description,
            )
        )
        return self

    def add_boolean(
        self,
        name: str,
        default: bool = False,
        required: bool = True,
        description: str = None,
    ) -> "SchemaBuilder":
        """Add boolean field (fluent interface)."""
        self._elements.append(
            BooleanField(
                name=name,
                default=default,
                required=required,
                description=description,
            )
        )
        return self

    def add_date(
        self,
        name: str,
        min_date: str = None,
        max_date: str = None,
        required: bool = True,
        description: str = None,
    ) -> "SchemaBuilder":
        """Add date field (fluent interface)."""
        self._elements.append(
            DateField(
                name=name,
                min_date=min_date,
                max_date=max_date,
                required=required,
                description=description,
            )
        )
        return self

    def add_array(
        self,
        name: str,
        item_type: SchemaElementType = SchemaElementType.STRING,
        min_items: int = 0,
        max_items: int = None,
        unique_items: bool = False,
        required: bool = True,
        description: str = None,
    ) -> "SchemaBuilder":
        """Add array field (fluent interface)."""
        self._elements.append(
            ArrayField(
                name=name,
                item_type=item_type,
                min_items=min_items,
                max_items=max_items,
                unique_items=unique_items,
                required=required,
                description=description,
            )
        )
        return self

    def build(self, visitor: SchemaVisitor) -> str:
        """
        Build schema using Visitor pattern.

        Args:
            visitor: Schema visitor (ZodSchemaVisitor, JoiSchemaVisitor, etc.)

        Returns:
            Generated schema string
        """
        schema_parts = []

        for element in self._elements:
            method_name = f"visit_{element.element_type.value}"
            method = getattr(visitor, method_name)
            schema_parts.append(method(element))

        return "\n".join(schema_parts)

    def build_zod(self) -> str:
        """Build Zod schema."""
        return self.build(ZodSchemaVisitor())

    def build_joi(self) -> str:
        """Build Joi schema."""
        return self.build(JoiSchemaVisitor())

    def build_pydantic(self) -> str:
        """Build Pydantic schema."""
        return self.build(PydanticSchemaVisitor())


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def create_login_schema(library: str = "zod") -> str:
    """
    Create login validation schema.

    Args:
        library: Validation library ("zod", "joi", "pydantic")

    Returns:
        Generated schema
    """
    builder = SchemaBuilder().set_model_name("LoginSchema")
    builder.add_email("email", required=True)
    builder.add_string("password", min_length=8, required=True)

    if library == "zod":
        return builder.build_zod()
    elif library == "joi":
        return builder.build_joi()
    else:
        return builder.build_pydantic()


def create_register_schema(library: str = "zod") -> str:
    """
    Create registration validation schema.

    Args:
        library: Validation library

    Returns:
        Generated schema
    """
    builder = SchemaBuilder().set_model_name("RegisterSchema")
    builder.add_string("name", min_length=2, max_length=100, required=True)
    builder.add_email("email", required=True)
    builder.add_string(
        "password",
        min_length=8,
        max_length=128,
        pattern=r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$",
        required=True,
        description="Must contain uppercase, lowercase, and digit",
    )
    builder.add_string("password_confirm", required=True)

    if library == "zod":
        return builder.build_zod()
    elif library == "joi":
        return builder.build_joi()
    else:
        return builder.build_pydantic()

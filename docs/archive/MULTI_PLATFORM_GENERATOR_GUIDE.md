# Multi-Platform Output Generator Guide

**Version:** 1.0.0 | **Updated:** 2026-03-25 | **Author:** Georgios-Chrysovalantis Chatzivantsidis

> **Generate full-stack, cross-platform applications** from a single project description. Supports Python, React, React Native, SwiftUI, Kotlin, FastAPI, and full-stack combinations.

---

## Quick Start

### Basic Usage

```python
from orchestrator.multi_platform_generator import (
    generate_multi_platform,
    OutputTarget,
    ProjectOutputConfig,
)

# Generate React web app + FastAPI backend
result = await generate_multi_platform(
    project_description="Build a todo application with user authentication",
    config=ProjectOutputConfig(
        targets=[
            OutputTarget.REACT_WEB_APP,
            OutputTarget.FASTAPI_BACKEND,
        ],
        include_auth=True,
        include_database=True,
        use_typescript=True,
    ),
    project_name="my_todo_app",
)

print(result.summary())
# Project: my_todo_app
# Platforms: react, fastapi
# Total files: 24
# Shared files: 5
```

### Single Platform

```python
# Generate SwiftUI iOS app only
result = await generate_multi_platform(
    project_description="Build a fitness tracking app",
    config=ProjectOutputConfig(
        targets=[OutputTarget.SWIFTUI_IOS],
        ios_deployment=True,
        include_privacy_policy=True,
        hig_compliance=True,
    ),
)
```

---

## Table of Contents

1. [Output Targets](#output-targets)
2. [Configuration Options](#configuration-options)
3. [Platform Generators](#platform-generators)
4. [Examples](#examples)
5. [App Store Compliance](#app-store-compliance)
6. [API Reference](#api-reference)

---

## Output Targets

### Available Platforms

| Target | Value | Description |
|--------|-------|-------------|
| `PYTHON_LIBRARY` | `"python"` | Python library/package |
| `REACT_WEB_APP` | `"react"` | React + Next.js web app |
| `REACT_NATIVE_MOBILE` | `"react_native"` | React Native (iOS + Android) |
| `SWIFTUI_IOS` | `"swiftui"` | Native iOS with SwiftUI |
| `KOTLIN_ANDROID` | `"kotlin"` | Native Android with Kotlin |
| `FASTAPI_BACKEND` | `"fastapi"` | FastAPI REST API backend |
| `FLASK_BACKEND` | `"flask"` | Simple Flask backend |
| `FULL_STACK` | `"full_stack"` | Frontend + Backend + Database |
| `PWA` | `"pwa"` | Progressive Web App |

### Platform Combinations

```python
# Mobile + Backend
config = ProjectOutputConfig(
    targets=[
        OutputTarget.REACT_NATIVE_MOBILE,
        OutputTarget.FASTAPI_BACKEND,
    ],
)

# Web + Backend + Database
config = ProjectOutputConfig(
    targets=[
        OutputTarget.REACT_WEB_APP,
        OutputTarget.FASTAPI_BACKEND,
    ],
    include_database=True,
)

# Full-stack (all-in-one)
config = ProjectOutputConfig(
    targets=[OutputTarget.FULL_STACK],
)

# iOS + Android + Backend
config = ProjectOutputConfig(
    targets=[
        OutputTarget.SWIFTUI_IOS,
        OutputTarget.KOTLIN_ANDROID,
        OutputTarget.FASTAPI_BACKEND,
    ],
)
```

---

## Configuration Options

### ProjectOutputConfig

```python
@dataclass
class ProjectOutputConfig:
    targets: List[OutputTarget]
    
    # Deployment
    ios_deployment: bool = False
    android_deployment: bool = False
    web_deployment: bool = False
    
    # App Store compliance
    include_privacy_policy: bool = True
    include_app_store_assets: bool = True
    hig_compliance: bool = True
    
    # Code quality
    include_tests: bool = True
    include_documentation: bool = True
    include_ci_cd: bool = True
    
    # Database
    include_database: bool = True
    database_type: str = "sqlite"
    
    # Authentication
    include_auth: bool = True
    auth_type: str = "jwt"
    
    # Styling
    styling: str = "tailwind"
    
    # TypeScript
    use_typescript: bool = True
    
    # Additional
    extra_packages: List[str] = field(default_factory=list)
    custom_config: Dict[str, Any] = field(default_factory=dict)
```

### Configuration Examples

#### Production-Ready Web App

```python
config = ProjectOutputConfig(
    targets=[OutputTarget.REACT_WEB_APP, OutputTarget.FASTAPI_BACKEND],
    
    # Features
    include_auth=True,
    include_database=True,
    database_type="postgresql",
    
    # Quality
    include_tests=True,
    include_ci_cd=True,
    use_typescript=True,
    
    # Styling
    styling="tailwind",
)
```

#### App Store iOS App

```python
config = ProjectOutputConfig(
    targets=[OutputTarget.SWIFTUI_IOS],
    
    # Deployment
    ios_deployment=True,
    
    # Compliance
    include_privacy_policy=True,
    include_app_store_assets=True,
    hig_compliance=True,
    
    # Quality
    include_tests=True,
    include_documentation=True,
)
```

#### Mobile App with Backend

```python
config = ProjectOutputConfig(
    targets=[
        OutputTarget.REACT_NATIVE_MOBILE,
        OutputTarget.FASTAPI_BACKEND,
    ],
    
    # Features
    include_auth=True,
    include_database=True,
    
    # Mobile
    ios_deployment=True,
    android_deployment=True,
    
    # Quality
    use_typescript=True,
    include_tests=True,
)
```

---

## Platform Generators

### React/Next.js Web App

**Generated Structure:**
```
├── app/
│   ├── page.tsx          # Homepage
│   ├── layout.tsx        # Root layout
│   └── globals.css       # Global styles
├── components/
│   └── Header.tsx        # Header component
├── public/               # Static assets
├── package.json
├── next.config.js
├── tsconfig.json
├── tailwind.config.js
└── __tests__/
    └── page.test.tsx
```

**Features:**
- Next.js 14 App Router
- TypeScript support
- Tailwind CSS styling
- Jest testing
- SEO-optimized

### React Native Mobile App

**Generated Structure:**
```
├── App.tsx               # Main app component
├── package.json
├── tsconfig.json
├── metro.config.js
├── app.json
└── __tests__/
    └── App.test.tsx
```

**Features:**
- Cross-platform (iOS + Android)
- Native components
- TypeScript support
- Hot reloading ready

### SwiftUI iOS App

**Generated Structure:**
```
├── MyAppApp.swift        # App entry point
├── MyApp/
│   ├── ContentView.swift # Main view
│   ├── Info.plist        # App configuration
│   └── Assets.xcassets/  # App assets
└── PrivacyPolicy.md      # Privacy policy
```

**Features:**
- SwiftUI framework
- iOS 17+ support
- HIG compliant
- Privacy policy included
- Dark Mode support

### Kotlin Android App

**Generated Structure:**
```
├── app/
│   ├── build.gradle.kts
│   └── src/main/
│       ├── AndroidManifest.xml
│       └── java/com/example/myapp/
│           └── MainActivity.kt
```

**Features:**
- Jetpack Compose UI
- Material Design 3
- Kotlin coroutines
- Modern Android architecture

### FastAPI Backend

**Generated Structure:**
```
├── main.py               # FastAPI app
├── models.py             # Pydantic models
├── requirements.txt
└── tests/
    └── test_main.py
```

**Features:**
- Automatic OpenAPI docs
- Async support
- Pydantic validation
- CORS middleware
- Optional auth & database

### Full-Stack Application

**Generated Structure:**
```
├── frontend/             # React/Next.js
├── backend/              # FastAPI
├── docker-compose.yml    # Docker orchestration
└── README.md
```

**Features:**
- Complete stack
- Docker ready
- API integration
- Database included

---

## Examples

### Example 1: Todo App (Web + Backend)

```python
from orchestrator.multi_platform_generator import (
    generate_multi_platform,
    OutputTarget,
    ProjectOutputConfig,
)

result = await generate_multi_platform(
    project_description="Build a todo application with user accounts and task sharing",
    config=ProjectOutputConfig(
        targets=[
            OutputTarget.REACT_WEB_APP,
            OutputTarget.FASTAPI_BACKEND,
        ],
        include_auth=True,
        include_database=True,
        database_type="postgresql",
        use_typescript=True,
    ),
    project_name="todo_app",
)

# Access generated files
for target, output in result.outputs.items():
    print(f"\n{target.value}:")
    for file in output.files:
        print(f"  - {file.path}")
```

### Example 2: Fitness Tracker (iOS + Android + Backend)

```python
result = await generate_multi_platform(
    project_description="Fitness tracking app with workout logging and progress charts",
    config=ProjectOutputConfig(
        targets=[
            OutputTarget.SWIFTUI_IOS,
            OutputTarget.KOTLIN_ANDROID,
            OutputTarget.FASTAPI_BACKEND,
        ],
        ios_deployment=True,
        android_deployment=True,
        include_auth=True,
        include_database=True,
        include_privacy_policy=True,
    ),
    project_name="fitness_tracker",
)
```

### Example 3: E-commerce PWA

```python
result = await generate_multi_platform(
    project_description="E-commerce store with product catalog and shopping cart",
    config=ProjectOutputConfig(
        targets=[OutputTarget.PWA],
        include_database=True,
        include_auth=True,
        styling="tailwind",
    ),
    project_name="ecommerce_store",
)
```

### Example 4: Full-Stack SaaS

```python
result = await generate_multi_platform(
    project_description="SaaS platform with subscription billing and dashboard",
    config=ProjectOutputConfig(
        targets=[OutputTarget.FULL_STACK],
        include_auth=True,
        include_database=True,
        database_type="postgresql",
        include_ci_cd=True,
        use_typescript=True,
    ),
    project_name="saas_platform",
)
```

---

## App Store Compliance

### iOS App Store

The SwiftUI generator includes:
- ✅ Privacy policy URL in Info.plist
- ✅ HIG-compliant UI components
- ✅ Dark Mode support
- ✅ Accessibility features
- ✅ App Store-ready structure

```python
config = ProjectOutputConfig(
    targets=[OutputTarget.SWIFTUI_IOS],
    ios_deployment=True,
    include_privacy_policy=True,
    hig_compliance=True,
    include_app_store_assets=True,
)
```

### Google Play Store

The Kotlin generator includes:
- ✅ Privacy policy
- ✅ Material Design
- ✅ Modern Android architecture
- ✅ Play Store-ready structure

```python
config = ProjectOutputConfig(
    targets=[OutputTarget.KOTLIN_ANDROID],
    android_deployment=True,
    include_privacy_policy=True,
)
```

### Integration with App Store Validator

```python
from orchestrator.app_store_validator import validate_app_store_compliance

# Generate iOS app
result = await generate_multi_platform(
    project_description="Build an iOS app",
    config=ProjectOutputConfig(
        targets=[OutputTarget.SWIFTUI_IOS],
        ios_deployment=True,
    ),
)

# Validate for App Store
compliance_result = await validate_app_store_compliance(
    project_path=Path("./generated"),
    platform=AppStorePlatform.IOS,
)

if not compliance_result.passed:
    print(f"Violations: {compliance_result.violations}")
```

---

## API Reference

### OutputTarget

```python
class OutputTarget(str, Enum):
    PYTHON_LIBRARY = "python"
    REACT_WEB_APP = "react"
    REACT_NATIVE_MOBILE = "react_native"
    SWIFTUI_IOS = "swiftui"
    KOTLIN_ANDROID = "kotlin"
    FASTAPI_BACKEND = "fastapi"
    FLASK_BACKEND = "flask"
    FULL_STACK = "full_stack"
    PWA = "pwa"
```

### ProjectOutputConfig

```python
@dataclass
class ProjectOutputConfig:
    """Configuration for project output generation."""
    targets: List[OutputTarget]
    ios_deployment: bool = False
    android_deployment: bool = False
    web_deployment: bool = False
    include_privacy_policy: bool = True
    include_app_store_assets: bool = True
    hig_compliance: bool = True
    include_tests: bool = True
    include_documentation: bool = True
    include_ci_cd: bool = True
    include_database: bool = True
    database_type: str = "sqlite"
    include_auth: bool = True
    auth_type: str = "jwt"
    styling: str = "tailwind"
    use_typescript: bool = True
    extra_packages: List[str] = field(default_factory=list)
    custom_config: Dict[str, Any] = field(default_factory=dict)
```

### MultiPlatformGenerator

```python
class MultiPlatformGenerator:
    """Generate multi-platform output."""
    
    def __init__(self, output_dir: Optional[Path] = None)
    
    async def generate(
        self,
        project_description: str,
        config: Optional[ProjectOutputConfig] = None,
        project_name: Optional[str] = None,
    ) -> MultiPlatformResult:
        """Generate multi-platform output."""
```

### MultiPlatformResult

```python
@dataclass
class MultiPlatformResult:
    """Result of multi-platform generation."""
    project_name: str
    project_description: str
    outputs: Dict[OutputTarget, PlatformOutput]
    shared_files: List[GeneratedFile]
    config: Optional[ProjectOutputConfig]
    metadata: Dict[str, Any]
    
    @property
    def total_files(self) -> int
    def summary(self) -> str
```

### Convenience Function

```python
async def generate_multi_platform(
    project_description: str,
    config: Optional[ProjectOutputConfig] = None,
    project_name: Optional[str] = None,
) -> MultiPlatformResult:
    """Convenience function for multi-platform generation."""
```

---

## Testing

### Run Tests

```bash
# Run all multi-platform generator tests
pytest tests/test_multi_platform_generator.py -v

# Run specific test category
pytest tests/test_multi_platform_generator.py::TestMultiPlatformGenerator -v

# Run with coverage
pytest tests/test_multi_platform_generator.py --cov=orchestrator.multi_platform_generator
```

### Test Results

```
============================= 28 passed in 11.01s =============================
```

---

## Related Documentation

- [APP_STORE_VALIDATOR_GUIDE.md](./APP_STORE_VALIDATOR_GUIDE.md) — App Store compliance validation
- [USAGE_GUIDE.md](./USAGE_GUIDE.md) — Main usage guide
- [CAPABILITIES.md](./CAPABILITIES.md) — Full capabilities overview

---

**License:** MIT | **Author:** Georgios-Chrysovalantis Chatzivantsidis

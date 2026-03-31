"""
Multi-Platform Output Generator
================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Generates cross-platform output from AI Orchestrator projects.
Supports: Python, React, React Native, SwiftUI, FastAPI, Full-Stack

Usage:
    from orchestrator.multi_platform_generator import (
        MultiPlatformGenerator,
        OutputTarget,
        ProjectOutputConfig,
    )

    generator = MultiPlatformGenerator()
    result = await generator.generate(
        project_description="Build a todo app",
        config=ProjectOutputConfig(
            targets=[OutputTarget.REACT_NATIVE_MOBILE, OutputTarget.FASTAPI_BACKEND],
            include_privacy_policy=True,
        ),
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .log_config import get_logger

logger = get_logger(__name__)


class OutputTarget(str, Enum):
    """Target output platform/type."""

    PYTHON_LIBRARY = "python"  # Current behavior
    REACT_WEB_APP = "react"  # React + Next.js
    REACT_NATIVE_MOBILE = "react_native"  # iOS + Android
    SWIFTUI_IOS = "swiftui"  # Native iOS
    KOTLIN_ANDROID = "kotlin"  # Native Android
    FASTAPI_BACKEND = "fastapi"  # Backend API
    FLASK_BACKEND = "flask"  # Simple backend
    FULL_STACK = "full_stack"  # Frontend + Backend + Database
    PWA = "pwa"  # Progressive Web App


class DeploymentTarget(str, Enum):
    """Deployment target platform."""

    APP_STORE = "app_store"  # Apple App Store
    PLAY_STORE = "play_store"  # Google Play Store
    WEB = "web"  # Web deployment
    DESKTOP = "desktop"  # Desktop (Mac, Windows, Linux)


@dataclass
class ProjectOutputConfig:
    """Configuration for project output generation."""

    targets: list[OutputTarget] = field(default_factory=list)
    ios_deployment: bool = False
    android_deployment: bool = False
    web_deployment: bool = False

    # App Store compliance
    include_privacy_policy: bool = True
    include_app_store_assets: bool = True  # Screenshots, descriptions
    hig_compliance: bool = True  # Apple Human Interface Guidelines

    # Code quality
    include_tests: bool = True
    include_documentation: bool = True
    include_ci_cd: bool = True

    # Database
    include_database: bool = True
    database_type: str = "sqlite"  # sqlite, postgresql, mongodb

    # Authentication
    include_auth: bool = True
    auth_type: str = "jwt"  # jwt, oauth2, session

    # Styling
    styling: str = "tailwind"  # tailwind, bootstrap, material-ui, none

    # TypeScript
    use_typescript: bool = True

    # Additional options
    extra_packages: list[str] = field(default_factory=list)
    custom_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": [t.value for t in self.targets],
            "ios_deployment": self.ios_deployment,
            "android_deployment": self.android_deployment,
            "web_deployment": self.web_deployment,
            "include_privacy_policy": self.include_privacy_policy,
            "include_app_store_assets": self.include_app_store_assets,
            "hig_compliance": self.hig_compliance,
            "include_tests": self.include_tests,
            "include_documentation": self.include_documentation,
            "include_ci_cd": self.include_ci_cd,
            "include_database": self.include_database,
            "database_type": self.database_type,
            "include_auth": self.include_auth,
            "auth_type": self.auth_type,
            "styling": self.styling,
            "use_typescript": self.use_typescript,
            "extra_packages": self.extra_packages,
            "custom_config": self.custom_config,
        }


@dataclass
class GeneratedFile:
    """A generated file."""

    path: str
    content: str
    description: str = ""
    is_binary: bool = False


@dataclass
class PlatformOutput:
    """Output for a specific platform."""

    target: OutputTarget
    files: list[GeneratedFile] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    dev_dependencies: list[str] = field(default_factory=list)
    scripts: dict[str, str] = field(default_factory=dict)
    readme: str = ""

    def add_file(self, path: str, content: str, description: str = "") -> None:
        self.files.append(GeneratedFile(path=path, content=content, description=description))

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.value,
            "file_count": len(self.files),
            "dependencies": self.dependencies,
            "dev_dependencies": self.dev_dependencies,
            "scripts": self.scripts,
            "has_readme": bool(self.readme),
        }


@dataclass
class MultiPlatformResult:
    """Result of multi-platform generation."""

    project_name: str
    project_description: str
    outputs: dict[OutputTarget, PlatformOutput] = field(default_factory=dict)
    shared_files: list[GeneratedFile] = field(default_factory=list)
    config: ProjectOutputConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_files(self) -> int:
        total = len(self.shared_files)
        for output in self.outputs.values():
            total += len(output.files)
        return total

    def summary(self) -> str:
        platforms = ", ".join([t.value for t in self.outputs])
        return (
            f"Project: {self.project_name}\n"
            f"Platforms: {platforms}\n"
            f"Total files: {self.total_files}\n"
            f"Shared files: {len(self.shared_files)}"
        )


class MultiPlatformGenerator:
    """
    Generate multi-platform output from project specifications.

    Usage:
        generator = MultiPlatformGenerator()
        result = await generator.generate(
            project_description="Build a todo app",
            config=ProjectOutputConfig(
                targets=[OutputTarget.REACT_WEB_APP, OutputTarget.FASTAPI_BACKEND],
            ),
        )
    """

    def __init__(self, output_dir: Path | None = None):
        """
        Initialize generator.

        Args:
            output_dir: Base output directory
        """
        self.output_dir = output_dir or Path("./generated")
        self._generators = {
            OutputTarget.PYTHON_LIBRARY: self._generate_python,
            OutputTarget.REACT_WEB_APP: self._generate_react,
            OutputTarget.REACT_NATIVE_MOBILE: self._generate_react_native,
            OutputTarget.SWIFTUI_IOS: self._generate_swiftui,
            OutputTarget.KOTLIN_ANDROID: self._generate_kotlin,
            OutputTarget.FASTAPI_BACKEND: self._generate_fastapi,
            OutputTarget.FLASK_BACKEND: self._generate_flask,
            OutputTarget.FULL_STACK: self._generate_full_stack,
            OutputTarget.PWA: self._generate_pwa,
        }

    async def generate(
        self,
        project_description: str,
        config: ProjectOutputConfig | None = None,
        project_name: str | None = None,
    ) -> MultiPlatformResult:
        """
        Generate multi-platform output.

        Args:
            project_description: Project description
            config: Output configuration
            project_name: Optional project name

        Returns:
            MultiPlatformResult with all generated files
        """
        if config is None:
            config = ProjectOutputConfig(
                targets=[OutputTarget.PYTHON_LIBRARY],
            )

        if project_name is None:
            project_name = self._infer_project_name(project_description)

        result = MultiPlatformResult(
            project_name=project_name,
            project_description=project_description,
            config=config,
        )

        # Generate output for each target
        for target in config.targets:
            generator = self._generators.get(target)
            if generator:
                output = await generator(project_description, project_name, config)
                result.outputs[target] = output

        # Generate shared files
        result.shared_files = await self._generate_shared_files(project_name, config)

        # Add metadata
        result.metadata = {
            "generated_at": str(Path.cwd()),
            "config": config.to_dict(),
        }

        return result

    def _infer_project_name(self, description: str) -> str:
        """Infer project name from description."""
        # Simple heuristic: first few words
        words = description.lower().split()[:3]
        name = "_".join(words)
        # Remove non-alphanumeric
        name = "".join(c if c.isalnum() else "_" for c in name)
        return name[:50]  # Limit length

    async def _generate_shared_files(
        self,
        project_name: str,
        config: ProjectOutputConfig,
    ) -> list[GeneratedFile]:
        """Generate shared files (README, LICENSE, etc.)."""
        files = []

        # README.md
        readme_content = self._generate_readme(project_name, config)
        files.append(
            GeneratedFile(
                path="README.md",
                content=readme_content,
                description="Project README",
            )
        )

        # LICENSE
        license_content = self._generate_license()
        files.append(
            GeneratedFile(
                path="LICENSE",
                content=license_content,
                description="MIT License",
            )
        )

        # Privacy Policy (if requested)
        if config.include_privacy_policy:
            privacy_content = self._generate_privacy_policy(project_name)
            files.append(
                GeneratedFile(
                    path="PRIVACY_POLICY.md",
                    content=privacy_content,
                    description="Privacy Policy",
                )
            )

        # .gitignore
        gitignore_content = self._generate_gitignore(config)
        files.append(
            GeneratedFile(
                path=".gitignore",
                content=gitignore_content,
                description="Git ignore file",
            )
        )

        # CI/CD (if requested)
        if config.include_ci_cd:
            ci_content = self._generate_github_actions(config)
            files.append(
                GeneratedFile(
                    path=".github/workflows/ci.yml",
                    content=ci_content,
                    description="GitHub Actions CI/CD",
                )
            )

        return files

    def _generate_readme(self, project_name: str, config: ProjectOutputConfig) -> str:
        """Generate README.md content."""
        targets_str = ", ".join([t.value for t in config.targets])

        return f"""# {project_name}

Auto-generated project created by AI Orchestrator.

## Platforms

- **Targets:** {targets_str}
- **iOS Deployment:** {"Yes" if config.ios_deployment else "No"}
- **Android Deployment:** {"Yes" if config.android_deployment else "No"}
- **Web Deployment:** {"Yes" if config.web_deployment else "No"}

## Features

- {"✅ Tests included" if config.include_tests else "❌ No tests"}
- {"✅ Documentation included" if config.include_documentation else "❌ No documentation"}
- {"✅ CI/CD pipeline" if config.include_ci_cd else "❌ No CI/CD"}
- {"✅ Authentication" if config.include_auth else "❌ No authentication"}
- {"✅ Database" if config.include_database else "❌ No database"}

## Getting Started

### Prerequisites

- Node.js 18+ (for React/React Native)
- Python 3.10+ (for backend)
- Xcode 15+ (for iOS)
- Android Studio (for Android)

### Installation

```bash
# Install dependencies
npm install
pip install -r requirements.txt

# Run development server
npm run dev
```

## Project Structure

See individual platform folders for structure details.

## License

MIT License - see LICENSE file for details.
"""

    def _generate_license(self) -> str:
        """Generate MIT License content."""
        return """MIT License

Copyright (c) 2026 AI Orchestrator

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

    def _generate_privacy_policy(self, project_name: str) -> str:
        """Generate privacy policy template."""
        return f"""# Privacy Policy for {project_name}

**Last Updated:** 2026-03-25

## 1. Introduction

This privacy policy explains how {project_name} collects, uses, and protects your information.

## 2. Information We Collect

### 2.1 Personal Information
- Email address (if you create an account)
- Usage data

### 2.2 Automatically Collected Information
- Device information
- IP address
- App usage analytics

## 3. How We Use Your Information

- To provide and maintain the service
- To improve user experience
- To communicate with you

## 4. Data Sharing

We do not sell your personal information. We may share data with:
- Service providers (hosting, analytics)
- Legal authorities (if required by law)

## 5. Data Security

We implement appropriate security measures to protect your data.

## 6. Your Rights

You have the right to:
- Access your data
- Delete your account
- Export your data

## 7. Contact Us

For privacy-related questions, please contact: privacy@example.com

## 8. Changes to This Policy

We may update this policy. Continued use constitutes acceptance.
"""

    def _generate_gitignore(self, config: ProjectOutputConfig) -> str:
        """Generate .gitignore content."""
        gitignore = """# Dependencies
node_modules/
__pycache__/
*.pyc
*.pyo
.venv/
venv/

# Build outputs
dist/
build/
*.egg-info/
.next/
out/

# Environment
.env
.env.local
.env.production

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
npm-debug.log*

# Testing
coverage/
.coverage
.pytest_cache/

# Database
*.db
*.sqlite
*.sqlite3

# App Store
*.ipa
*.apk
*.aab
"""
        return gitignore

    def _generate_github_actions(self, config: ProjectOutputConfig) -> str:
        """Generate GitHub Actions CI/CD workflow."""
        has_node = any(
            t in [OutputTarget.REACT_WEB_APP, OutputTarget.REACT_NATIVE_MOBILE, OutputTarget.PWA]
            for t in config.targets
        )
        has_python = any(
            t
            in [
                OutputTarget.PYTHON_LIBRARY,
                OutputTarget.FASTAPI_BACKEND,
                OutputTarget.FLASK_BACKEND,
            ]
            for t in config.targets
        )

        steps = []
        if has_node:
            steps.append("""
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test

      - name: Build
        run: npm run build
""")

        if has_python:
            steps.append("""
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        run: pytest

      - name: Run linting
        run: ruff check .
""")

        return f"""name: CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
{''.join(steps)}
      - name: Upload coverage
        uses: codecov/codecov-action@v3
"""

    # ─────────────────────────────────────────────
    # Platform-Specific Generators
    # ─────────────────────────────────────────────

    async def _generate_python(
        self,
        description: str,
        project_name: str,
        config: ProjectOutputConfig,
    ) -> PlatformOutput:
        """Generate Python library/package."""
        output = PlatformOutput(target=OutputTarget.PYTHON_LIBRARY)

        # pyproject.toml
        output.add_file("pyproject.toml", self._generate_pyproject(project_name, config))

        # Main module
        output.add_file(
            f"{project_name}/__init__.py",
            f'"""{project_name} package."""\n\n__version__ = "0.1.0"\n',
        )
        output.add_file(
            f"{project_name}/main.py", self._generate_python_main(description, project_name)
        )

        # Tests
        if config.include_tests:
            output.add_file(
                f"tests/test_{project_name}.py", self._generate_python_tests(project_name)
            )

        # Dependencies
        output.dependencies = ["pydantic>=2.0", "httpx>=0.24"]
        output.dev_dependencies = ["pytest>=8.0", "ruff>=0.1", "mypy>=1.5"]
        output.scripts = {
            "test": "pytest",
            "lint": "ruff check .",
            "type-check": "mypy .",
        }

        return output

    def _generate_pyproject(self, project_name: str, config: ProjectOutputConfig) -> str:
        """Generate pyproject.toml."""
        return f"""[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{project_name}"
version = "0.1.0"
description = "Auto-generated Python project"
readme = "README.md"
requires-python = ">=3.10"
license = {{text = "MIT"}}

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
strict = true
"""

    def _generate_python_main(self, description: str, project_name: str) -> str:
        """Generate main Python module."""
        return f'''"""
{project_name} - {description}
"""

from pydantic import BaseModel


class Config(BaseModel):
    """Configuration model."""
    debug: bool = False
    max_items: int = 100


def main():
    """Main entry point."""
    config = Config()
    print(f"{project_name} initialized")
    return config


if __name__ == "__main__":
    main()
'''

    def _generate_python_tests(self, project_name: str) -> str:
        """Generate Python tests."""
        return f'''"""Tests for {project_name}."""

import pytest
from {project_name}.main import Config, main


def test_config_default():
    """Test default config."""
    config = Config()
    assert config.debug is False
    assert config.max_items == 100


def test_main():
    """Test main function."""
    result = main()
    assert isinstance(result, Config)
'''

    async def _generate_react(
        self,
        description: str,
        project_name: str,
        config: ProjectOutputConfig,
    ) -> PlatformOutput:
        """Generate React/Next.js web app."""
        output = PlatformOutput(target=OutputTarget.REACT_WEB_APP)

        ts = config.use_typescript
        ext = "tsx" if ts else "jsx"

        # package.json
        output.add_file("package.json", self._generate_package_json(project_name, config, web=True))

        # Next.js config
        output.add_file("next.config.js", self._generate_next_config())

        # TypeScript config
        if ts:
            output.add_file("tsconfig.json", self._generate_tsconfig())

        # Main app
        output.add_file(f"app/page.{ext}", self._generate_react_homepage(description, ts))
        output.add_file(f"app/layout.{ext}", self._generate_react_layout(project_name, ts))

        # Components
        output.add_file(f"components/Header.{ext}", self._generate_header_component(ts))

        # Styles
        if config.styling == "tailwind":
            output.add_file("tailwind.config.js", self._generate_tailwind_config())
            output.add_file("app/globals.css", self._generate_tailwind_globals())

        # Tests
        if config.include_tests:
            output.add_file(
                "__tests__/page.test.tsx" if ts else "__tests__/page.test.jsx",
                self._generate_react_tests(ts),
            )

        # Dependencies
        output.dependencies = ["next@14", "react@18", "react-dom@18"]
        if config.styling == "tailwind":
            output.dependencies.extend(["tailwindcss", "postcss", "autoprefixer"])
        output.dev_dependencies = [
            "typescript",
            "@types/react",
            "@types/node",
            "jest",
            "testing-library",
        ]
        output.scripts = {
            "dev": "next dev",
            "build": "next build",
            "start": "next start",
            "lint": "next lint",
            "test": "jest",
        }

        output.readme = self._generate_react_readme(project_name, description)

        return output

    def _generate_package_json(
        self, project_name: str, config: ProjectOutputConfig, web: bool = True
    ) -> str:
        """Generate package.json."""
        return json.dumps(
            {
                "name": project_name,
                "version": "0.1.0",
                "private": True,
                "scripts": {
                    "dev": "next dev",
                    "build": "next build",
                    "start": "next start",
                    "lint": "next lint",
                    "test": "jest",
                },
                "dependencies": {
                    "next": "14.0.0",
                    "react": "18.2.0",
                    "react-dom": "18.2.0",
                },
                "devDependencies": {
                    "@types/node": "20.0.0",
                    "@types/react": "18.2.0",
                    "typescript": "5.0.0",
                },
            },
            indent=2,
        )

    def _generate_next_config(self) -> str:
        """Generate next.config.js."""
        return """/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
}

module.exports = nextConfig
"""

    def _generate_tsconfig(self) -> str:
        """Generate tsconfig.json."""
        return json.dumps(
            {
                "compilerOptions": {
                    "target": "es5",
                    "lib": ["dom", "dom.iterable", "esnext"],
                    "allowJs": True,
                    "skipLibCheck": True,
                    "strict": True,
                    "noEmit": True,
                    "esModuleInterop": True,
                    "module": "esnext",
                    "moduleResolution": "bundler",
                    "resolveJsonModule": True,
                    "isolatedModules": True,
                    "jsx": "preserve",
                    "incremental": True,
                    "plugins": [{"name": "next"}],
                },
                "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
                "exclude": ["node_modules"],
            },
            indent=2,
        )

    def _generate_react_homepage(self, description: str, ts: bool) -> str:
        """Generate React homepage."""
        return f"""import React from 'react';

export default function Home() {{
  return (
    <main className="min-h-screen p-8">
      <h1 className="text-4xl font-bold mb-4">Welcome</h1>
      <p className="text-lg text-gray-600">{description}</p>
    </main>
  );
}}
"""

    def _generate_react_layout(self, project_name: str, ts: bool) -> str:
        """Generate React layout."""
        return f"""import React from 'react';
import Header from '../components/Header';

export const metadata = {{
  title: '{project_name}',
  description: 'Auto-generated React app',
}};

export default function RootLayout({{
  children,
}}: {{
  children: React.ReactNode;
}}) {{
  return (
    <html lang="en">
      <body>
        <Header />
        {{children}}
      </body>
    </html>
  );
}}
"""

    def _generate_header_component(self, ts: bool) -> str:
        """Generate Header component."""
        return """import React from 'react';

export default function Header() {
  return (
    <header className="border-b p-4">
      <nav className="max-w-6xl mx-auto flex justify-between">
        <span className="font-bold">App</span>
        <ul className="flex gap-4">
          <li><a href="/" className="hover:underline">Home</a></li>
          <li><a href="/about" className="hover:underline">About</a></li>
        </ul>
      </nav>
    </header>
  );
}
"""

    def _generate_tailwind_config(self) -> str:
        """Generate tailwind.config.js."""
        return """/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
"""

    def _generate_tailwind_globals(self) -> str:
        """Generate Tailwind globals.css."""
        return """@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --foreground-rgb: 0, 0, 0;
  --background-rgb: 255, 255, 255;
}

body {
  color: rgb(var(--foreground-rgb));
  background: rgb(var(--background-rgb));
}
"""

    def _generate_react_tests(self, ts: bool) -> str:
        """Generate React tests."""
        return """import { render, screen } from '@testing-library/react';
import Home from '../app/page';

describe('Home', () => {
  it('renders welcome message', () => {
    render(<Home />);
    expect(screen.getByText(/welcome/i)).toBeInTheDocument();
  });
});
"""

    def _generate_react_readme(self, project_name: str, description: str) -> str:
        """Generate React-specific README section."""
        return """
## React App

This is a Next.js 14 application with React 18.

### Development

```bash
npm run dev    # Start development server
npm run build  # Build for production
npm run start  # Start production server
```

### Project Structure

```
├── app/              # Next.js app directory
│   ├── page.tsx      # Homepage
│   └── layout.tsx    # Root layout
├── components/       # React components
├── public/           # Static assets
└── __tests__/        # Tests
```
"""

    async def _generate_react_native(
        self,
        description: str,
        project_name: str,
        config: ProjectOutputConfig,
    ) -> PlatformOutput:
        """Generate React Native mobile app."""
        output = PlatformOutput(target=OutputTarget.REACT_NATIVE_MOBILE)

        ts = config.use_typescript
        ext = "tsx" if ts else "jsx"

        # package.json
        output.add_file(
            "package.json", self._generate_package_json(project_name, config, web=False)
        )

        # App entry
        output.add_file(
            f"App.{ext}", self._generate_react_native_app(description, project_name, ts)
        )

        # TypeScript config
        if ts:
            output.add_file("tsconfig.json", self._generate_tsconfig())

        # Metro config
        output.add_file("metro.config.js", self._generate_metro_config())

        # App.json
        output.add_file("app.json", self._generate_app_json(project_name))

        # Dependencies
        output.dependencies = ["react-native@0.73", "react@18"]
        output.dev_dependencies = ["@types/react", "@types/react-native", "typescript"]
        output.scripts = {
            "android": "react-native run-android",
            "ios": "react-native run-ios",
            "start": "react-native start",
            "test": "jest",
            "lint": "eslint .",
        }

        return output

    def _generate_metro_config(self) -> str:
        """Generate metro.config.js."""
        return """const {getDefaultConfig, mergeConfig} = require('@react-native/metro-config');

const config = {};

module.exports = mergeConfig(getDefaultConfig(__dirname), config);
"""

    def _generate_app_json(self, project_name: str) -> str:
        """Generate app.json."""
        return json.dumps(
            {
                "name": project_name,
                "displayName": project_name.title(),
            },
            indent=2,
        )

    def _generate_react_native_app(self, description: str, project_name: str, ts: bool) -> str:
        """Generate React Native App.tsx."""
        return f"""import React, {{useState}} from 'react';
import {{
  SafeAreaView,
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
}} from 'react-native';

function App(): React.JSX.Element {{
  const [count, setCount] = useState(0);

  return (
    <SafeAreaView style={{styles.container}}>
      <View style={{styles.content}}>
        <Text style={{styles.title}}>{project_name.title()}</Text>
        <Text style={{styles.description}}>{description}</Text>

        <TouchableOpacity
          style={{styles.button}}
          onPress={{() => setCount(count + 1)}}>
          <Text style={{styles.buttonText}}>Count: {{count}}</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}}

const styles = StyleSheet.create({{
  container: {{
    flex: 1,
    backgroundColor: '#f5f5f5',
  }},
  content: {{
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  }},
  title: {{
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 10,
  }},
  description: {{
    fontSize: 16,
    color: '#666',
    marginBottom: 30,
  }},
  button: {{
    backgroundColor: '#007AFF',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 10,
  }},
  buttonText: {{
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  }},
}});

export default App;
"""

    async def _generate_swiftui(
        self,
        description: str,
        project_name: str,
        config: ProjectOutputConfig,
    ) -> PlatformOutput:
        """Generate SwiftUI iOS app."""
        output = PlatformOutput(target=OutputTarget.SWIFTUI_IOS)

        # Project name formatted for iOS
        ios_name = project_name.title().replace("_", "")

        # App entry
        output.add_file(f"{ios_name}App.swift", self._generate_swiftui_app(ios_name))

        # ContentView
        output.add_file(
            f"{ios_name}/ContentView.swift",
            self._generate_swiftui_contentview(description, ios_name),
        )

        # Info.plist
        output.add_file(f"{ios_name}/Info.plist", self._generate_ios_infoplist(ios_name, config))

        # Assets
        output.add_file(
            f"{ios_name}/Assets.xcassets/AccentColor.colorset/Contents.json",
            self._generate_colorset(),
        )

        # Privacy policy
        if config.include_privacy_policy:
            output.add_file(f"{ios_name}/PrivacyPolicy.md", self._generate_privacy_policy(ios_name))

        # Dependencies
        output.dependencies = []
        output.dev_dependencies = []
        output.scripts = {
            "build": f"xcodebuild -scheme {ios_name}",
            "run": f"open {ios_name}.xcodeproj",
        }

        output.readme = self._generate_swiftui_readme(ios_name, description)

        return output

    def _generate_swiftui_app(self, ios_name: str) -> str:
        """Generate SwiftUI App entry."""
        return f"""import SwiftUI

@main
struct {ios_name}App: App {{
    var body: some Scene {{
        WindowGroup {{
            ContentView()
        }}
    }}
}}
"""

    def _generate_swiftui_contentview(self, description: str, ios_name: str) -> str:
        """Generate SwiftUI ContentView."""
        return f"""import SwiftUI

struct ContentView: View {{
    @State private var count = 0

    var body: some View {{
        NavigationView {{
            VStack(spacing: 20) {{
                Text("{ios_name}")
                    .font(.largeTitle)
                    .fontWeight(.bold)

                Text("{description}")
                    .font(.body)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .padding()

                Button(action: {{
                    count += 1
                }}) {{
                    Text("Count: \\(count)")
                        .font(.title2)
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                        .padding(.horizontal, 30)
                        .padding(.vertical, 15)
                        .background(Color.blue)
                        .cornerRadius(10)
                }}

                Spacer()
            }}
            .padding()
            .navigationTitle("Home")
        }}
    }}
}}

#Preview {{
    ContentView()
}}
"""

    def _generate_ios_infoplist(self, ios_name: str, config: ProjectOutputConfig) -> str:
        """Generate Info.plist."""
        privacy_keys = ""
        if config.ios_deployment:
            privacy_keys = """
    <key>NSPrivacyPolicyURL</key>
    <string>https://example.com/privacy</string>
"""

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>$(DEVELOPMENT_LANGUAGE)</string>
    <key>CFBundleExecutable</key>
    <string>$(EXECUTABLE_NAME)</string>
    <key>CFBundleIdentifier</key>
    <string>$(PRODUCT_BUNDLE_IDENTIFIER)</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$(PRODUCT_NAME)</string>
    <key>CFBundlePackageType</key>
    <string>$(PRODUCT_BUNDLE_PACKAGE_TYPE)</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSRequiresIPhoneOS</key>
    <true/>
    <key>UIApplicationSceneManifest</key>
    <dict>
        <key>UIApplicationSupportsMultipleScenes</key>
        <false/>
    </dict>
    <key>UILaunchScreen</key>
    <dict/>
    <key>UIRequiredDeviceCapabilities</key>
    <array>
        <string>armv7</string>
    </array>
    <key>UISupportedInterfaceOrientations</key>
    <array>
        <string>UIInterfaceOrientationPortrait</string>
        <string>UIInterfaceOrientationLandscapeLeft</string>
        <string>UIInterfaceOrientationLandscapeRight</string>
    </array>
    <key>UISupportedInterfaceOrientations~ipad</key>
    <array>
        <string>UIInterfaceOrientationPortrait</string>
        <string>UIInterfaceOrientationPortraitUpsideDown</string>
        <string>UIInterfaceOrientationLandscapeLeft</string>
        <string>UIInterfaceOrientationLandscapeRight</string>
    </array>
{privacy_keys}
</dict>
</plist>
"""

    def _generate_colorset(self) -> str:
        """Generate colorset JSON."""
        return json.dumps(
            {
                "colors": [{"idiom": "universal"}],
                "info": {"author": "xcode", "version": 1},
            },
            indent=2,
        )

    def _generate_swiftui_readme(self, ios_name: str, description: str) -> str:
        """Generate SwiftUI README."""
        return f"""
## SwiftUI iOS App

Native iOS application built with SwiftUI.

### Requirements

- Xcode 15+
- iOS 17+

### Development

```bash
# Open in Xcode
open {ios_name}.xcodeproj

# Build from command line
xcodebuild -scheme {ios_name}
```

### App Store Submission

1. Update Info.plist with your bundle identifier
2. Add app icons in Assets.xcassets
3. Configure signing in Xcode
4. Archive and upload to App Store Connect
"""

    async def _generate_kotlin(
        self,
        description: str,
        project_name: str,
        config: ProjectOutputConfig,
    ) -> PlatformOutput:
        """Generate Kotlin Android app."""
        output = PlatformOutput(target=OutputTarget.KOTLIN_ANDROID)

        android_name = project_name.lower().replace("_", "")

        # build.gradle.kts (app)
        output.add_file("app/build.gradle.kts", self._generate_android_build_gradle(android_name))

        # MainActivity.kt
        output.add_file(
            f"app/src/main/java/com/example/{android_name}/MainActivity.kt",
            self._generate_main_activity(description, android_name),
        )

        # AndroidManifest.xml
        output.add_file(
            "app/src/main/AndroidManifest.xml",
            self._generate_android_manifest(android_name, config),
        )

        # Dependencies
        output.dependencies = ["com.android.application", "org.jetbrains.kotlin.android"]
        output.dev_dependencies = []
        output.scripts = {
            "build": "./gradlew assembleDebug",
            "run": "./gradlew installDebug",
            "test": "./gradlew test",
        }

        return output

    def _generate_android_build_gradle(self, android_name: str) -> str:
        """Generate build.gradle.kts."""
        return f"""plugins {{
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}}

android {{
    namespace = "com.example.{android_name}"
    compileSdk = 34

    defaultConfig {{
        applicationId = "com.example.{android_name}"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }}

    buildTypes {{
        release {{
            isMinifyEnabled = false
        }}
    }}
    compileOptions {{
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }}
    kotlinOptions {{
        jvmTarget = "1.8"
    }}
}}

dependencies {{
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:2.1.0")
    implementation("androidx.activity:activity-ktx:1.8.2")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
}}
"""

    def _generate_main_activity(self, description: str, android_name: str) -> str:
        """Generate MainActivity.kt."""
        return f"""package com.example.{android_name}

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

class MainActivity : ComponentActivity() {{
    override fun onCreate(savedInstanceState: Bundle?) {{
        super.onCreate(savedInstanceState)
        setContent {{
            MaterialTheme {{
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {{
                    MainContent()
                }}
            }}
        }}
    }}
}}

@Composable
fun MainContent() {{
    var count by remember {{ mutableStateOf(0) }}

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {{
        Text(
            text = "{android_name.title()}",
            style = MaterialTheme.typography.headlineLarge
        )

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "{description}",
            style = MaterialTheme.typography.bodyMedium
        )

        Spacer(modifier = Modifier.height(32.dp))

        Button(onClick = {{ count++ }}) {{
            Text("Count: $count")
        }}
    }}
}}
"""

    def _generate_android_manifest(self, android_name: str, config: ProjectOutputConfig) -> str:
        """Generate AndroidManifest.xml."""
        return f"""<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:roundIcon="@mipmap/ic_launcher_round"
        android:supportsRtl="true"
        android:theme="@style/Theme.{android_name.title()}">

        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>

    </application>

</manifest>
"""

    async def _generate_fastapi(
        self,
        description: str,
        project_name: str,
        config: ProjectOutputConfig,
    ) -> PlatformOutput:
        """Generate FastAPI backend."""
        output = PlatformOutput(target=OutputTarget.FASTAPI_BACKEND)

        # requirements.txt
        output.add_file("requirements.txt", self._generate_fastapi_requirements())

        # main.py
        output.add_file("main.py", self._generate_fastapi_main(description, project_name, config))

        # models.py
        output.add_file("models.py", self._generate_fastapi_models())

        # Tests
        if config.include_tests:
            output.add_file("tests/test_main.py", self._generate_fastapi_tests())

        # Dependencies
        output.dependencies = ["fastapi>=0.109", "uvicorn>=0.27", "pydantic>=2.0"]
        if config.include_database:
            output.dependencies.extend(["sqlalchemy>=2.0", "aiosqlite>=0.19"])
        if config.include_auth:
            output.dependencies.extend(["python-jose[cryptography]", "passlib[bcrypt]"])
        output.dev_dependencies = ["pytest>=8.0", "httpx>=0.24", "pytest-asyncio>=0.23"]
        output.scripts = {
            "dev": "uvicorn main:app --reload",
            "prod": "uvicorn main:app --host 0.0.0.0 --port 8000",
            "test": "pytest",
        }

        output.readme = self._generate_fastapi_readme(project_name, description)

        return output

    def _generate_fastapi_requirements(self) -> str:
        """Generate requirements.txt."""
        return """fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
"""

    def _generate_fastapi_main(
        self, description: str, project_name: str, config: ProjectOutputConfig
    ) -> str:
        """Generate FastAPI main.py."""
        auth_import = ""
        auth_router = ""
        if config.include_auth:
            auth_import = (
                "from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm"
            )
            auth_router = """

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # TODO: Implement actual authentication
    return {"access_token": "fake_token", "token_type": "bearer"}
"""

        db_import = ""
        db_setup = ""
        if config.include_database:
            db_import = """
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
"""
            db_setup = """
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
"""

        return f'''"""
{project_name} - {description}
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
{auth_import}
{db_import}

app = FastAPI(
    title="{project_name.title()}",
    description="{description}",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

{db_setup}

class Item(BaseModel):
    name: str
    description: str | None = None

@app.get("/")
async def root():
    return {{"message": f"Welcome to {project_name.title()}"}}

@app.get("/health")
async def health_check():
    return {{"status": "healthy"}}

@app.get("/items/{{item_id}}")
async def read_item(item_id: int):
    return {{"item_id": item_id, "name": f"Item {{item_id}}"}}

@app.post("/items")
async def create_item(item: Item):
    return item
{auth_router}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

    def _generate_fastapi_models(self) -> str:
        """Generate models.py."""
        return """from pydantic import BaseModel
from typing import Optional

class User(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool = True

class Item(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    price: float
"""

    def _generate_fastapi_tests(self) -> str:
        """Generate FastAPI tests."""
        return """import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_read_item():
    response = client.get("/items/1")
    assert response.status_code == 200
    assert "item_id" in response.json()

def test_create_item():
    response = client.post(
        "/items",
        json={"name": "Test Item", "description": "A test item"}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Test Item"
"""

    def _generate_fastapi_readme(self, project_name: str, description: str) -> str:
        """Generate FastAPI README."""
        return f"""
## FastAPI Backend

RESTful API built with FastAPI.

### Features

- Automatic OpenAPI documentation
- Async support
- Pydantic validation
- {"✅ Authentication" if "auth" else "❌ No auth"}
- {"✅ Database" if "db" else "❌ No database"}

### Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload

# Access API docs
# http://localhost:8000/docs
# http://localhost:8000/redoc
```

### API Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check
- `GET /items/{{item_id}}` - Get item
- `POST /items` - Create item
"""

    async def _generate_flask(
        self,
        description: str,
        project_name: str,
        config: ProjectOutputConfig,
    ) -> PlatformOutput:
        """Generate Flask backend."""
        output = PlatformOutput(target=OutputTarget.FLASK_BACKEND)

        # requirements.txt
        output.add_file("requirements.txt", "flask>=3.0\npython-dotenv>=1.0\n")

        # app.py
        output.add_file(
            "app.py",
            f'''"""
{project_name} - {description}
"""

from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/")
def hello():
    return jsonify({{"message": "Hello from {project_name}!"}})

@app.route("/health")
def health():
    return jsonify({{"status": "healthy"}})

if __name__ == "__main__":
    app.run(debug=True)
''',
        )

        output.dependencies = ["flask>=3.0", "python-dotenv>=1.0"]
        output.dev_dependencies = ["pytest>=8.0"]
        output.scripts = {
            "dev": "python app.py",
            "test": "pytest",
        }

        return output

    async def _generate_full_stack(
        self,
        description: str,
        project_name: str,
        config: ProjectOutputConfig,
    ) -> PlatformOutput:
        """Generate full-stack application."""
        output = PlatformOutput(target=OutputTarget.FULL_STACK)

        # Generate backend
        backend = await self._generate_fastapi(description, f"{project_name}_backend", config)
        for f in backend.files:
            output.add_file(f"backend/{f.path}", f.content, f.description)
        output.dependencies.extend([f"backend: {d}" for d in backend.dependencies])

        # Generate frontend
        frontend = await self._generate_react(description, f"{project_name}_frontend", config)
        for f in frontend.files:
            output.add_file(f"frontend/{f.path}", f.content, f.description)
        output.dependencies.extend([f"frontend: {d}" for d in frontend.dependencies])

        # Docker Compose
        output.add_file("docker-compose.yml", self._generate_docker_compose(project_name))

        output.scripts = {
            "dev": "docker-compose up",
            "build": "docker-compose build",
        }

        output.readme = self._generate_fullstack_readme(project_name, description)

        return output

    def _generate_docker_compose(self, project_name: str) -> str:
        """Generate docker-compose.yml."""
        return """version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:///./app.db
    volumes:
      - ./backend:/app

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
"""

    def _generate_fullstack_readme(self, project_name: str, description: str) -> str:
        """Generate full-stack README."""
        return """
## Full-Stack Application

Complete application with React frontend and FastAPI backend.

### Architecture

```
├── frontend/     # Next.js React app
├── backend/      # FastAPI REST API
└── docker-compose.yml
```

### Development

```bash
# Start all services
docker-compose up

# Frontend: http://localhost:3000
# Backend: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Production

```bash
docker-compose -f docker-compose.prod.yml up
```
"""

    async def _generate_pwa(
        self,
        description: str,
        project_name: str,
        config: ProjectOutputConfig,
    ) -> PlatformOutput:
        """Generate Progressive Web App."""
        output = PlatformOutput(target=OutputTarget.PWA)

        # Similar to React but with PWA additions
        react_output = await self._generate_react(description, project_name, config)
        output.files = react_output.files
        output.dependencies = react_output.dependencies

        # Add manifest.json
        output.add_file("public/manifest.json", self._generate_pwa_manifest(project_name))

        # Add service worker
        output.add_file("public/sw.js", self._generate_service_worker())

        # Add PWA configuration
        output.add_file("next.config.js", self._generate_pwa_next_config())

        output.readme = self._generate_pwa_readme(project_name, description)

        return output

    def _generate_pwa_manifest(self, project_name: str) -> str:
        """Generate manifest.json for PWA."""
        return json.dumps(
            {
                "name": project_name.title(),
                "short_name": project_name[:15],
                "description": "Auto-generated PWA",
                "start_url": "/",
                "display": "standalone",
                "background_color": "#ffffff",
                "theme_color": "#000000",
                "icons": [
                    {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
                    {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"},
                ],
            },
            indent=2,
        )

    def _generate_service_worker(self) -> str:
        """Generate service worker."""
        return """const CACHE_NAME = 'app-cache-v1';
const urlsToCache = ['/', '/offline'];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
  );
});
"""

    def _generate_pwa_next_config(self) -> str:
        """Generate Next.js config with PWA support."""
        return """/** @type {import('next').NextConfig} */
const withPWA = require('next-pwa')({
  dest: 'public',
  disable: process.env.NODE_ENV === 'development',
});

module.exports = withPWA({
  reactStrictMode: true,
});
"""

    def _generate_pwa_readme(self, project_name: str, description: str) -> str:
        """Generate PWA README."""
        return """
## Progressive Web App

Installable web application with offline support.

### Features

- ✅ Works offline
- ✅ Installable on home screen
- ✅ Push notifications ready
- ✅ Fast loading with caching

### Testing PWA

1. Build: `npm run build`
2. Start: `npm run start`
3. Open Chrome DevTools → Application → Service Workers
4. Test offline mode in Network tab
"""


async def generate_multi_platform(
    project_description: str,
    config: ProjectOutputConfig | None = None,
    project_name: str | None = None,
) -> MultiPlatformResult:
    """
    Convenience function to generate multi-platform output.

    Usage:
        result = await generate_multi_platform(
            "Build a todo app",
            config=ProjectOutputConfig(
                targets=[OutputTarget.REACT_WEB_APP, OutputTarget.FASTAPI_BACKEND],
            ),
        )
    """
    generator = MultiPlatformGenerator()
    return await generator.generate(project_description, config, project_name)

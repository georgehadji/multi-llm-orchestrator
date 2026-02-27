"""Create configuration files."""
from pathlib import Path

# .editorconfig
editorconfig_content = '''# EditorConfig
# https://editorconfig.org

root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space
indent_size = 4
max_line_length = 100

[*.{yml,yaml}]
indent_size = 2

[*.{json,js,ts}]
indent_size = 2

[*.md]
trim_trailing_whitespace = false
max_line_length = 80

[Makefile]
indent_style = tab
'''

# LICENSE (MIT)
license_content = '''MIT License

Copyright (c) 2024 Georgios-Chrysovalantis Chatzivantsidis

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
'''

# CHANGELOG.md
changelog_content = '''# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Production-ready project structure
- Exception hierarchy for proper error handling
- Structured logging with correlation IDs
- Docker multi-stage build
- GitHub Actions CI/CD pipeline
- Pre-commit hooks
- Development scripts (setup.sh, lint.sh, test.sh, release.sh)
- Comprehensive Makefile with all development tasks

### Changed
- Enhanced pyproject.toml with full tool configurations
- Updated README with development instructions
- Added CONTRIBUTING.md guidelines

## [1.1.0] - 2024-01-15

### Added
- Multi-provider LLM support (DeepSeek, OpenAI, Google, Kimi, Minimax, Zhipu)
- Cost-optimized routing with budget hierarchy
- Policy enforcement system
- Resume capability for interrupted projects
- Deterministic validation
- Real-time telemetry and OpenTracing support

## [1.0.0] - 2024-01-01

### Added
- Initial release
- Basic orchestration engine
- Support for OpenAI and DeepSeek providers
'''

# docker-compose.yml
docker_compose_content = '''version: "3.8"

services:
  # ═══════════════════════════════════════════════════════════════════════════════
  # Main Application
  # ═══════════════════════════════════════════════════════════════════════════════
  orchestrator:
    build:
      context: .
      target: development
    volumes:
      - ./outputs:/app/outputs
      - ./logs:/app/logs
      - ./data:/app/data
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=DEBUG
      - LOG_FORMAT=text
    env_file:
      - .env
    command: python main.py --help

  # ═══════════════════════════════════════════════════════════════════════════════
  # Cache/State Storage (Redis)
  # ═══════════════════════════════════════════════════════════════════════════════
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  # ═══════════════════════════════════════════════════════════════════════════════
  # OpenTelemetry Collector (for tracing)
  # ═══════════════════════════════════════════════════════════════════════════════
  otel-collector:
    image: otel/opentelemetry-collector:latest
    ports:
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP
      - "55679:55679" # zpages
    volumes:
      - ./docker/otel-collector-config.yaml:/etc/otel-collector-config.yaml
    command: ["--config=/etc/otel-collector-config.yaml"]
    profiles:
      - tracing

  # ═══════════════════════════════════════════════════════════════════════════════
  # Prometheus (for metrics)
  # ═══════════════════════════════════════════════════════════════════════════════
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./docker/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    profiles:
      - monitoring

  # ═══════════════════════════════════════════════════════════════════════════════
  # Grafana (for visualization)
  # ═══════════════════════════════════════════════════════════════════════════════
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./docker/grafana/dashboards:/etc/grafana/provisioning/dashboards
    profiles:
      - monitoring

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
'''

# Write files
files_to_create = [
    (".editorconfig", editorconfig_content),
    ("LICENSE", license_content),
    ("CHANGELOG.md", changelog_content),
    ("docker-compose.yml", docker_compose_content),
]

for filename, content in files_to_create:
    filepath = Path(filename)
    filepath.write_text(content, encoding='utf-8')
    print(f"✓ Created: {filepath}")

print("\n✓ All configuration files created!")

# Self cleanup
Path(__file__).unlink()

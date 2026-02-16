"""
MetricsExporter — pluggable metrics export for live orchestrator telemetry.
===========================================================================
Follows the same ABC + concrete-class pattern as OptimizationBackend.

Three built-in exporters:
  ConsoleExporter    — ASCII table to stdout (human-readable)
  JSONExporter       — JSON file dump (for dashboards)
  PrometheusExporter — Prometheus text format (for monitoring stacks)

Usage:
    orch.set_metrics_exporter(ConsoleExporter())
    orch.export_metrics()        # pull stats from live profiles and export

The metrics dict shape (produced by Orchestrator._build_metrics_dict()):
{
    "kimi-k2-5": {
        "call_count": 12,
        "failure_count": 0,
        "success_rate": 1.0,
        "avg_latency_ms": 1830.4,
        "latency_p95_ms": 2900.0,
        "quality_score": 0.87,
        "trust_factor": 1.0,
        "avg_cost_usd": 0.000042,
        "validator_fail_count": 0,
        "error_rate": 0.0,
    },
    ...
}
"""
from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# MetricsExporter ABC
# ─────────────────────────────────────────────────────────────────────────────

class MetricsExporter(ABC):
    """
    Abstract base for all metrics export targets.

    Implementations receive a metrics dict keyed by model name and are
    responsible for formatting and delivering the data to their target.
    """

    @abstractmethod
    def export(self, metrics: dict) -> None:
        """
        Export the metrics dict.

        Parameters
        ----------
        metrics : dict
            Per-model stats dict as produced by Orchestrator._build_metrics_dict().
        """


# ─────────────────────────────────────────────────────────────────────────────
# ConsoleExporter
# ─────────────────────────────────────────────────────────────────────────────

_CONSOLE_COLS = [
    ("model",                20),
    ("calls",                 6),
    ("success%",              9),
    ("avg_lat_ms",           11),
    ("p95_lat_ms",           11),
    ("quality",               8),
    ("trust",                 7),
    ("avg_cost_usd",         13),
    ("val_fails",            10),
    ("err_rate",              9),
]


class ConsoleExporter(MetricsExporter):
    """
    Prints a fixed-width ASCII table to stdout.

    Example output:
        model                calls  success%  avg_lat_ms  p95_lat_ms  quality  trust  avg_cost_usd  val_fails  err_rate
        ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
        kimi-k2-5               12   100.00%      1830.4      2900.0    0.870  1.000     0.0000420          0      0.0%
    """

    def export(self, metrics: dict) -> None:
        header = "  ".join(name.ljust(width) for name, width in _CONSOLE_COLS)
        sep = "─" * len(header)
        print(header)
        print(sep)
        for model_name, stats in sorted(metrics.items()):
            row_parts = [
                model_name[:20].ljust(20),
                str(stats.get("call_count", 0)).rjust(6),
                f"{stats.get('success_rate', 0) * 100:.2f}%".rjust(9),
                f"{stats.get('avg_latency_ms', 0):.1f}".rjust(11),
                f"{stats.get('latency_p95_ms', 0):.1f}".rjust(11),
                f"{stats.get('quality_score', 0):.3f}".rjust(8),
                f"{stats.get('trust_factor', 0):.3f}".rjust(7),
                f"{stats.get('avg_cost_usd', 0):.7f}".rjust(13),
                str(stats.get("validator_fail_count", 0)).rjust(10),
                f"{stats.get('error_rate', 0) * 100:.1f}%".rjust(9),
            ]
            print("  ".join(row_parts))


# ─────────────────────────────────────────────────────────────────────────────
# JSONExporter
# ─────────────────────────────────────────────────────────────────────────────

class JSONExporter(MetricsExporter):
    """
    Writes the metrics dict to a JSON file with 2-space indentation.

    The file is overwritten on each call (not appended).

    Usage:
        orch.set_metrics_exporter(JSONExporter("/tmp/orchestrator_metrics.json"))
        orch.export_metrics()
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def export(self, metrics: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# PrometheusExporter
# ─────────────────────────────────────────────────────────────────────────────

_PROMETHEUS_METRICS: list[tuple[str, str, str]] = [
    # (key_in_stats,            prometheus_metric_name,                  help_text)
    ("call_count",              "orchestrator_model_calls_total",         "Total API calls made to this model"),
    ("success_rate",            "orchestrator_success_rate",              "Rolling success rate (0-1)"),
    ("avg_latency_ms",          "orchestrator_latency_avg_ms",            "Exponential moving average latency in ms"),
    ("latency_p95_ms",          "orchestrator_latency_p95_ms",            "p95 latency in ms (last 50 samples)"),
    ("quality_score",           "orchestrator_quality_score",             "EMA of LLM evaluator quality scores"),
    ("trust_factor",            "orchestrator_trust_factor",              "Trust factor (degrades on failures)"),
    ("avg_cost_usd",            "orchestrator_cost_avg_usd",              "EMA of per-call USD cost"),
    ("validator_fail_count",    "orchestrator_validator_failures_total",  "Cumulative deterministic validator failures"),
    ("error_rate",              "orchestrator_error_rate",                "error_rate = failure_count / call_count"),
]


def _sanitize_label(value: str) -> str:
    """Sanitize a model name for use as a Prometheus label value.

    Prometheus label values must not contain certain chars.
    We replace hyphens with underscores for compatibility.
    """
    return value.replace("-", "_").replace(".", "_")


class PrometheusExporter(MetricsExporter):
    """
    Formats metrics in Prometheus text exposition format.

    Pure stdlib — no prometheus_client dependency required.
    Each metric family is emitted as a gauge with a ``model`` label.

    Usage:
        # Write to stdout:
        orch.set_metrics_exporter(PrometheusExporter())

        # Write to a file (for node_exporter textfile collector):
        orch.set_metrics_exporter(PrometheusExporter(output_file="/var/lib/node_exporter/orchestrator.prom"))
    """

    def __init__(self, output_file: Optional[str | Path] = None) -> None:
        self._output_file: Optional[Path] = Path(output_file) if output_file else None

    def export(self, metrics: dict) -> None:
        lines: list[str] = []
        for stat_key, metric_name, help_text in _PROMETHEUS_METRICS:
            lines.append(f"# HELP {metric_name} {help_text}")
            lines.append(f"# TYPE {metric_name} gauge")
            for model_name, stats in sorted(metrics.items()):
                label_val = _sanitize_label(model_name)
                value = stats.get(stat_key, 0)
                lines.append(f'{metric_name}{{model="{label_val}"}} {value}')
        content = "\n".join(lines) + "\n"

        if self._output_file is not None:
            self._output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._output_file, "w", encoding="utf-8") as fh:
                fh.write(content)
        else:
            sys.stdout.write(content)

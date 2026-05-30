"""
Drift — Drift detection
======================
Module for detecting and managing concept/model drift in the orchestrator.

Pattern: Observer
Async: Yes — for I/O-bound operations
Layer: L6 Observability

Usage:
    from orchestrator.drift import DriftDetector
    detector = DriftDetector(window_size=100, threshold=0.05)
    is_drifting = await detector.check_drift(new_sample=np.array([...]))
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger("orchestrator.drift")


class DriftDetectionResult:
    """Result of a drift detection operation."""

    def __init__(
        self, is_drifting: bool, confidence: float, drift_score: float, details: dict[str, Any]
    ):
        self.is_drifting = is_drifting
        self.confidence = confidence
        self.drift_score = drift_score
        self.details = details
        self.timestamp = datetime.now()


class DriftDetector:
    """Detects and manages concept/model drift in the orchestrator."""

    def __init__(
        self,
        window_size: int = 100,
        threshold: float = 0.05,
        metric: str = "ks_test",
        warmup_samples: int = 50,
    ):
        """
        Initialize the drift detector.

        Args:
            window_size: Size of the sliding window for comparison
            threshold: Threshold for drift detection
            metric: Metric to use for drift detection ('ks_test', 'hellinger', 'bhattacharyya')
            warmup_samples: Number of samples needed before detection begins
        """
        self.window_size = window_size
        self.threshold = threshold
        self.metric = metric
        self.warmup_samples = warmup_samples

        # Sliding windows for reference and current data
        self.reference_window = deque(maxlen=window_size)
        self.current_window = deque(maxlen=window_size)

        self.samples_seen = 0
        self.drift_history: list[DriftDetectionResult] = []
        self.last_drift_timestamp = None
        self.drift_count = 0

        # Statistics for monitoring
        self.metrics_history = deque(maxlen=1000)

    def add_sample(self, sample: np.ndarray | list[float] | float) -> DriftDetectionResult:
        """
        Add a new sample and check for drift.

        Args:
            sample: New sample to add to the current window

        Returns:
            DriftDetectionResult indicating if drift was detected
        """
        # Convert sample to numpy array if needed
        if not isinstance(sample, np.ndarray):
            sample = np.array(sample)

        # Add to current window
        self.current_window.append(sample)
        self.samples_seen += 1

        # If we haven't seen enough samples for warmup, return negative result
        if self.samples_seen < self.warmup_samples:
            return DriftDetectionResult(
                is_drifting=False,
                confidence=0.0,
                drift_score=0.0,
                details={"status": "warmup", "samples_seen": self.samples_seen},
            )

        # If reference window is not filled, fill it with initial samples
        if (
            len(self.reference_window) < self.window_size
            and self.samples_seen <= self.warmup_samples + self.window_size
        ):
            self.reference_window.append(sample)
            return DriftDetectionResult(
                is_drifting=False,
                confidence=0.0,
                drift_score=0.0,
                details={
                    "status": "building_reference",
                    "reference_size": len(self.reference_window),
                },
            )

        # Calculate drift score using the selected metric
        if self.metric == "ks_test":
            drift_score = self._ks_test_drift()
        elif self.metric == "hellinger":
            drift_score = self._hellinger_distance()
        elif self.metric == "bhattacharyya":
            drift_score = self._bhattacharyya_distance()
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # Determine if drift is occurring
        is_drifting = drift_score > self.threshold
        confidence = min(drift_score / self.threshold, 1.0)  # Normalize confidence

        # Create result
        result = DriftDetectionResult(
            is_drifting=is_drifting,
            confidence=confidence,
            drift_score=drift_score,
            details={
                "metric": self.metric,
                "threshold": self.threshold,
                "sample_shape": sample.shape if hasattr(sample, "shape") else len(sample),
            },
        )

        # Record in history
        self.drift_history.append(result)
        self.metrics_history.append(
            {"timestamp": result.timestamp, "drift_score": drift_score, "is_drifting": is_drifting}
        )

        # Update counters if drift detected
        if is_drifting:
            self.drift_count += 1
            self.last_drift_timestamp = result.timestamp

            # Log drift detection
            logger.warning(f"Drift detected! Score: {drift_score:.4f}, Threshold: {self.threshold}")

        return result

    def _ks_test_drift(self) -> float:
        """Calculate drift using Kolmogorov-Smirnov test."""
        if len(self.reference_window) < 2 or len(self.current_window) < 2:
            return 0.0

        # Flatten arrays if they're multidimensional
        ref_flat = np.concatenate([np.atleast_1d(arr.flatten()) for arr in self.reference_window])
        curr_flat = np.concatenate([np.atleast_1d(arr.flatten()) for arr in self.current_window])

        # Perform KS test
        from scipy import stats

        ks_statistic, p_value = stats.ks_2samp(ref_flat, curr_flat)

        # Return the KS statistic as the drift score
        return float(ks_statistic)

    def _hellinger_distance(self) -> float:
        """Calculate drift using Hellinger distance."""
        if len(self.reference_window) < 2 or len(self.current_window) < 2:
            return 0.0

        # Convert windows to probability distributions
        ref_concat = np.concatenate([np.atleast_1d(arr.flatten()) for arr in self.reference_window])
        curr_concat = np.concatenate([np.atleast_1d(arr.flatten()) for arr in self.current_window])

        # Create histograms
        bins = min(50, len(np.unique(np.concatenate([ref_concat, curr_concat]))))
        ref_hist, _ = np.histogram(ref_concat, bins=bins, density=True)
        curr_hist, _ = np.histogram(curr_concat, bins=bins, density=True)

        # Normalize histograms
        ref_hist = ref_hist / np.sum(ref_hist)
        curr_hist = curr_hist / np.sum(curr_hist)

        # Calculate Hellinger distance
        hellinger_dist = np.sqrt(np.sum((np.sqrt(ref_hist) - np.sqrt(curr_hist)) ** 2)) / np.sqrt(2)

        return float(hellinger_dist)

    def _bhattacharyya_distance(self) -> float:
        """Calculate drift using Bhattacharyya distance."""
        if len(self.reference_window) < 2 or len(self.current_window) < 2:
            return 0.0

        # Convert windows to probability distributions
        ref_concat = np.concatenate([np.atleast_1d(arr.flatten()) for arr in self.reference_window])
        curr_concat = np.concatenate([np.atleast_1d(arr.flatten()) for arr in self.current_window])

        # Create histograms
        bins = min(50, len(np.unique(np.concatenate([ref_concat, curr_concat]))))
        ref_hist, _ = np.histogram(ref_concat, bins=bins, density=True)
        curr_hist, _ = np.histogram(curr_concat, bins=bins, density=True)

        # Normalize histograms
        ref_hist = ref_hist / np.sum(ref_hist)
        curr_hist = curr_hist / np.sum(curr_hist)

        # Calculate Bhattacharyya coefficient
        bc = np.sum(np.sqrt(ref_hist * curr_hist))

        # Calculate Bhattacharyya distance
        bhattacharyya_dist = -np.log(bc) if bc > 0 else float("inf")

        return float(bhattacharyya_dist)

    def reset_reference(self):
        """Reset the reference window with current data."""
        self.reference_window.clear()
        for item in self.current_window:
            self.reference_window.append(item)

        logger.info("Reference window reset with current data")

    def get_drift_history(self, limit: int = 100) -> list[DriftDetectionResult]:
        """Get recent drift detection history."""
        return self.drift_history[-limit:]

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about drift detection."""
        if not self.metrics_history:
            return {
                "samples_seen": self.samples_seen,
                "drift_count": self.drift_count,
                "drift_rate": 0.0,
                "last_drift": None,
                "avg_drift_score": 0.0,
            }

        drift_rates = [m["drift_score"] for m in self.metrics_history]

        return {
            "samples_seen": self.samples_seen,
            "drift_count": self.drift_count,
            "drift_rate": self.drift_count / max(self.samples_seen - self.warmup_samples, 1),
            "last_drift": (
                self.last_drift_timestamp.isoformat() if self.last_drift_timestamp else None
            ),
            "avg_drift_score": float(np.mean(drift_rates)),
            "std_drift_score": float(np.std(drift_rates)),
            "min_drift_score": float(np.min(drift_rates)),
            "max_drift_score": float(np.max(drift_rates)),
        }

    def update_threshold(self, new_threshold: float):
        """Update the drift detection threshold."""
        old_threshold = self.threshold
        self.threshold = new_threshold
        logger.info(f"Drift detection threshold updated from {old_threshold} to {new_threshold}")

    def calibrate(self, calibration_data: list[np.ndarray], percentile: float = 95.0) -> float:
        """
        Calibrate the threshold based on calibration data.

        Args:
            calibration_data: List of samples from the expected distribution
            percentile: Percentile to use for threshold setting

        Returns:
            New threshold value
        """
        # Temporarily store current state
        current_ref = list(self.reference_window)
        current_curr = list(self.current_window)

        # Clear windows
        self.reference_window.clear()
        self.current_window.clear()

        # Add calibration data to reference window
        for sample in calibration_data:
            self.reference_window.append(sample)

        # Compute distances for calibration data against itself
        distances = []
        for sample in calibration_data:
            # Add sample to current window
            self.current_window.append(sample)

            # Calculate drift score
            if self.metric == "ks_test":
                dist = self._ks_test_drift()
            elif self.metric == "hellinger":
                dist = self._hellinger_distance()
            elif self.metric == "bhattacharyya":
                dist = self._bhattacharyya_distance()

            distances.append(dist)

            # Remove sample from current window
            if self.current_window:
                self.current_window.popleft()

        # Calculate threshold as the specified percentile of distances
        threshold = float(np.percentile(distances, percentile))

        # Restore original state
        self.reference_window.clear()
        for item in current_ref:
            self.reference_window.append(item)

        self.current_window.clear()
        for item in current_curr:
            self.current_window.append(item)

        # Update threshold
        self.update_threshold(threshold)

        logger.info(f"Threshold calibrated to {threshold:.4f} using {percentile}th percentile")
        return threshold


class ModelDriftMonitor:
    """Monitors drift in ML models used by the orchestrator."""

    def __init__(self, detector: DriftDetector, model_name: str = "unknown"):
        """
        Initialize the model drift monitor.

        Args:
            detector: Drift detector instance
            model_name: Name of the model being monitored
        """
        self.detector = detector
        self.model_name = model_name
        self.performance_history = deque(maxlen=1000)
        self.drift_alerts = []

    async def monitor_input_drift(self, input_sample: np.ndarray) -> DriftDetectionResult:
        """
        Monitor for drift in input data.

        Args:
            input_sample: Input sample to the model

        Returns:
            DriftDetectionResult
        """
        return self.detector.add_sample(input_sample)

    async def monitor_performance_drift(
        self, input_sample: np.ndarray, expected_output: np.ndarray, actual_output: np.ndarray
    ) -> dict[str, Any]:
        """
        Monitor for drift based on model performance.

        Args:
            input_sample: Input to the model
            expected_output: Expected output (if available)
            actual_output: Actual output from the model

        Returns:
            Dict with performance metrics
        """
        # Calculate performance metric (e.g., accuracy, loss)
        if expected_output is not None:
            # Calculate error or accuracy
            error = np.mean((expected_output - actual_output) ** 2)  # MSE
            accuracy = 1 / (1 + error)  # Simple transformation
        else:
            # If no expected output, we can't calculate performance
            error = 0.0
            accuracy = 1.0

        # Add to performance history
        perf_record = {
            "timestamp": datetime.now(),
            "error": float(error),
            "accuracy": float(accuracy),
            "input_sample": (
                input_sample.tolist() if isinstance(input_sample, np.ndarray) else input_sample
            ),
        }
        self.performance_history.append(perf_record)

        # Check if performance degradation indicates drift
        is_degrading = self._is_performance_degrading()

        result = {
            "performance_drift": is_degrading,
            "current_error": float(error),
            "current_accuracy": float(accuracy),
            "avg_error": float(np.mean([p["error"] for p in self.performance_history])),
            "avg_accuracy": float(np.mean([p["accuracy"] for p in self.performance_history])),
        }

        if is_degrading:
            alert = {"timestamp": datetime.now(), "type": "performance_drift", "details": result}
            self.drift_alerts.append(alert)
            logger.warning(f"Performance drift detected for model {self.model_name}: {result}")

        return result

    def _is_performance_degrading(self) -> bool:
        """Check if model performance is degrading."""
        if len(self.performance_history) < 10:  # Need minimum samples
            return False

        # Get recent performance metrics
        recent_metrics = list(self.performance_history)[-10:]
        older_metrics = (
            list(self.performance_history)[-20:-10]
            if len(self.performance_history) >= 20
            else list(self.performance_history)[:10]
        )

        recent_avg_error = np.mean([m["error"] for m in recent_metrics])
        older_avg_error = np.mean([m["error"] for m in older_metrics])

        # If recent error is significantly higher than older error, flag as degrading
        threshold = 0.1  # 10% increase in error
        if older_avg_error > 0:
            error_increase = (recent_avg_error - older_avg_error) / older_avg_error
            return error_increase > threshold

        return False

    def get_monitoring_stats(self) -> dict[str, Any]:
        """Get statistics about model monitoring."""
        drift_stats = self.detector.get_statistics()

        if self.performance_history:
            perf_errors = [p["error"] for p in self.performance_history]
            perf_accuracies = [p["accuracy"] for p in self.performance_history]

            performance_stats = {
                "avg_error": float(np.mean(perf_errors)),
                "std_error": float(np.std(perf_errors)),
                "min_error": float(np.min(perf_errors)),
                "max_error": float(np.max(perf_errors)),
                "avg_accuracy": float(np.mean(perf_accuracies)),
                "std_accuracy": float(np.std(perf_accuracies)),
                "total_performance_records": len(self.performance_history),
            }
        else:
            performance_stats = {
                "avg_error": 0.0,
                "std_error": 0.0,
                "min_error": 0.0,
                "max_error": 0.0,
                "avg_accuracy": 1.0,
                "std_accuracy": 0.0,
                "total_performance_records": 0,
            }

        return {
            "model_name": self.model_name,
            "drift_stats": drift_stats,
            "performance_stats": performance_stats,
            "total_drift_alerts": len(self.drift_alerts),
        }


# Global drift detector for the orchestrator
_global_drift_detector: DriftDetector | None = None


def get_global_drift_detector(
    window_size: int = 100,
    threshold: float = 0.05,
    metric: str = "ks_test",
    warmup_samples: int = 50,
) -> DriftDetector:
    """
    Get the global drift detector instance, creating it if it doesn't exist.

    Args:
        window_size: Size of the sliding window for comparison
        threshold: Threshold for drift detection
        metric: Metric to use for drift detection
        warmup_samples: Number of samples needed before detection begins

    Returns:
        DriftDetector instance
    """
    global _global_drift_detector
    if _global_drift_detector is None:
        _global_drift_detector = DriftDetector(
            window_size=window_size,
            threshold=threshold,
            metric=metric,
            warmup_samples=warmup_samples,
        )
    return _global_drift_detector

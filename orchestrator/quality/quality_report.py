"""QualityReport — Quality assessment with score."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class QualityReport:
    overall_score: float = 0.0
    lint_errors: int = 0
    type_errors: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    coverage_pct: float = 0.0
    security_issues: int = 0
    recommendation: str = "PASS"

    def compute_score(self):
        score = 10.0
        score -= self.lint_errors * 0.5
        score -= self.type_errors * 0.5
        score -= self.tests_failed * 1.0
        if self.coverage_pct < 80:
            score -= (80 - self.coverage_pct) * 0.05
        score -= self.security_issues * 2.0
        return max(0.0, round(score, 1))

    def auto_recommend(self):
        s = self.compute_score()
        if s >= 7.0:
            return "PASS"
        elif s >= 4.0:
            return "REVISE"
        return "BLOCK"

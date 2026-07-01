"""ProgressReporter — Formatted progress tables."""


class ProgressReporter:
    def report(self, sprint):
        lines = [f"Sprint: {sprint.goal}", ""]
        for ms in sprint.milestones:
            tasks = {tid: sprint.tasks[tid] for tid in ms.task_ids if tid in sprint.tasks}
            done = sum(1 for t in tasks.values() if t.status == "done")
            total = len(tasks)
            pct = (done / total * 100) if total else 0
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "=" * filled + " " * (bar_len - filled)
            lines.append(f"  {ms.title}: {done}/{total} tasks  [{bar}] {pct:.0f}%")
            for t in tasks.values():
                icon = {"done": "OK", "failed": "XX", "running": ".."}.get(t.status, "  ")
                lines.append(f"    [{icon}] {t.description}")
        lines.append("")
        return "\n".join(lines)

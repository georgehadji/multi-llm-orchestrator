"""
AutomationScheduler - Cron, schedules, entity events, webhooks.
================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 5, Phase B3 (Base44-inspired).
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass
from enum import Enum
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class ScheduleType(str, Enum):
    CRON = "cron"
    INTERVAL = "interval"
    EVENT = "event"
    WEBHOOK = "webhook"
    ONCE = "once"


@dataclass
class ScheduledTask:
    name: str
    schedule_type: ScheduleType
    cron_expr: str = ""
    interval_seconds: int = 0
    event_entity: str = ""
    webhook_path: str = ""
    last_run: float = 0.0
    run_count: int = 0
    enabled: bool = True

    def to_dict(self):
        return {
            "name": self.name,
            "type": self.schedule_type.value,
            "cron": self.cron_expr,
            "interval": self.interval_seconds,
            "entity": self.event_entity,
            "webhook": self.webhook_path,
            "last_run": self.last_run,
            "run_count": self.run_count,
            "enabled": self.enabled,
        }


class CronParser:
    @staticmethod
    def matches(cron, timestamp=None):
        if not cron:
            return False
        t = time.localtime(timestamp or time.time())
        # Python tm_wday: Mon=0..Sun=6.  Cron weekday: Sun=0..Sat=6.
        fields = [t.tm_min, t.tm_hour, t.tm_mday, t.tm_mon, (t.tm_wday + 1) % 7]
        cron_fields = cron.split()
        if len(cron_fields) != 5:
            return False
        for cf, val in zip(cron_fields, fields):
            if cf == "*":
                continue
            if cf.startswith("*/"):
                step = int(cf[2:])
                if step == 0:
                    return False
                if val % step != 0:
                    return False
            else:
                parts = [int(p) for p in cf.split(",")]
                if val not in parts:
                    return False
        return True


class AutomationScheduler:
    """Manages scheduled tasks with cron, interval, event, and webhook triggers."""

    def __init__(self, storage_dir=None):
        self._dir = Path(storage_dir or Path.home() / ".orchestrator_cache" / "automations")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._tasks = {}
        self._handlers = {}
        self._load()

    def _load(self):
        fp = self._dir / "schedules.json"
        if fp.exists():
            try:
                for d in json.loads(fp.read_text(encoding="utf-8")):
                    self._tasks[d["name"]] = ScheduledTask(
                        name=d["name"],
                        schedule_type=ScheduleType(d["type"]),
                        cron_expr=d.get("cron", ""),
                        interval_seconds=d.get("interval", 0),
                        event_entity=d.get("entity", ""),
                        webhook_path=d.get("webhook", ""),
                        last_run=d.get("last_run", 0.0),
                        run_count=d.get("run_count", 0),
                        enabled=d.get("enabled", True),
                    )
            except Exception as e:
                logger.warning(f"Failed to load schedules: {e}")

    def _save(self):
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / "schedules.json").write_text(
            json.dumps([t.to_dict() for t in self._tasks.values()], indent=2), encoding="utf-8"
        )

    def register(self, task, handler):
        self._tasks[task.name] = task
        self._handlers[task.name] = handler
        self._save()

    def remove(self, name):
        self._tasks.pop(name, None)
        self._handlers.pop(name, None)
        self._save()

    async def fire_event(self, entity, payload=None):
        count = 0
        for task in list(self._tasks.values()):
            if task.schedule_type == ScheduleType.EVENT and task.event_entity == entity:
                if task.name in self._handlers:
                    try:
                        handler = self._handlers[task.name]
                        if asyncio.iscoroutinefunction(handler):
                            await handler(payload)
                        else:
                            handler(payload)
                        task.run_count += 1
                        task.last_run = time.time()
                        count += 1
                    except Exception as e:
                        logger.error(f"Scheduled task '{task.name}' failed: {e}")
        self._save()
        return count

    async def run_cycle(self):
        count = 0
        now = time.time()
        for task in list(self._tasks.values()):
            if not task.enabled:
                continue
            should_run = False
            if task.schedule_type == ScheduleType.CRON:
                should_run = CronParser.matches(task.cron_expr)
            elif task.schedule_type == ScheduleType.INTERVAL:
                should_run = (now - task.last_run) >= task.interval_seconds
            elif task.schedule_type == ScheduleType.ONCE:
                should_run = task.run_count == 0
            if should_run and task.name in self._handlers:
                try:
                    handler = self._handlers[task.name]
                    if asyncio.iscoroutinefunction(handler):
                        await handler()
                    else:
                        handler()
                    task.run_count += 1
                    task.last_run = now
                    count += 1
                except Exception as e:
                    logger.error(f"Scheduled task '{task.name}' failed: {e}")
        self._save()
        return count

    def webhook_handler(self, path):
        for task in self._tasks.values():
            if task.schedule_type == ScheduleType.WEBHOOK and task.webhook_path == path:
                return self._handlers.get(task.name)
        return None

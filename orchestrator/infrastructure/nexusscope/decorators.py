"""Decorators for NexusScope profiling."""

from __future__ import annotations
import functools

from .profiler import get_profiler


def profile_sync(name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            with profiler.session(name or func.__name__):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def profile_async(name=None):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            profiler = get_profiler()
            async with profiler.async_session(name or func.__name__):
                return await func(*args, **kwargs)

        return wrapper

    return decorator

"""
Dependency Injection Container
==============================

Lightweight DI container for managing dependencies and their lifecycle.

Usage:
    from orchestrator.container import Container, singleton, transient

    container = Container()

    # Register services
    container.register_singleton(Cache, DiskCache)
    container.register_transient(Validator, MyValidator)

    # Resolve
    cache = container.resolve(Cache)
    validator = container.resolve(Validator)
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, TypeVar, get_type_hints

from .log_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

T = TypeVar('T')


class Lifecycle(Enum):
    """Dependency lifecycle types."""
    SINGLETON = auto()
    SCOPED = auto()
    TRANSIENT = auto()


@dataclass
class Registration:
    """Registration of a service in the container."""
    interface: type
    implementation: type | None = None
    factory: Callable[..., Any] | None = None
    instance: Any | None = None
    lifecycle: Lifecycle = Lifecycle.TRANSIENT
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.implementation is None and self.factory is None and self.instance is None:
            self.implementation = self.interface


class Scope:
    """A scope for managing scoped dependencies."""

    def __init__(self, container: Container, name: str = "default"):
        self.container = container
        self.name = name
        self._instances: dict[type, Any] = {}
        self._disposables: list[Any] = []

    def resolve(self, interface: type[T]) -> T:
        """Resolve a dependency within this scope."""
        registration = self.container._registrations.get(interface)

        if registration is None:
            raise KeyError(f"No registration found for {interface}")

        if registration.lifecycle == Lifecycle.SCOPED:
            if interface in self._instances:
                return self._instances[interface]

            instance = self.container._create_instance(registration, self)
            self._instances[interface] = instance

            if hasattr(instance, 'close') or hasattr(instance, '__aexit__'):
                self._disposables.append(instance)

            return instance

        return self.container.resolve(interface)

    async def close(self) -> None:
        """Close the scope and dispose all scoped instances."""
        for instance in self._disposables:
            try:
                if hasattr(instance, '__aexit__'):
                    await instance.__aexit__(None, None, None)
                elif hasattr(instance, '__exit__'):
                    instance.__exit__(None, None, None)
                elif hasattr(instance, 'close'):
                    if inspect.iscoroutinefunction(instance.close):
                        await instance.close()
                    else:
                        instance.close()
            except Exception as e:
                logger.error(f"Error disposing instance: {e}")

        self._instances.clear()
        self._disposables.clear()


class Container:
    """Lightweight dependency injection container."""

    def __init__(self):
        self._registrations: dict[type, Registration] = {}
        self._singletons: dict[type, Any] = {}
        self._scopes: list[Scope] = []

    def register(
        self,
        interface: type[T],
        implementation: type | None = None,
        lifecycle: Lifecycle = Lifecycle.TRANSIENT,
        **kwargs
    ) -> Registration:
        """Register a service."""
        registration = Registration(
            interface=interface,
            implementation=implementation,
            lifecycle=lifecycle,
            kwargs=kwargs,
        )
        self._registrations[interface] = registration
        logger.debug(f"Registered {interface.__name__} with {lifecycle.name}")
        return registration

    def register_singleton(
        self,
        interface: type[T],
        implementation: type | None = None,
        **kwargs
    ) -> Registration:
        """Register as singleton."""
        return self.register(interface, implementation, Lifecycle.SINGLETON, **kwargs)

    def register_scoped(
        self,
        interface: type[T],
        implementation: type | None = None,
        **kwargs
    ) -> Registration:
        """Register as scoped."""
        return self.register(interface, implementation, Lifecycle.SCOPED, **kwargs)

    def register_transient(
        self,
        interface: type[T],
        implementation: type | None = None,
        **kwargs
    ) -> Registration:
        """Register as transient."""
        return self.register(interface, implementation, Lifecycle.TRANSIENT, **kwargs)

    def register_factory(
        self,
        interface: type[T],
        factory: Callable[..., T],
        lifecycle: Lifecycle = Lifecycle.TRANSIENT,
    ) -> Registration:
        """Register using a factory function."""
        registration = Registration(
            interface=interface,
            factory=factory,
            lifecycle=lifecycle,
        )
        self._registrations[interface] = registration
        return registration

    def register_instance(
        self,
        interface: type[T],
        instance: T,
    ) -> Registration:
        """Register a pre-created instance as singleton."""
        registration = Registration(
            interface=interface,
            instance=instance,
            lifecycle=Lifecycle.SINGLETON,
        )
        self._registrations[interface] = registration
        self._singletons[interface] = instance
        return registration

    def resolve(self, interface: type[T], scope: Scope | None = None) -> T:
        """Resolve a dependency."""
        if scope:
            try:
                return scope.resolve(interface)
            except KeyError:
                pass

        registration = self._registrations.get(interface)
        if registration is None:
            raise KeyError(f"No registration found for {interface}")

        if registration.lifecycle == Lifecycle.SINGLETON and interface in self._singletons:
            return self._singletons[interface]

        instance = self._create_instance(registration, scope)

        if registration.lifecycle == Lifecycle.SINGLETON:
            self._singletons[interface] = instance

        return instance

    def try_resolve(self, interface: type[T], scope: Scope | None = None) -> T | None:
        """Try to resolve, return None if not registered."""
        try:
            return self.resolve(interface, scope)
        except KeyError:
            return None

    def resolve_all(self, interface: type[T]) -> list[T]:
        """Resolve all implementations."""
        instance = self.try_resolve(interface)
        return [instance] if instance else []

    def _create_instance(
        self,
        registration: Registration,
        scope: Scope | None = None,
    ) -> Any:
        """Create an instance based on registration."""
        if registration.instance is not None:
            return registration.instance

        if registration.factory is not None:
            return self._invoke_with_injection(
                registration.factory,
                scope,
                registration.kwargs,
            )

        impl = registration.implementation or registration.interface
        return self._create_class_instance(impl, scope, registration.kwargs)

    def _create_class_instance(
        self,
        cls: type,
        scope: Scope | None,
        additional_kwargs: dict[str, Any],
    ) -> Any:
        """Create class instance with constructor injection."""
        try:
            sig = inspect.signature(cls.__init__)
            type_hints = get_type_hints(cls.__init__)
        except (ValueError, TypeError):
            return cls(**additional_kwargs)

        kwargs = dict(additional_kwargs)

        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            if name in kwargs:
                continue

            param_type = type_hints.get(name)
            if param_type and param_type in self._registrations:
                try:
                    kwargs[name] = self.resolve(param_type, scope)
                except KeyError:
                    if param.default is inspect.Parameter.empty:
                        raise
            elif param.default is not inspect.Parameter.empty:
                pass
            elif param.default is inspect.Parameter.empty:
                raise KeyError(f"Cannot resolve parameter {name} for {cls}")

        return cls(**kwargs)

    def _invoke_with_injection(
        self,
        func: Callable,
        scope: Scope | None,
        additional_kwargs: dict[str, Any],
    ) -> Any:
        """Invoke a function with parameter injection."""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        kwargs = dict(additional_kwargs)

        for name, param in sig.parameters.items():
            if name in kwargs:
                continue

            param_type = type_hints.get(name)
            if param_type and param_type in self._registrations:
                try:
                    kwargs[name] = self.resolve(param_type, scope)
                except KeyError:
                    if param.default is inspect.Parameter.empty:
                        raise
            elif param.default is not inspect.Parameter.empty:
                pass
            elif param.default is inspect.Parameter.empty:
                raise KeyError(f"Cannot resolve parameter {name} for {func}")

        return func(**kwargs)

    def create_scope(self, name: str = "scoped") -> Scope:
        """Create a new scope."""
        scope = Scope(self, name)
        self._scopes.append(scope)
        return scope

    def is_registered(self, interface: type) -> bool:
        """Check if interface is registered."""
        return interface in self._registrations

    def unregister(self, interface: type) -> None:
        """Unregister a service."""
        if interface in self._registrations:
            del self._registrations[interface]
        if interface in self._singletons:
            del self._singletons[interface]

    def clear(self) -> None:
        """Clear all registrations."""
        self._registrations.clear()
        self._singletons.clear()
        self._scopes.clear()

    def build_provider(self) -> ServiceProvider:
        """Create immutable service provider."""
        return ServiceProvider(self)


class ServiceProvider:
    """Immutable, thread-safe service provider."""

    def __init__(self, container: Container):
        self._container = container

    def get_service(self, interface: type[T]) -> T:
        """Get a service."""
        return self._container.resolve(interface)

    def get_required_service(self, interface: type[T]) -> T:
        """Get a service or raise."""
        if not self._container.is_registered(interface):
            raise KeyError(f"Service {interface} not registered")
        return self._container.resolve(interface)

    def get_services(self, interface: type[T]) -> list[T]:
        """Get all services of type."""
        return self._container.resolve_all(interface)


# Global Container
_default_container: Container | None = None


def get_container() -> Container:
    """Get global container."""
    global _default_container
    if _default_container is None:
        _default_container = Container()
    return _default_container


def reset_container() -> None:
    """Reset global container."""
    global _default_container
    _default_container = None


def configure_services(configure: Callable[[Container], None]) -> Container:
    """Configure services using the global container."""
    container = get_container()
    configure(container)
    return container

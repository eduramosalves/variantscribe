"""Optional Langfuse tracing.

Tracing is strictly opt-in: with no Langfuse keys configured, `observe` is the identity
decorator and nothing is imported or sent. This keeps the graph runnable (and CI green)
without observability infra, while giving full per-node traces when keys are present.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

from variantscribe.config import settings

F = TypeVar("F", bound=Callable)


def tracing_enabled() -> bool:
    return bool(
        (settings.langfuse_public_key or os.environ.get("LANGFUSE_PUBLIC_KEY"))
        and (settings.langfuse_secret_key or os.environ.get("LANGFUSE_SECRET_KEY"))
    )


def _export_env() -> None:
    """Mirror settings into the env vars the Langfuse SDK reads."""
    if settings.langfuse_public_key:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    if settings.langfuse_secret_key:
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)


def observe(fn: F) -> F:
    """Decorator: trace `fn` via Langfuse when configured, else return it unchanged."""
    if not tracing_enabled():
        return fn
    _export_env()
    try:  # Support both Langfuse v3 (`langfuse.observe`) and v2 (`langfuse.decorators`).
        try:
            from langfuse import observe as _lf_observe
        except ImportError:
            from langfuse.decorators import observe as _lf_observe
        return _lf_observe()(fn) if _needs_call(_lf_observe) else _lf_observe(fn)
    except Exception:
        return fn


def _needs_call(observe_obj) -> bool:
    # langfuse.decorators.observe is used as @observe() (a factory); be tolerant.
    return getattr(observe_obj, "__name__", "") == "observe"


def flush() -> None:
    """Flush buffered traces (call once at the end of a run)."""
    if not tracing_enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass

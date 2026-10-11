# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""OBSERVE / ALERTS (enterprise observability).

`OBSERVE` and `ALERTS` are each a single command name with a subcommand as
the first argument (`OBSERVE INFO`, `ALERTS TEST`); the "command name" passed
to `execute_command` simply contains a space.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._shared import CallbackMap, ReadonlySet, named_fields

__all__ = ["AdminCommands", "CALLBACKS", "READONLY"]


def _cb_observe_info(reply: Any, **_options: Any) -> Dict[str, Any]:
    return named_fields(reply, texts={"enterprise_features"})


def _cb_alerts_info(reply: Any, **_options: Any) -> Dict[str, Any]:
    info = named_fields(
        reply, texts={"webhook_url", "provider", "supported_providers", "min_severity"}
    )
    info["enabled"] = bool(info.get("enabled"))
    info["flag_enabled"] = bool(info.get("flag_enabled"))
    return info


def _cb_alerts_ok(reply: Any, **_options: Any) -> bool:
    return str(reply).startswith("OK")


CALLBACKS: CallbackMap = {
    "OBSERVE INFO": _cb_observe_info,
    "OBSERVE METRICS": lambda reply, **_: named_fields(reply),
    "ALERTS INFO": _cb_alerts_info,
    "ALERTS TEST": _cb_alerts_ok,
    "ALERTS SETURL": _cb_alerts_ok,
    "ALERTS SETSEVERITY": _cb_alerts_ok,
    "ALERTS ON": _cb_alerts_ok,
    "ALERTS OFF": _cb_alerts_ok,
}

READONLY: ReadonlySet = frozenset(
    {
        "OBSERVE INFO",
        "OBSERVE METRICS",
        "ALERTS INFO",
    }
)


class AdminCommands:
    """Typed wrappers for OBSERVE and ALERTS."""

    def observe_info(self) -> Dict[str, Any]:
        """Return uptime, client and command counters, and memory use."""
        return self.execute_command("OBSERVE INFO")

    def observe_metrics(self) -> Dict[str, Any]:
        """Return the fuller OBSERVE METRICS counter set."""
        return self.execute_command("OBSERVE METRICS")

    def alerts_info(self) -> Dict[str, Any]:
        """Return alerting configuration and delivery counters."""
        return self.execute_command("ALERTS INFO")

    def alerts_test(self, severity: Optional[str] = None, message: Optional[str] = None) -> bool:
        """Send a test alert. `True` if it was accepted for delivery."""
        args: List[Any] = []
        if severity is not None:
            args.append(severity)
        if message is not None:
            args.append(message)
        return self.execute_command("ALERTS TEST", *args)

    def alerts_seturl(self, url: str) -> bool:
        """Set the alert webhook URL (empty string disables alerting)."""
        return self.execute_command("ALERTS SETURL", url)

    def alerts_setseverity(self, severity: str) -> bool:
        """Set the minimum severity ("info"/"warning"/"critical") that is delivered."""
        return self.execute_command("ALERTS SETSEVERITY", severity)

    def alerts_enable(self) -> bool:
        """Enable alerting without clearing the webhook URL."""
        return self.execute_command("ALERTS ON")

    def alerts_disable(self) -> bool:
        """Disable alerting without clearing the webhook URL."""
        return self.execute_command("ALERTS OFF")

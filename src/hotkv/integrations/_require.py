# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Internal helper: turn a missing optional framework dependency into one
clear, actionable `ImportError` instead of whatever import error the
framework itself (or Python) would otherwise raise."""

from __future__ import annotations

import importlib


def require(module_name: str, package: str, extra: str) -> None:
    """Import `module_name`, or raise a clear `pip install` error.

    Called at the top of every `hotkv.integrations.<extra>` subpackage
    before anything else in it is imported, so a missing framework fails
    fast with a message that says exactly what to install.
    """
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{package} is required for hotkv.integrations.{extra}. "
            f'Install it with: pip install "hotkv[{extra}]"'
        ) from exc

"""Central logging configuration.

Call :func:`get_logger` from any module instead of calling
``logging.getLogger`` directly, so log level/format stay consistent and are
controllable via the ``ADAPTIVE_OPT_LOG_LEVEL`` environment variable
(see ``.env.example``).
"""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.environ.get("ADAPTIVE_OPT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger("adaptive_optimization")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ``adaptive_optimization`` root."""
    _configure_root()
    return logging.getLogger(f"adaptive_optimization.{name}")

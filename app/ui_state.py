"""
Pure helpers for the Streamlit entry point's session state.

Kept free of Streamlit imports so the rules can be unit tested directly.
"""
from __future__ import annotations

from typing import Any, Optional

from app.config.settings import DEFAULT_CONFIG_PATH

RESULTS_KEY = "run_results"


def upload_key(name: str, size: int) -> str:
    """Identity of an uploaded file. A different file or size is a different run."""
    return f"{name}::{size}"


def is_stale(stored: Optional[dict[str, Any]], current_key: Optional[str]) -> bool:
    """True if stored results belong to a different upload, or no upload is present.
    Results from a previous upload must never be shown for the current one."""
    if stored is None:
        return False
    return current_key is None or stored.get("upload_key") != current_key


def is_explicit_config(path: str) -> bool:
    """True when the user typed a configuration path other than the default."""
    return path.strip() != DEFAULT_CONFIG_PATH

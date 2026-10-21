"""Smoke test: the Streamlit entry point renders its initial state without exceptions.
Requires streamlit (installed via requirements.txt). Does not process video or load YOLO weights."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "streamlit_app.py"


def test_app_initial_render_has_no_exception():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("Upload a local video" in info.value for info in at.info)

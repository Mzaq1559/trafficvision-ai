"""Smoke tests for the Streamlit entry point. They do not upload a video,
process frames, or load YOLO weights."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "streamlit_app.py"


@pytest.fixture
def at(monkeypatch):
    monkeypatch.chdir(ROOT)  # the default configuration path is relative to the repo root
    return AppTest.from_file(str(APP), default_timeout=60)


def test_initial_render_asks_for_upload(at):
    at.run()
    assert not at.exception
    assert any("Upload a local video" in info.value for info in at.info)


def test_default_configuration_is_reported_as_loaded(at):
    at.run()
    captions = [c.value for c in at.sidebar.caption]
    assert any("Configuration loaded: configs/default.yaml" in c for c in captions)


def test_missing_explicit_configuration_is_reported_in_sidebar(at):
    at.run()
    at.sidebar.text_input[0].set_value("configs/does_not_exist.yaml").run()
    assert not at.exception
    assert any("not found" in e.value for e in at.sidebar.error)


def test_empty_configuration_field_means_default_file(at):
    at.run()
    at.sidebar.text_input[0].set_value("   ").run()
    assert not at.exception
    assert not any(e.value for e in at.sidebar.error)
    assert any("Configuration loaded: configs/default.yaml" in c.value for c in at.sidebar.caption)

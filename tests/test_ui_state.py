"""Session-state rules used by the Streamlit entry point."""
from app.ui_state import is_explicit_config, is_stale, upload_key


def test_upload_key_changes_with_file_identity():
    assert upload_key("a.mp4", 100) != upload_key("b.mp4", 100)
    assert upload_key("a.mp4", 100) != upload_key("a.mp4", 101)
    assert upload_key("a.mp4", 100) == upload_key("a.mp4", 100)


def test_no_stored_results_is_not_stale():
    assert is_stale(None, upload_key("a.mp4", 1)) is False


def test_results_for_same_upload_are_kept():
    stored = {"upload_key": upload_key("a.mp4", 1)}
    assert is_stale(stored, upload_key("a.mp4", 1)) is False


def test_results_for_a_different_upload_are_stale():
    stored = {"upload_key": upload_key("a.mp4", 1)}
    assert is_stale(stored, upload_key("b.mp4", 1)) is True


def test_removed_upload_makes_results_stale():
    stored = {"upload_key": upload_key("a.mp4", 1)}
    assert is_stale(stored, None) is True


def test_default_config_path_is_not_explicit():
    assert is_explicit_config("configs/default.yaml") is False
    assert is_explicit_config("  configs/default.yaml ") is False
    assert is_explicit_config("configs/other.yaml") is True

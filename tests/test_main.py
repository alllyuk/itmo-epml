"""Tests for main module."""

import pytest
from itmo_epml.main import main


def test_main_runs_without_error():
    """Test that main function runs without raising exceptions."""
    # This is a basic smoke test
    try:
        main()
    except Exception as e:
        pytest.fail(f"main() raised an exception: {e}")

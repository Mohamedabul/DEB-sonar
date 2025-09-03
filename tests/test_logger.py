import os
import time
import tempfile
from datetime import datetime, timedelta
import threading

import pytest

from src import logger


def test_cleanup_old_logs(tmp_path):
    """Old logs should be deleted, recent logs should remain."""
    old_log = tmp_path / "old.log"
    recent_log = tmp_path / "recent.log"

    # Create an old log file (10 days old)
    old_log.write_text("old log")
    old_time = datetime.now() - timedelta(days=10)
    os.utime(old_log, (old_time.timestamp(), old_time.timestamp()))

    # Create a recent log file (1 day old)
    recent_log.write_text("recent log")

    # Run cleanup
    logger.cleanup_old_logs(str(tmp_path), days=7)

    # Old log should be deleted, recent log should remain
    assert not old_log.exists()
    assert recent_log.exists()


def test_periodic_cleanup_runs_once(tmp_path):
    """Test that periodic_cleanup runs and deletes old logs once."""
    old_log = tmp_path / "to_delete.log"
    old_log.write_text("delete me")
    old_time = datetime.now() - timedelta(days=8)
    os.utime(old_log, (old_time.timestamp(), old_time.timestamp()))

    # Run periodic_cleanup in a short-lived thread
    cleanup_thread = threading.Thread(
        target=logger.periodic_cleanup,
        args=(1, str(tmp_path), 7),  # run every 1 second
        daemon=True
    )
    cleanup_thread.start()

    time.sleep(2)  # allow cleanup to run at least once
    assert not old_log.exists()


def test_cleanup_handles_non_files(tmp_path):
    """Ensure cleanup skips directories without crashing."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    try:
        logger.cleanup_old_logs(str(tmp_path), days=7)
    except Exception as e:
        pytest.fail(f"cleanup_old_logs raised an exception on directory: {e}")

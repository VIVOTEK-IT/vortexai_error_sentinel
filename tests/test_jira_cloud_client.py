import os
import sys
import unittest
from pathlib import Path

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from error_log_monitor.jira_cloud_client import JiraCloudClient


class CleanTracebackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = object.__new__(JiraCloudClient)

    def test_clean_traceback_removes_code_markers_and_newlines(self) -> None:
        raw = "{code:python}\n       Traceback\nline 1\nline 2 {code}"
        expected = "Traceback\nline 1\nline 2"
        self.assertEqual(self.client.clean_traceback(raw), expected)

    def test_clean_traceback_handles_missing_markers(self) -> None:
        raw = "Traceback only\nwithout markers"
        expected = "Traceback only\nwithout markers"
        self.assertEqual(self.client.clean_traceback(raw), expected)

    def test_clean_traceback_handles_only_prefix(self) -> None:
        raw = "{code:python} Traceback with only prefix"
        expected = "Traceback with only prefix"
        self.assertEqual(self.client.clean_traceback(raw), expected)

    def test_clean_traceback_handles_only_suffix(self) -> None:
        raw = "Traceback with only suffix {code}"
        expected = "Traceback with only suffix"
        self.assertEqual(self.client.clean_traceback(raw), expected)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

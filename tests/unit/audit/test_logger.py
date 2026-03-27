"""Tests for AuditLogger."""

import pytest
import json
import logging
from unittest.mock import MagicMock

from sklearn_meta.audit.logger import AuditLogger, FoldLog, TrialLog


class TestAuditLoggerLogFold:
    """Tests for AuditLogger.log_fold method."""

    def test_log_fold_stores_entry(self):
        """Verify log_fold stores entry."""
        logger = AuditLogger(console_level=logging.CRITICAL)  # Suppress output

        fold = MagicMock()
        fold.fold_idx = 0
        fold.repeat_idx = 0

        logger.log_fold(
            node_name="test_node",
            fold=fold,
            score=0.95,
            fit_time=10.0,
            params={"a": 1},
        )

        assert len(logger._fold_logs) == 1
        assert logger._fold_logs[0].node_name == "test_node"
        assert logger._fold_logs[0].score == 0.95

    def test_log_fold_with_extra(self):
        """Verify log_fold stores extra metadata."""
        logger = AuditLogger(console_level=logging.CRITICAL)

        fold = MagicMock()
        fold.fold_idx = 0
        fold.repeat_idx = 0

        logger.log_fold(
            node_name="test",
            fold=fold,
            score=0.9,
            fit_time=5.0,
            params={},
            extra={"custom_metric": 0.5},
        )

        assert logger._fold_logs[0].extra == {"custom_metric": 0.5}

class TestAuditLoggerLogTrial:
    """Tests for AuditLogger.log_trial method."""

    def test_log_trial_stores_entry(self):
        """Verify log_trial stores entry."""
        logger = AuditLogger(console_level=logging.CRITICAL)

        logger.log_trial(
            node_name="test_node",
            trial_id=5,
            params={"lr": 0.1},
            score=0.85,
            duration=30.0,
        )

        assert len(logger._trial_logs) == 1
        assert logger._trial_logs[0].node_name == "test_node"
        assert logger._trial_logs[0].trial_id == 5

class TestAuditLoggerGetFoldSummary:
    """Tests for AuditLogger.get_fold_summary method."""

    def test_summary_with_logs(self):
        """Verify summary calculations are correct."""
        logger = AuditLogger(console_level=logging.CRITICAL)

        # Add some fold logs directly
        logger._fold_logs = [
            FoldLog("node", 0, 0, 0.8, 10.0, {}, "2024-01-01"),
            FoldLog("node", 1, 0, 0.9, 15.0, {}, "2024-01-01"),
            FoldLog("node", 2, 0, 1.0, 20.0, {}, "2024-01-01"),
        ]

        summary = logger.get_fold_summary()

        assert summary["n_folds"] == 3
        assert summary["mean_score"] == pytest.approx(0.9)
        assert summary["total_time"] == 45.0
        assert summary["mean_time"] == 15.0

    def test_summary_filter_by_node(self):
        """Verify filtering by node name."""
        logger = AuditLogger(console_level=logging.CRITICAL)

        logger._fold_logs = [
            FoldLog("node1", 0, 0, 0.8, 10.0, {}, "2024-01-01"),
            FoldLog("node1", 1, 0, 0.9, 15.0, {}, "2024-01-01"),
            FoldLog("node2", 0, 0, 0.7, 5.0, {}, "2024-01-01"),
        ]

        summary = logger.get_fold_summary(node_name="node1")

        assert summary["n_folds"] == 2
        assert summary["mean_score"] == pytest.approx(0.85)

    def test_summary_std_score(self):
        """Verify standard deviation calculation."""
        logger = AuditLogger(console_level=logging.CRITICAL)

        logger._fold_logs = [
            FoldLog("node", 0, 0, 0.8, 10.0, {}, "2024-01-01"),
            FoldLog("node", 1, 0, 0.8, 10.0, {}, "2024-01-01"),
        ]

        summary = logger.get_fold_summary()

        assert summary["std_score"] == pytest.approx(0.0)


class TestAuditLoggerGetTrialSummary:
    """Tests for AuditLogger.get_trial_summary method."""

    def test_summary_with_logs(self):
        """Verify summary calculations are correct."""
        logger = AuditLogger(console_level=logging.CRITICAL)

        logger._trial_logs = [
            TrialLog("node", 0, {}, 0.5, 10.0, "2024-01-01"),
            TrialLog("node", 1, {}, 0.3, 15.0, "2024-01-01"),
            TrialLog("node", 2, {}, 0.4, 20.0, "2024-01-01"),
        ]

        summary = logger.get_trial_summary()

        assert summary["n_trials"] == 3
        assert summary["best_score"] == 0.3  # Lower is better
        assert summary["worst_score"] == 0.5
        assert summary["total_duration"] == 45.0

    def test_summary_filter_by_node(self):
        """Verify filtering by node name."""
        logger = AuditLogger(console_level=logging.CRITICAL)

        logger._trial_logs = [
            TrialLog("node1", 0, {}, 0.5, 10.0, "2024-01-01"),
            TrialLog("node1", 1, {}, 0.3, 15.0, "2024-01-01"),
            TrialLog("node2", 0, {}, 0.7, 5.0, "2024-01-01"),
        ]

        summary = logger.get_trial_summary(node_name="node1")

        assert summary["n_trials"] == 2
        assert summary["best_score"] == 0.3


class TestAuditLoggerExportLogs:
    """Tests for AuditLogger.export_logs method."""

    def test_export_includes_fold_logs(self, tmp_path):
        """Verify export includes fold logs."""
        logger = AuditLogger(console_level=logging.CRITICAL)

        logger._fold_logs = [
            FoldLog("node", 0, 0, 0.9, 10.0, {"a": 1}, "2024-01-01"),
        ]

        export_path = tmp_path / "export.json"
        logger.export_logs(str(export_path))

        with open(export_path) as f:
            data = json.load(f)

        assert len(data["fold_logs"]) == 1
        assert data["fold_logs"][0]["node_name"] == "node"
        assert data["fold_logs"][0]["score"] == 0.9

    def test_export_includes_trial_logs(self, tmp_path):
        """Verify export includes trial logs."""
        logger = AuditLogger(console_level=logging.CRITICAL)

        logger._trial_logs = [
            TrialLog("node", 5, {"lr": 0.1}, 0.8, 30.0, "2024-01-01"),
        ]

        export_path = tmp_path / "export.json"
        logger.export_logs(str(export_path))

        with open(export_path) as f:
            data = json.load(f)

        assert len(data["trial_logs"]) == 1
        assert data["trial_logs"][0]["trial_id"] == 5


class TestAuditLoggerFileLogging:
    """Tests for file logging functionality."""

    def test_logs_to_file(self, tmp_path):
        """Verify logs are written to file."""
        log_file = tmp_path / "test.log"
        logger = AuditLogger(
            log_file=str(log_file),
            console_level=logging.CRITICAL,
            file_level=logging.INFO,
        )

        fold = MagicMock()
        fold.fold_idx = 0
        fold.repeat_idx = 0

        logger.log_fold(
            node_name="test_node",
            fold=fold,
            score=0.95,
            fit_time=10.0,
            params={},
        )

        # Force flush
        for handler in logger._logger.handlers:
            handler.flush()

        with open(log_file) as f:
            content = f.read()

        assert "test_node" in content
        assert "0.95" in content

    def test_file_level_filtering(self, tmp_path):
        """Verify file level filtering works."""
        log_file = tmp_path / "test.log"
        logger = AuditLogger(
            log_file=str(log_file),
            console_level=logging.CRITICAL,
            file_level=logging.WARNING,  # Only warnings and above
        )

        logger._logger.info("Info message")
        logger._logger.warning("Warning message")

        for handler in logger._logger.handlers:
            handler.flush()

        with open(log_file) as f:
            content = f.read()

        assert "Info message" not in content
        assert "Warning message" in content

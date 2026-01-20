"""Tests for AuditLogger."""

import pytest
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from auto_sklearn.audit.logger import AuditLogger, FoldLog, TrialLog


class TestFoldLog:
    """Tests for FoldLog dataclass."""

    def test_create_fold_log(self):
        """Verify fold log can be created."""
        log = FoldLog(
            node_name="test_node",
            fold_idx=0,
            repeat_idx=0,
            score=0.95,
            fit_time=10.5,
            params={"n_estimators": 100},
            timestamp="2024-01-01T00:00:00",
        )

        assert log.node_name == "test_node"
        assert log.score == 0.95
        assert log.fit_time == 10.5

    def test_fold_log_with_extra(self):
        """Verify fold log with extra metadata."""
        log = FoldLog(
            node_name="test",
            fold_idx=0,
            repeat_idx=0,
            score=0.9,
            fit_time=5.0,
            params={},
            timestamp="2024-01-01",
            extra={"memory_mb": 256},
        )

        assert log.extra == {"memory_mb": 256}

    def test_fold_log_default_extra(self):
        """Verify default extra is empty dict."""
        log = FoldLog(
            node_name="test",
            fold_idx=0,
            repeat_idx=0,
            score=0.9,
            fit_time=5.0,
            params={},
            timestamp="2024-01-01",
        )

        assert log.extra == {}


class TestTrialLog:
    """Tests for TrialLog dataclass."""

    def test_create_trial_log(self):
        """Verify trial log can be created."""
        log = TrialLog(
            node_name="test_node",
            trial_id=5,
            params={"learning_rate": 0.1},
            score=0.85,
            duration=30.0,
            timestamp="2024-01-01T00:00:00",
        )

        assert log.node_name == "test_node"
        assert log.trial_id == 5
        assert log.score == 0.85


class TestAuditLoggerInit:
    """Tests for AuditLogger initialization."""

    def test_default_init(self):
        """Verify default initialization."""
        logger = AuditLogger()

        assert logger.name == "auto_sklearn"
        assert len(logger._fold_logs) == 0
        assert len(logger._trial_logs) == 0

    def test_custom_name(self):
        """Verify custom logger name."""
        logger = AuditLogger(name="my_logger")

        assert logger.name == "my_logger"

    def test_with_log_file(self, tmp_path):
        """Verify log file is created."""
        log_file = tmp_path / "test.log"
        logger = AuditLogger(log_file=str(log_file))

        assert logger.log_file == log_file

    def test_creates_log_directory(self, tmp_path):
        """Verify log directory is created."""
        log_file = tmp_path / "subdir" / "test.log"
        logger = AuditLogger(log_file=str(log_file))

        assert log_file.parent.exists()

    def test_repr(self):
        """Verify repr includes useful info."""
        logger = AuditLogger(name="test_logger")

        repr_str = repr(logger)

        assert "AuditLogger" in repr_str
        assert "test_logger" in repr_str


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

    def test_log_fold_sets_timestamp(self):
        """Verify log_fold sets timestamp."""
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
        )

        assert logger._fold_logs[0].timestamp is not None
        assert len(logger._fold_logs[0].timestamp) > 0


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

    def test_log_trial_sets_timestamp(self):
        """Verify log_trial sets timestamp."""
        logger = AuditLogger(console_level=logging.CRITICAL)

        logger.log_trial(
            node_name="test",
            trial_id=0,
            params={},
            score=0.9,
            duration=10.0,
        )

        assert logger._trial_logs[0].timestamp is not None


class TestAuditLoggerLogMethods:
    """Tests for various logging methods."""

    def test_log_layer_start(self, caplog):
        """Verify log_layer_start outputs message."""
        logger = AuditLogger(console_level=logging.INFO)

        with caplog.at_level(logging.INFO):
            logger.log_layer_start(0, ["node1", "node2"])

        assert "layer 1" in caplog.text.lower()
        assert "node1" in caplog.text

    def test_log_layer_complete(self, caplog):
        """Verify log_layer_complete outputs message."""
        logger = AuditLogger(console_level=logging.INFO)

        with caplog.at_level(logging.INFO):
            logger.log_layer_complete(0, {"node1": 0.95}, 10.0)

        assert "completed" in caplog.text.lower()
        assert "0.95" in caplog.text

    def test_log_node_start(self, caplog):
        """Verify log_node_start outputs message."""
        logger = AuditLogger(console_level=logging.INFO)

        with caplog.at_level(logging.INFO):
            logger.log_node_start("test_node")

        assert "test_node" in caplog.text

    def test_log_node_complete(self, caplog):
        """Verify log_node_complete outputs message."""
        logger = AuditLogger(console_level=logging.INFO)

        with caplog.at_level(logging.INFO):
            logger.log_node_complete("test_node", 0.95, {"a": 1}, 10.0)

        assert "test_node" in caplog.text
        assert "0.95" in caplog.text

    def test_log_warning(self, caplog):
        """Verify log_warning outputs warning."""
        logger = AuditLogger()

        with caplog.at_level(logging.WARNING):
            logger.log_warning("Test warning message")

        assert "test warning" in caplog.text.lower()

    def test_log_error(self, caplog):
        """Verify log_error outputs error."""
        logger = AuditLogger()

        with caplog.at_level(logging.ERROR):
            logger.log_error("Test error message")

        assert "test error" in caplog.text.lower()

    def test_log_error_with_exception(self, caplog):
        """Verify log_error includes exception info."""
        logger = AuditLogger()

        try:
            raise ValueError("Test exception")
        except ValueError as e:
            with caplog.at_level(logging.ERROR):
                logger.log_error("Error occurred", exc=e)

        assert "error occurred" in caplog.text.lower()


class TestAuditLoggerGetFoldSummary:
    """Tests for AuditLogger.get_fold_summary method."""

    def test_empty_summary(self):
        """Verify empty summary for no logs."""
        logger = AuditLogger(console_level=logging.CRITICAL)

        summary = logger.get_fold_summary()

        assert summary == {}

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

    def test_empty_summary(self):
        """Verify empty summary for no logs."""
        logger = AuditLogger(console_level=logging.CRITICAL)

        summary = logger.get_trial_summary()

        assert summary == {}

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

    def test_export_creates_file(self, tmp_path):
        """Verify export creates file."""
        logger = AuditLogger(console_level=logging.CRITICAL)
        export_path = tmp_path / "export.json"

        logger.export_logs(str(export_path))

        assert export_path.exists()

    def test_export_valid_json(self, tmp_path):
        """Verify export creates valid JSON."""
        logger = AuditLogger(console_level=logging.CRITICAL)
        export_path = tmp_path / "export.json"

        logger.export_logs(str(export_path))

        with open(export_path) as f:
            data = json.load(f)

        assert "fold_logs" in data
        assert "trial_logs" in data

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


class TestAuditLoggerClear:
    """Tests for AuditLogger.clear method."""

    def test_clear_removes_all_logs(self):
        """Verify clear removes all logs."""
        logger = AuditLogger(console_level=logging.CRITICAL)

        logger._fold_logs = [
            FoldLog("node", 0, 0, 0.9, 10.0, {}, "2024-01-01"),
        ]
        logger._trial_logs = [
            TrialLog("node", 0, {}, 0.8, 10.0, "2024-01-01"),
        ]

        logger.clear()

        assert len(logger._fold_logs) == 0
        assert len(logger._trial_logs) == 0


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

"""AuditLogger: Logging for per-fold timing, scores, and debugging."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, TYPE_CHECKING

if TYPE_CHECKING:
    from auto_sklearn.core.data.cv import CVFold


@dataclass
class FoldLog:
    """Log entry for a single fold."""

    node_name: str
    fold_idx: int
    repeat_idx: int
    score: float
    fit_time: float
    params: Dict[str, Any]
    timestamp: str
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialLog:
    """Log entry for an optimization trial."""

    node_name: str
    trial_id: int
    params: Dict[str, Any]
    score: float
    duration: float
    timestamp: str


class AuditLogger:
    """
    Logger for tracking tuning and training progress.

    Records:
    - Per-fold timing and scores
    - Optimization trial history
    - Warnings and errors
    - Resource usage (optional)
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        name: str = "auto_sklearn",
    ) -> None:
        """
        Initialize the audit logger.

        Args:
            log_file: Path to log file (optional).
            console_level: Logging level for console output.
            file_level: Logging level for file output.
            name: Logger name.
        """
        self.name = name
        self._fold_logs: List[FoldLog] = []
        self._trial_logs: List[TrialLog] = []

        # Set up Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self._logger.addHandler(console_handler)

        # File handler
        if log_file:
            self.log_file = Path(log_file)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self._logger.addHandler(file_handler)
        else:
            self.log_file = None

    def log_fold(
        self,
        node_name: str,
        fold: CVFold,
        score: float,
        fit_time: float,
        params: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log results from a CV fold.

        Args:
            node_name: Name of the model node.
            fold: The CV fold.
            score: Validation score.
            fit_time: Time to fit in seconds.
            params: Parameters used.
            extra: Additional metadata.
        """
        log_entry = FoldLog(
            node_name=node_name,
            fold_idx=fold.fold_idx,
            repeat_idx=fold.repeat_idx,
            score=score,
            fit_time=fit_time,
            params=params,
            timestamp=datetime.now().isoformat(),
            extra=extra or {},
        )
        self._fold_logs.append(log_entry)

        self._logger.info(
            f"[{node_name}] Fold {fold.fold_idx}: score={score:.4f}, "
            f"time={fit_time:.2f}s"
        )

    def log_trial(
        self,
        node_name: str,
        trial_id: int,
        params: Dict[str, Any],
        score: float,
        duration: float,
    ) -> None:
        """
        Log an optimization trial.

        Args:
            node_name: Name of the model node.
            trial_id: Trial number.
            params: Parameters tried.
            score: Objective value.
            duration: Trial duration in seconds.
        """
        log_entry = TrialLog(
            node_name=node_name,
            trial_id=trial_id,
            params=params,
            score=score,
            duration=duration,
            timestamp=datetime.now().isoformat(),
        )
        self._trial_logs.append(log_entry)

        self._logger.debug(
            f"[{node_name}] Trial {trial_id}: score={score:.4f}, "
            f"duration={duration:.2f}s"
        )

    def log_layer_start(self, layer_idx: int, nodes: List[str]) -> None:
        """Log the start of a layer."""
        self._logger.info(f"Starting layer {layer_idx + 1}: {', '.join(nodes)}")

    def log_layer_complete(
        self,
        layer_idx: int,
        scores: Dict[str, float],
        duration: float,
    ) -> None:
        """Log completion of a layer."""
        scores_str = ", ".join(f"{k}={v:.4f}" for k, v in scores.items())
        self._logger.info(
            f"Completed layer {layer_idx + 1}: {scores_str} ({duration:.1f}s)"
        )

    def log_node_start(self, node_name: str) -> None:
        """Log the start of fitting a node."""
        self._logger.info(f"Fitting node: {node_name}")

    def log_node_complete(
        self,
        node_name: str,
        best_score: float,
        best_params: Dict[str, Any],
        duration: float,
    ) -> None:
        """Log completion of a node."""
        self._logger.info(
            f"Completed {node_name}: best_score={best_score:.4f}, "
            f"duration={duration:.1f}s"
        )
        self._logger.debug(f"Best params for {node_name}: {best_params}")

    def log_warning(self, message: str) -> None:
        """Log a warning."""
        self._logger.warning(message)

    def log_error(self, message: str, exc: Optional[Exception] = None) -> None:
        """Log an error."""
        if exc:
            self._logger.error(f"{message}: {exc}", exc_info=True)
        else:
            self._logger.error(message)

    def get_fold_summary(self, node_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary statistics for fold logs.

        Args:
            node_name: Filter by node name (optional).

        Returns:
            Dictionary with summary statistics.
        """
        logs = self._fold_logs
        if node_name:
            logs = [l for l in logs if l.node_name == node_name]

        if not logs:
            return {}

        scores = [l.score for l in logs]
        times = [l.fit_time for l in logs]

        return {
            "n_folds": len(logs),
            "mean_score": sum(scores) / len(scores),
            "std_score": (
                sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)
            )
            ** 0.5,
            "total_time": sum(times),
            "mean_time": sum(times) / len(times),
        }

    def get_trial_summary(self, node_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary statistics for trial logs.

        Args:
            node_name: Filter by node name (optional).

        Returns:
            Dictionary with summary statistics.
        """
        logs = self._trial_logs
        if node_name:
            logs = [l for l in logs if l.node_name == node_name]

        if not logs:
            return {}

        scores = [l.score for l in logs]
        durations = [l.duration for l in logs]

        return {
            "n_trials": len(logs),
            "best_score": min(scores),
            "worst_score": max(scores),
            "total_duration": sum(durations),
        }

    def export_logs(self, path: str) -> None:
        """
        Export all logs to a JSON file.

        Args:
            path: Output file path.
        """
        export_data = {
            "fold_logs": [
                {
                    "node_name": l.node_name,
                    "fold_idx": l.fold_idx,
                    "repeat_idx": l.repeat_idx,
                    "score": l.score,
                    "fit_time": l.fit_time,
                    "params": l.params,
                    "timestamp": l.timestamp,
                    "extra": l.extra,
                }
                for l in self._fold_logs
            ],
            "trial_logs": [
                {
                    "node_name": l.node_name,
                    "trial_id": l.trial_id,
                    "params": l.params,
                    "score": l.score,
                    "duration": l.duration,
                    "timestamp": l.timestamp,
                }
                for l in self._trial_logs
            ],
        }

        with open(path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

    def clear(self) -> None:
        """Clear all stored logs."""
        self._fold_logs.clear()
        self._trial_logs.clear()

    def __repr__(self) -> str:
        return (
            f"AuditLogger(name={self.name}, folds={len(self._fold_logs)}, "
            f"trials={len(self._trial_logs)})"
        )

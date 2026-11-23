"""JSON-lines experiment tracker.

Lightweight experiment logging to JSONL files.
"""

import json
import time
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from src.config import PROJECT_ROOT


EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


class ExperimentTracker:
    """Track experiments using JSON-lines format."""

    def __init__(
        self,
        experiment_name: str,
        experiments_dir: Optional[Path] = None,
    ):
        self.experiment_name = experiment_name
        self.experiments_dir = experiments_dir or EXPERIMENTS_DIR
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.experiments_dir / f"{experiment_name}.jsonl"
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters for this experiment run."""
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics for this experiment run."""
        self.metrics.update(metrics)

    def log_step(self, step: int, metrics: dict[str, float]) -> None:
        """Log metrics for a specific training step."""
        entry = {
            "run_id": self.run_id,
            "experiment": self.experiment_name,
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        self._append_entry(entry)

    def save(self) -> None:
        """Save the complete experiment run to the log file."""
        elapsed = time.time() - self.start_time

        entry = {
            "run_id": self.run_id,
            "experiment": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "params": self.params,
            "metrics": self.metrics,
        }
        self._append_entry(entry)

    def _append_entry(self, entry: dict) -> None:
        """Append a JSON entry to the log file."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    @classmethod
    def load_runs(cls, experiment_name: str, experiments_dir: Optional[Path] = None) -> list[dict]:
        """Load all runs for an experiment.

        Args:
            experiment_name: Name of the experiment
            experiments_dir: Directory containing experiment logs

        Returns:
            List of run dictionaries
        """
        experiments_dir = experiments_dir or EXPERIMENTS_DIR
        log_file = experiments_dir / f"{experiment_name}.jsonl"

        if not log_file.exists():
            return []

        runs = []
        with open(log_file) as f:
            for line in f:
                runs.append(json.loads(line.strip()))

        return runs

    @classmethod
    def get_best_run(
        cls,
        experiment_name: str,
        metric: str,
        maximize: bool = True,
        experiments_dir: Optional[Path] = None,
    ) -> Optional[dict]:
        """Get the best run for an experiment based on a metric.

        Args:
            experiment_name: Name of the experiment
            metric: Metric name to optimize
            maximize: If True, maximize metric; else minimize
            experiments_dir: Directory containing experiment logs

        Returns:
            Best run dictionary or None
        """
        runs = cls.load_runs(experiment_name, experiments_dir)

        valid_runs = [
            r for r in runs
            if "metrics" in r and metric in r["metrics"]
        ]

        if not valid_runs:
            return None

        if maximize:
            return max(valid_runs, key=lambda r: r["metrics"][metric])
        else:
            return min(valid_runs, key=lambda r: r["metrics"][metric])

    @classmethod
    def summary(cls, experiments_dir: Optional[Path] = None) -> list[dict]:
        """Get summary of all experiments.

        Args:
            experiments_dir: Directory containing experiment logs

        Returns:
            List of experiment summaries
        """
        experiments_dir = experiments_dir or EXPERIMENTS_DIR

        if not experiments_dir.exists():
            return []

        summaries = []
        for log_file in experiments_dir.glob("*.jsonl"):
            runs = cls.load_runs(log_file.stem, experiments_dir)
            if runs:
                last_run = runs[-1]
                summaries.append({
                    "experiment": log_file.stem,
                    "n_runs": len(runs),
                    "last_run_id": last_run.get("run_id"),
                    "last_timestamp": last_run.get("timestamp"),
                    "metrics": last_run.get("metrics", {}),
                })

        return summaries

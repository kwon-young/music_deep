import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class BaseMetrics:
    """Base class for all experiment metrics."""
    step: int
    timestamp: float = field(default_factory=time.time, init=False)


class ExperimentLogger:
    def __init__(self, exp_dir: Path, stage_name: str):
        """
        Initializes the logger, creating the necessary directory structure.
        """
        self.exp_dir = exp_dir
        self.stage_dir = self.exp_dir / stage_name
        
        self.metrics_file = self.stage_dir / "metrics.jsonl"
        self.checkpoint_dir = self.stage_dir / "checkpoints"
        self.vis_dir = self.stage_dir / "visualizations"
        
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

    def log_metrics(self, metrics: BaseMetrics) -> None:
        """
        Appends a strongly-typed metric dataclass as a JSON line.
        """
        with self.metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")

    def get_checkpoint_dir(self) -> Path:
        return self.checkpoint_dir

    def get_visualizations_dir(self) -> Path:
        return self.vis_dir

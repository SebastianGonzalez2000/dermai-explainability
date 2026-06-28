from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Phase:
    name: str
    epochs: int
    lr: float
    unfreeze_backbone: bool


@dataclass(frozen=True)
class Config:
    model_id: str
    phases: list[Phase]
    data_dir: Path
    output_dir: Path
    batch_size: int = 32
    num_workers: int = 0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text())
        phases = [Phase(**p) for p in raw.pop("phases")]
        raw["data_dir"] = Path(raw["data_dir"])
        raw["output_dir"] = Path(raw["output_dir"])
        return cls(phases=phases, **raw)

    @property
    def run_name(self) -> str:
        return self.model_id.split("/")[-1]

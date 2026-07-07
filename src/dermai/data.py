from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
LABEL_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}


class HAM10000Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_paths: dict[str, Path], processor) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        image = Image.open(self.image_paths[row.image_id]).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt")["pixel_values"][0]
        label = torch.tensor(LABEL_TO_INDEX[row.dx], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": label, "image_id": row.image_id}


class DataModule:
    def __init__(self, data_dir: Path, processor, batch_size: int, num_workers: int, seed: int) -> None:
        self.data_dir = data_dir
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.splits: dict[str, pd.DataFrame] = {}

    def setup(self) -> None:
        metadata = pd.read_csv(self.data_dir / "HAM10000_metadata", sep=None, engine="python")
        image_paths = self._index_images()
        metadata = metadata[metadata.image_id.isin(image_paths)].reset_index(drop=True)
        self.image_paths = image_paths
        train_val, test = self._group_split(metadata, test_fraction_splits=10)
        train, val = self._group_split(train_val, test_fraction_splits=9)
        self.splits = {"train": train, "val": val, "test": test}

    def _index_images(self) -> dict[str, Path]:
        return {path.stem: path for path in self.data_dir.glob("*.jpg")}

    def _group_split(self, frame: pd.DataFrame, test_fraction_splits: int):
        splitter = StratifiedGroupKFold(n_splits=test_fraction_splits, shuffle=True, random_state=self.seed)
        keep_idx, hold_idx = next(splitter.split(frame, frame.dx, groups=frame.lesion_id))
        return frame.iloc[keep_idx].reset_index(drop=True), frame.iloc[hold_idx].reset_index(drop=True)

    def class_weights(self) -> torch.Tensor:
        counts = self.splits["train"].dx.value_counts()
        total = counts.sum()
        weights = [total / (len(CLASSES) * counts[name]) for name in CLASSES]
        return torch.tensor(weights, dtype=torch.float)

    def loader(self, split: str) -> DataLoader:
        dataset = HAM10000Dataset(self.splits[split], self.image_paths, self.processor)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=self.num_workers,
        )

    def split_sizes(self) -> dict[str, int]:
        return {name: len(frame) for name, frame in self.splits.items()}

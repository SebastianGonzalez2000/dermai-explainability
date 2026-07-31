from __future__ import annotations

import logging
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

from .data import CLASSES, LABEL_TO_INDEX

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

BLOCK_PREFIXES = ("efficientnet.encoder.blocks.", "vit.layers.")
BLOCK_SPANS = {"early": (0.0, 1 / 3), "middle": (1 / 3, 2 / 3), "late": (2 / 3, 1.0)}


def block_index(parameter_name: str) -> int | None:
    for prefix in BLOCK_PREFIXES:
        if parameter_name.startswith(prefix):
            return int(parameter_name[len(prefix):].split(".")[0])
    return None


class ModelFactory:
    @staticmethod
    def build(model_id: str) -> nn.Module:
        return AutoModelForImageClassification.from_pretrained(
            model_id,
            num_labels=len(CLASSES),
            id2label={i: name for name, i in LABEL_TO_INDEX.items()},
            label2id=LABEL_TO_INDEX,
            ignore_mismatched_sizes=True,
        )

    @staticmethod
    def load(checkpoint: str) -> nn.Module:
        return AutoModelForImageClassification.from_pretrained(checkpoint)

    @staticmethod
    def processor(model_id: str):
        return AutoImageProcessor.from_pretrained(model_id)

    @staticmethod
    def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
        for name, parameter in model.named_parameters():
            if not name.startswith("classifier"):
                parameter.requires_grad = trainable

    @staticmethod
    def set_block_span_trainable(model: nn.Module, span: str) -> None:
        """Train only the blocks in one third of the backbone's depth, plus the classifier head."""
        indices = {i for i in (block_index(name) for name, _ in model.named_parameters()) if i is not None}
        if not indices:
            raise ValueError("no indexed backbone blocks found, cannot apply a surgical span")
        start_fraction, end_fraction = BLOCK_SPANS[span]
        depth = max(indices) + 1
        selected = range(round(start_fraction * depth), round(end_fraction * depth))
        for name, parameter in model.named_parameters():
            index = block_index(name)
            parameter.requires_grad = name.startswith("classifier") or index in selected

    @staticmethod
    def freeze_norm_statistics(model: nn.Module) -> None:
        """requires_grad=False leaves batch-norm running statistics updating, so hold the frozen ones in eval mode."""
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and not module.weight.requires_grad:
                module.eval()

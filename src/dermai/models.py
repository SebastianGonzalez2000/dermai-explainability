from __future__ import annotations

import logging
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

from .data import CLASSES, LABEL_TO_INDEX

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


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
    def cam_target_layer(model: nn.Module) -> nn.Module:
        """The last conv feature map before pooling, for each CNN architecture we support.

        Keeping this mapping here (rather than in the Grad-CAM framework itself) means
        adding a new CNN backbone only requires one line here; GradCAM stays architecture-agnostic.
        """
        backbone = getattr(model, "efficientnet", None)
        if backbone is not None:
            return backbone.encoder.top_activation
        raise ValueError(f"no Grad-CAM target layer registered for {type(model).__name__}")

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from .config import Config, Phase
from .data import DataModule
from .metrics import ClassificationMetrics
from .models import ModelFactory
from .utils import Timer, get_logger

logger = get_logger()


class Trainer:
    def __init__(self, model: nn.Module, data: DataModule, config: Config, device: torch.device) -> None:
        self.model = model.to(device)
        self.data = data
        self.config = config
        self.device = device
        self.use_amp = device.type == "cuda"
        self.loss_fn = nn.CrossEntropyLoss(weight=data.class_weights().to(device))
        self.train_loader = data.loader("train")
        self.val_loader = data.loader("val")
        self.test_loader = data.loader("test")
        self.best_metric = 0.0
        self.checkpoint_path = config.output_dir / config.run_name / "best.pt"

    def fit(self) -> None:
        run_timer = Timer()
        for phase in self.config.phases:
            self._run_phase(phase)
        logger.info("training complete in %s, best val macro-F1 %.4f", Timer.format(run_timer.elapsed()), self.best_metric)

    def test(self) -> dict[str, float]:
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        metrics = self.evaluate(self.test_loader)
        logger.info("test  macro_f1 %.4f  bal_acc %.4f", metrics["macro_f1"], metrics["balanced_accuracy"])
        return metrics

    def _run_phase(self, phase: Phase) -> None:
        ModelFactory.set_backbone_trainable(self.model, phase.unfreeze_backbone)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("phase '%s': %d epochs, lr %.0e, backbone %s, %d trainable params",
                    phase.name, phase.epochs, phase.lr, "unfrozen" if phase.unfreeze_backbone else "frozen", trainable)
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                          lr=phase.lr, weight_decay=self.config.weight_decay)
        steps = phase.epochs * len(self.train_loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(self.config.warmup_ratio * steps), steps)
        for epoch in range(1, phase.epochs + 1):
            self._train_epoch(optimizer, scheduler, phase.name, epoch)

    def _train_epoch(self, optimizer, scheduler, phase_name: str, epoch: int) -> None:
        self.model.train()
        timer = Timer()
        running_loss = 0.0
        for batch in self.train_loader:
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            optimizer.zero_grad()
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(pixel_values=pixel_values).logits
                loss = self.loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        train_loss = running_loss / len(self.train_loader)
        metrics = self.evaluate(self.val_loader)
        logger.info("[%s] epoch %d  train_loss %.4f  val_macro_f1 %.4f  val_bal_acc %.4f  (%s)",
                    phase_name, epoch, train_loss, metrics["macro_f1"], metrics["balanced_accuracy"], Timer.format(timer.elapsed()))
        self._save_if_best(metrics["macro_f1"])

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        self.model.eval()
        metrics = ClassificationMetrics()
        for batch in loader:
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            logits = self.model(pixel_values=pixel_values).logits
            metrics.update(logits, labels)
        return metrics.compute()

    def _save_if_best(self, metric: float) -> None:
        if metric <= self.best_metric:
            return
        self.best_metric = metric
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.checkpoint_path)

"""Report best validation macro-F1 per fine-tuning arm, read from the training logs."""

import re
import sys
from pathlib import Path

EPOCH_LINE = re.compile(
    r"\[(?P<phase>\w+)\] epoch (?P<epoch>\d+) .*val_macro_f1 (?P<f1>[\d.]+)\s+val_bal_acc (?P<acc>[\d.]+)"
)


def best_epoch(log: Path) -> dict | None:
    rows = [m.groupdict() for m in EPOCH_LINE.finditer(log.read_text())]
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r["f1"]))
    return {
        "arm": log.stem.replace("train_", ""),
        "epochs_done": len(rows),
        "phase": best["phase"],
        "epoch": int(best["epoch"]),
        "macro_f1": float(best["f1"]),
        "balanced_accuracy": float(best["acc"]),
    }


def main() -> None:
    logs = sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "outputs").glob("train_*.log"))
    results = [row for row in (best_epoch(log) for log in logs) if row]
    if not results:
        print("no training logs with completed epochs found")
        return
    results.sort(key=lambda r: r["macro_f1"], reverse=True)
    print(f"{'arm':38s} {'best val macro-F1':>18s} {'bal acc':>9s} {'at':>14s} {'epochs':>7s}")
    for row in results:
        at = f"{row['phase']} e{row['epoch']}"
        print(f"{row['arm']:38s} {row['macro_f1']:18.4f} {row['balanced_accuracy']:9.4f} {at:>14s} {row['epochs_done']:7d}")


if __name__ == "__main__":
    main()

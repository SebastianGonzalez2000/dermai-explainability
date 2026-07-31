"""Report best validation and final test metrics per fine-tuning arm, read from the training logs."""

import re
import sys
from pathlib import Path

EPOCH_LINE = re.compile(
    r"\[(?P<phase>\w+)\] epoch (?P<epoch>\d+) .*val_macro_f1 (?P<f1>[\d.]+)\s+val_bal_acc (?P<acc>[\d.]+)"
)
TEST_LINE = re.compile(r"test\s+macro_f1 (?P<f1>[\d.]+)\s+bal_acc (?P<acc>[\d.]+)")


def summarize(log: Path) -> dict | None:
    text = log.read_text()
    epochs = [m.groupdict() for m in EPOCH_LINE.finditer(text)]
    if not epochs:
        return None
    best = max(epochs, key=lambda r: float(r["f1"]))
    test = TEST_LINE.search(text)
    return {
        "arm": log.stem.replace("train_", ""),
        "epochs_done": len(epochs),
        "at": f"{best['phase']} e{best['epoch']}",
        "val_f1": float(best["f1"]),
        "val_acc": float(best["acc"]),
        "test_f1": float(test["f1"]) if test else None,
        "test_acc": float(test["acc"]) if test else None,
    }


def main() -> None:
    logs = sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "outputs").glob("train_*.log"))
    results = [row for row in (summarize(log) for log in logs) if row]
    if not results:
        print("no training logs with completed epochs found")
        return
    results.sort(key=lambda r: r["val_f1"], reverse=True)
    header = f"{'arm':34s} {'val F1':>8s} {'val bal':>8s} {'test F1':>8s} {'test bal':>9s} {'best at':>10s} {'epochs':>7s}"
    print(header)
    print("-" * len(header))
    for row in results:
        test_f1 = f"{row['test_f1']:.4f}" if row["test_f1"] is not None else "running"
        test_acc = f"{row['test_acc']:.4f}" if row["test_acc"] is not None else "running"
        print(f"{row['arm']:34s} {row['val_f1']:8.4f} {row['val_acc']:8.4f} {test_f1:>8s} {test_acc:>9s} "
              f"{row['at']:>10s} {row['epochs_done']:7d}")


if __name__ == "__main__":
    main()

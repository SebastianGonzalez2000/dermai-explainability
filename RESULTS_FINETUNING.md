# Fine-Tuning Strategy Experiments

Recorded 2026-07-31. Six runs comparing full fine-tuning against surgical
block-span fine-tuning on HAM10000, 7 classes.

## Results

Best validation epoch per run, and test metrics from that checkpoint.

| Model | Strategy | val macro-F1 | val bal acc | test macro-F1 | test bal acc | Best epoch | Wall clock |
|---|---|---|---|---|---|---|---|
| ViT-B/16 | full fine-tune | **0.7924** | 0.7691 | **0.7130** | 0.7173 | n/a | n/a |
| ViT-B/16 | early half | 0.7762 | 0.7885 | 0.6770 | 0.6964 | early e7 | 112m |
| ViT-B/16 | late half | 0.7572 | 0.7458 | 0.6324 | 0.6289 | late e6 | 96m |
| EfficientNet-B0 | late half | **0.7516** | 0.7471 | 0.6624 | 0.6723 | late e11 | 26m |
| EfficientNet-B0 | full fine-tune | 0.6872 | 0.6583 | 0.6464 | 0.6402 | full e15 | 36m |
| EfficientNet-B0 | early half | 0.6386 | 0.7009 | 0.5987 | 0.6688 | early e11 | 33m |

The ViT full fine-tune row comes from the checkpoint trained in an earlier
session, re-evaluated on the same splits; its per-epoch log was not retained.

## Findings

**The two architectures prefer opposite treatment.** Restricting updates to half
the depth helps the CNN, whose late-half arm beats full fine-tuning by 0.064 val
F1 and 0.016 test F1 while training 10 minutes faster. It hurts the ViT, where
full fine-tuning leads both halves by 0.036 to 0.081 test F1.

**The preferred half also flips between them.** The CNN ranks late > full >
early, so its output-side layers need freedom and disturbing its early texture
filters is the worst option. The ViT ranks full > early > late, so when
constrained it would rather adapt the patch-embedding side that has to learn
dermoscopic appearance.

**ViT-B/16 wins in every configuration**, 0.7130 versus 0.6624 best test F1.

**Every run overfits.** Train loss reached 0.0021 for the ViT late arm while val
F1 sat near 0.72, every best epoch landed at 6 to 11 of 15, and the val-to-test
drop is 0.10 to 0.13 F1 throughout. No augmentation is applied anywhere in the
pipeline, which is a larger lever on these numbers than any freezing choice.

## Caveats

Single seed per arm, so gaps below roughly 0.03 F1 are not separable from noise.

The CNN halves are lopsided in capacity, 7.9% of parameters in the early half
against 82.1% in the late half, because EfficientNet stacks most of its weight
near the output. Its early-versus-late gap therefore conflates depth with
capacity. The ViT split is even at 49.6% each, so only the ViT comparison
isolates depth.

## Reproducing

```
python train.py --config configs/efficientnet.yaml
python train.py --config configs/efficientnet_surgical_early.yaml
python train.py --config configs/efficientnet_surgical_late.yaml
python train.py --config configs/vit.yaml
python train.py --config configs/vit_surgical_early.yaml
python train.py --config configs/vit_surgical_late.yaml

python scripts/compare_arms.py
```

Every arm shares the same phase 1, 5 epochs head-only at lr 1e-3, and differs
only in which blocks phase 2 unfreezes over 15 epochs. Common settings: AdamW,
weight decay 0.01, 10% linear warmup, class-weighted cross-entropy, seed 42,
lesion-grouped stratified 80/10/10 split, batch 32 for the CNN and 16 for the
ViT, Apple Silicon MPS.

Surgical spans follow Lee et al., Surgical Fine-Tuning Improves Adaptation to
Distribution Shifts, ICLR 2023, arXiv:2210.11466. The two-stage schedule follows
Kumar et al., Fine-Tuning can Distort Pretrained Features and Underperform
Out-of-Distribution, ICLR 2022, arXiv:2202.10054.

## Related: batch-norm freezing fix

Commit `ec36820` on `main` stopped batch-norm running statistics from updating
while the backbone is frozen. Evaluating the pre-fix and post-fix CNN
checkpoints on identical splits:

| Checkpoint | val macro-F1 | val bal acc | test macro-F1 | test bal acc |
|---|---|---|---|---|
| pre-fix, statistics drifting | 0.6763 | 0.7024 | 0.6477 | 0.7119 |
| post-fix, statistics held | 0.6872 | 0.6583 | 0.6464 | 0.6402 |

Accuracy is unchanged within noise. The fix matters for correctness rather than
score: before it, the claim that phase 1 holds the pretrained features fixed was
true for the LayerNorm-based ViT but false for the CNN, an undisclosed asymmetry
on the axis this project compares.

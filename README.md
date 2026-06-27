# DermAI: Explainability in CNNs vs. Vision Transformers

Quantifying explanation faithfulness and localization in a CNN (EfficientNet-B0) and a Vision Transformer (ViT-B/16) for skin lesion classification on HAM10000.

CS 7643 Deep Learning, Georgia Tech.

## Proposal

See [proposal.pdf](proposal.pdf).

## Dataset

See [DATASET.md](DATASET.md) for an overview of HAM10000, its class imbalance, and the lesion segmentation masks we use as localization ground truth.

To download the data, run:

```bash
python scripts/download_data.py
```

This fetches HAM10000 from the Harvard Dataverse (about 2.8 GB), unzips it, and lays everything out under `data/` exactly as the project expects. Pass `--isic` to also download the optional ISIC 2018 test set. The `data/` directory is gitignored.

## Team

- Xiaoyan Xing
- Somayeh Bahrami
- Sebastian Gonzalez
- Ka Wing Ariel Lee

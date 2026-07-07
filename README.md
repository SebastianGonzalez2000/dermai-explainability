# DermAI: Explainability in CNNs vs. Vision Transformers

Quantifying explanation faithfulness and localization in a CNN (EfficientNet-B0) and a Vision Transformer (ViT-B/16) for skin lesion classification on HAM10000.

CS 7643 Deep Learning, Georgia Tech.

## Proposal

See [proposal.pdf](proposal.pdf).

## Setup

Install the dependencies into your Python environment:

```bash
pip install -r requirements.txt
```

The code runs unchanged on CUDA (GT PACE), Apple Silicon (MPS), and CPU; the device is detected automatically. On Apple Silicon, export `PYTORCH_ENABLE_MPS_FALLBACK=1` so any op without an MPS kernel falls back to CPU instead of erroring.

## Dataset

See [DATASET.md](DATASET.md) for an overview of HAM10000, its class imbalance, and the lesion segmentation masks we use as localization ground truth.

To download the data, run:

```bash
python scripts/download_data.py
```

This fetches HAM10000 from the Harvard Dataverse (about 2.8 GB), unzips it, and lays everything out under `data/` exactly as the project expects. Pass `--isic` to also download the optional ISIC 2018 test set. The `data/` directory is gitignored.

## Fine-tuning

Each run is driven entirely by a YAML config. Train a model end to end (two-stage fine-tuning followed by test-set evaluation) with:

```bash
python train.py --config configs/efficientnet.yaml
python train.py --config configs/vit.yaml
```

On Apple Silicon, prefix with the fallback flag:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --config configs/vit.yaml
```

Switching models means switching config files; no code changes. Each config defines the two-stage schedule under `phases` (stage 1 trains the new classification head with the backbone frozen, stage 2 unfreezes and fine-tunes end to end at a lower learning rate). Epochs, learning rates, batch size, and seed are all set in the YAML. Override the auto-detected device with `--device cpu|mps|cuda`.

The best checkpoint by validation macro-F1 is saved in Hugging Face format to `outputs/<model>/`, then restored for the final test-set evaluation. The `outputs/` directory is gitignored.

## Grad-CAM heatmaps

For the CNN (EfficientNet-B0), generate Grad-CAM heatmaps over a split's images with:

```bash
python explain_gradcam.py --config configs/efficientnet.yaml --checkpoint sgonzalez2000/dermai-efficientnet-b0
```

This writes one overlay PNG per image (named `<image_id>__true-<class>__pred-<class>.png`) to
`outputs/gradcam/<checkpoint-name>/test/` by default. Pass `--split train|val|test`, `--output` to
choose a different directory, or `--zip` to also produce a `.zip` archive of the output directory.
`ModelFactory.cam_target_layer` maps each supported CNN architecture to the conv layer Grad-CAM hooks
into; the `GradCAM` class itself (in `src/dermai/gradcam.py`) is architecture-agnostic.

## Publishing a fine-tuned model

After you have run the fine-tuning workflow, you can push your trained checkpoint to the Hugging Face Hub so downstream experiments can load it directly instead of retraining. Authenticate once with a write token, then run the upload script with the checkpoint folder name:

```bash
hf auth login
scripts/upload_model.sh efficientnet-b0
scripts/upload_model.sh vit-base-patch16-224
```

This creates a public model repo under your account and uploads the model and its preprocessor. Anyone can then load it with `AutoModelForImageClassification.from_pretrained("<your-username>/dermai-<model>")`. Set `HF_USER` at the top of the script to your own username before pushing.

## Team

- Xiaoyan Xing
- Somayeh Bahrami
- Sebastian Gonzalez
- Ka Wing Ariel Lee

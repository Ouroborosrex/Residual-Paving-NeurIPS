# Residual Paving

Code for **Residual Paving: Diagnosing the Routing Bottleneck in Selective Refusal Editing**, Bryce Hinkley and Peyman Najafirad, NeurIPS 2026. [Paper](https://arxiv.org/abs/2605.20262).

Residual Paving trains an early-layer router and later-layer bottleneck residual experts while keeping the language-model backbone frozen. The repository includes the complete staged training pipeline, gate and harmful-veto calibration, checkpoint evaluation, completion export, simple steering controls, configurations, tests, and reported aggregate tables.

The training and evaluation code are public under the [MIT license](LICENSE). This release does not include pretrained model weights, trained controller checkpoints, raw benchmark datasets, or harmful completions. Model and dataset licenses apply separately; see [third-party terms](docs/THIRD_PARTY.md).

## Install

Use Python 3.12 in a fresh environment, from this repository's root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The pinned environment was checked on CPU with PyTorch 2.10.0 and Transformers 5.5.4. Install the appropriate PyTorch wheel for your CUDA environment when training on a GPU. Optional 4/8-bit loading requires Accelerate and bitsandbytes; PCA/probe steering controls require scikit-learn. These dependencies are listed in `requirements-optional.txt` and were not part of the CPU release checks.

## Check the pipeline without downloads

```bash
python -m unittest discover -s tests -q
python scripts/smoke_pipeline.py
```

The second command creates a tiny random model and synthetic prompts, then executes target construction, gate pretraining, contrastive warmup, expert fitting, joint fitting, calibration, checkpoint writing, and evaluation on CPU. It does not download a pretrained model or use benchmark data. The checked results are in [results/repro](results/repro).

## Train

Obtain the benchmark files from their original providers and follow [data/raw/README.md](data/raw/README.md) for paths and schemas. For Gemma, accept the model provider's access terms and configure your own Hugging Face authentication.

```bash
# Inspect the command before loading a model.
python -m src.main --config configs/smoke_gemma3.json --dry-run

# Small run through the complete training schedule on the real model.
python -m src.main --config configs/smoke_gemma3.json

# Longer schedule and paper architecture/loss settings, with a small data budget.
python -m src.main --config configs/gemma3_training_recipe.json
```

Arguments after `--` override recipe settings, for example:

```bash
python -m src.main --config configs/gemma3_training_recipe.json -- --data-seed 43
```

The runner prints its output directory under `experiments/runs/`. It writes configuration and run metadata, training/evaluation summaries, and `moe_gate_policy_smoke/checkpoint.pt`. Despite its historical filename, `experiments/moe_gate_policy_smoke.py` runs the full staged trainer. See [the reproduction guide](docs/REPRODUCING.md) for the stages, evaluation commands, and protocol distinctions.

## Reproduction scope

The recipes make the pipeline executable; the small supplied data budgets are **not** a reproduction of the original paper's numerical results. Exact historical prompt lists, original controller checkpoints, and proprietary judge outputs are not packaged here. The data loader and split-construction code are included, and new runs record their own settings. The recipe's reported operating scale is not a fresh calibration result.

The five-model follow-up exports used different checkpoints and settings from the main cross-backbone experiment. Their recorded export settings are in [configs/five_model_export_protocols.json](configs/five_model_export_protocols.json). Those slices preserve paired harmful-refusal labels but do not reproduce the main edit gains. [Reported aggregate tables](results/aggregate) are historical evidence, not outputs of the software checks.

The release tag `v1.0-neurips2026` identifies this packaged implementation. It is not a claim that every historical experiment ran at this Git commit.

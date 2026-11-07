# Training and evaluation guide

Run commands from the repository root after installing `requirements.txt`. GPU training requires enough memory for the frozen backbone, cached references, and training traces; the paper reports the original hardware and budgets. The download-free CPU check is a software execution check only.

## What the trainer does

1. Load a frozen instruction-tuned model and sample edit, benign-keep, and harmful-keep prompts. Partition them using the configured data and evaluation seeds.
2. Construct edit anchors. With `edit_target_mode=safe_reframe`, the frozen model generates a target from a safety-framed version of the same prompt, optionally with `edit_anchor_prefix`. Cache boundary features, anchor tokens, and reference distributions.
3. Pretrain the scalar route with edit/keep labels, then run contrastive warmup using token-distribution divergence and route classification. This is not a residual-cosine contrastive objective.
4. Fit the residual experts with a forced edit gate (stage 1).
5. Jointly fit the router and experts with a soft gate, edit objectives, and keep-distribution preservation (stage 2).
6. Calibrate the gate with the editor frozen (stage 3), and evaluate learned and diagnostic routing policies. The optional harmful-veto head is fitted from cached boundary features during setup, separately from editor training.

The `traj_weight` option is a historical name for the teacher-distribution penalty used by the trainer. It is separate from the residual-cosine trajectory diagnostic in the paper. The keep-distribution score uses exponentiated, nonnegative-clipped top-k KL; it is not a refusal rate. Refusal heuristics in the local evaluator are not interchangeable with the paper's LLM judges.

## Recipes and data

| File | Purpose |
| --- | --- |
| `configs/smoke_gemma3.json` | Small run through every training stage on Gemma-3-4B-IT; requires real model/data access. |
| `configs/gemma3_training_recipe.json` | Paper architecture, schedule and loss settings with a small requested sample budget. |
| `configs/five_model_export_protocols.json` | Recorded export-time settings for the separate Appendix C.2 checkpoints, not independently reconstructed training recipes. |

Data paths and schemas are in [data/raw/README.md](../data/raw/README.md). `--n-edit`, `--n-keep-benign`, and `--n-keep-harmful` set requested pool sizes before the internal split; they are not per-partition counts. Available data, filtering, and deduplication can reduce the realized counts. Read the run's configuration and split summary instead of assuming the request was met. `--data-seed` controls sampling and `--eval-seed` controls partitioning.

The original main 500/500/98 evaluation, external 232-ID assay, and separate five-model slices have distinct protocols. This release includes the construction code and reported aggregate tables, not the original prompt-list exports or trained checkpoints. Reconstructing an exact historical result requires those inputs, the original model revision and environment, the same calibration split, and the original judging protocol. The supplied small recipes do not establish that reproduction.

The Gemma recipe sets scale 8 as the reported operating scale. It does not select a scale anew. To study a new dataset or checkpoint, choose operating settings on a separate calibration partition and keep the evaluation partition held out. The runner's `--calibrate-scale` options and the checkpoint evaluator expose that split explicitly; consult `--help` for available controls.

## Commands and outputs

```bash
python -m src.main --config configs/gemma3_training_recipe.json --dry-run
python experiments/moe_gate_policy_smoke.py --help
python experiments/moe_checkpoint_reeval.py --help
python experiments/moe_export_completions.py --help
```

The recipe launcher passes an argument list directly to the existing trainer. Later occurrences of a value option override the recipe. Boolean flags follow the underlying command's argparse behavior.

After training, use the printed checkpoint path:

```bash
python experiments/moe_checkpoint_reeval.py \
  --checkpoint experiments/runs/YOUR_RUN/moe_gate_policy_smoke/checkpoint.pt \
  --skip-scale-calibration \
  --eval-path learned:thresholded_soft:router:selected
```

This re-evaluator rebuilds data and reference caches from the saved configuration, so it needs the same data sources and model access. It can calibrate scale when `--skip-scale-calibration` is omitted. Explicit `oracle` routes use held-out labels and are diagnostics, not deployable policies. Completion export also requires an explicit scale: for example, `--eval-path learned:thresholded_soft:router:8` specifies scale 8; the export command does not infer a scale calibration from the paper.

The checkpoint contains the router and residual-controller state, configuration, layer indices, and gate threshold. Evaluation reconstructs the experiment and its reference data and refits the optional veto from the configured training examples; it is not a self-contained deployed model. Only load checkpoints from a trusted source: the historical format uses Python/PyTorch serialization.

Source files are organized as follows:

| Path | Role |
| --- | --- |
| `src/main.py` | JSON recipe launcher. |
| `src/trainer/moe_residual_paver.py` | Router, expert modules, residual hooks, and generation. |
| `experiments/dataset.py` | Prompt loading, partitioning, target construction, and caching. |
| `experiments/moe_harness.py` | Staged optimization, harmful veto, and policy evaluation. |
| `experiments/moe_gate_policy_smoke.py` | Training orchestration and run metadata. |
| `experiments/moe_checkpoint_reeval.py` | Checkpoint reconstruction and evaluation. |
| `experiments/moe_export_completions.py` | Paired completion export for external scoring. |
| `experiments/simple_steering_baselines.py` | Simple activation-steering controls. |
| `tests/` | CPU tests for the implementation. |

## Release verification

The release checks use synthetic fixtures and a tiny random Qwen2 model. They cover execution and serialization, not the reported edit gains or safety preservation. No full-size training, proprietary judging, optional quantization, or shared-H200 workload was run for these packaging checks. The pinned dependencies describe the checked release environment, not a retroactive claim about every historical experiment's environment.

`source_manifest.json` records hashes of the input source snapshot before portability and release edits; Git records the packaged implementation. The tag is the release identity, not the unavailable historical run commit.

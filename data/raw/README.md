# Data Inputs

This directory is intentionally empty in the code release. Place public-source prompt files here, or set the matching environment variables before running the scripts.

## Expected Files

| File | Environment variable | Expected schema |
| --- | --- | --- |
| `salad_bench.json` | `RP_SALAD_PATH` | JSON list of strings or objects with `instruction`, `prompt`, `question`, `text`, or `Behavior`. Optional category/source metadata is preserved. |
| `alpaca.json` | `RP_ALPACA_PATH` | JSON list of objects with an `instruction` field. |
| `harmbench_val.csv` | `RP_HARMBENCH_PATH` | CSV rows with HarmBench-style harmful prompts, typically in `Behavior` or `instruction`. |
| `harmbench_test.csv` | `RP_HARMBENCH_TEST_PATH` | CSV rows for test-only harmful-keep evaluation. |

## Environment Override Example

```bash
export RP_SALAD_PATH=/path/to/salad_bench.json
export RP_ALPACA_PATH=/path/to/alpaca.json
export RP_HARMBENCH_PATH=/path/to/harmbench_val.csv
export RP_HARMBENCH_TEST_PATH=/path/to/harmbench_test.csv
```

Keep the external test source separate from training. `RP_HARMBENCH_TEST_PATH` identifies a test file for scripts that explicitly load it; setting it does not automatically add an external evaluation to the training runner. The runner samples each source and constructs train/evaluation partitions using `data_seed`, `eval_seed`, and `eval_fraction`.

The original benchmark files are not redistributed. Obtain them from [SALAD-Bench](https://huggingface.co/datasets/OpenSafetyLab/Salad-Data), [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), and [HarmBench](https://github.com/centerforaisafety/HarmBench). For SALAD-Bench, map the chosen question field to `instruction` while retaining its source ID and category fields; the choice of subset is part of the experiment protocol.

See [third-party terms](../../docs/THIRD_PARTY.md). Generated split exports and completion records are local experiment artifacts and are excluded from this code release.

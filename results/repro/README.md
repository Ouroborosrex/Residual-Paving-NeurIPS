# Release checks

Checked locally on CPU, November 6, 2026:

- **96 unit tests passed** using `python -m unittest discover -s tests -q`.
- **All-stage synthetic pipeline passed** using `python scripts/smoke_pipeline.py`; see [tiny_pipeline.json](tiny_pipeline.json).

Environment: Python 3.12.7, PyTorch 2.10.0+cu128 (CPU execution), Transformers 5.5.4, datasets 4.8.4, NumPy 2.2.6. CUDA devices were disabled; CPU thread counts were set to two. The CPU test creates its own tiny random model and benign synthetic fixtures, with no benchmark or pretrained-weight downloads.

The unit suite covers prompt construction, refusal/preservation helpers, route and veto policies, reference caches, checkpoint reconstruction, training helpers, and the recipe launcher. Some tests deliberately simulate out-of-memory exceptions to exercise batch splitting; their diagnostic messages are expected.

These checks establish software execution for the tested paths. They do not reproduce the paper's full-size training, behavioral gains, LLM-judge labels, quantized execution, or GPU memory measurements. Historical aggregate results are kept separately in `results/aggregate/`.

# Model and dataset terms

The MIT license in this repository covers the original Residual Paving code. It does not relicense models, datasets, or dependencies. No pretrained weights, trained controller checkpoints, benchmark files, or harmful completion records are included.

## Models

The following identifiers and provider pages correspond to the models discussed in the paper. License names below reflect the linked model cards checked on November 6, 2026; follow the provider's terms when obtaining or using the models.

| Model identifier and provider page | Listed license / terms |
| --- | --- |
| [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | Gemma |
| [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) | Gemma |
| [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | Llama 3.2 Community License |
| [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Apache-2.0 |
| [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Apache-2.0 |
| [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | Apache-2.0 |
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Apache-2.0 |
| [NousResearch/Hermes-3-Llama-3.1-8B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B) | Card lists `llama3`; consult its license and underlying Llama terms. |

Gemma access requires agreement to Google's terms. See the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and incorporated [Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy), including its safety-circumvention restrictions. Releasing this code does not change those restrictions.

## Datasets

| Dataset / source | Provider's license information |
| --- | --- |
| [SALAD-Data](https://huggingface.co/datasets/OpenSafetyLab/Salad-Data) | Dataset card lists Apache-2.0. |
| [HarmBench](https://github.com/centerforaisafety/HarmBench) | Repository lists MIT. |
| [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) | Dataset is CC BY-NC 4.0; its usage notice limits it to noncommercial research. Code and data have separate terms. |

The reproduction guide describes how to supply local copies of these sources. Keep their attribution and applicable notices with any derived data you distribute. This code release does not include the external XSTest/JailbreakBench prompt slices or proprietary judge outputs used in follow-up experiments; the aggregate tables identify the evaluated protocols.

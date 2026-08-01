# DeepSeek

| Model | MMLU | MMLU-Pro | GPQA Diamond | SimpleQA | HLE | Source |
|---|---|---|---|---|---|---|
| DeepSeek-V3 (671B-A37B) | 88.5 | 75.9 | 59.1 | 24.9 | — | https://huggingface.co/deepseek-ai/DeepSeek-V3 |
| DeepSeek-V3.1 (non-thinking) | 91.8 (Redux) | 83.7 | 74.9 | — | — | https://huggingface.co/deepseek-ai/DeepSeek-V3.1 |
| DeepSeek-V3.1-Think | 93.7 (Redux) | 84.8 | 80.1 | 93.4* | 15.9 | https://huggingface.co/deepseek-ai/DeepSeek-V3.1 |
| DeepSeek-V3.2-Exp | — | 85.0 | 79.9 | 97.1* | 19.8 | https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp |
| DeepSeek-R1 | 90.8 | 84.0 | 71.5 | 30.1 | — | https://huggingface.co/deepseek-ai/DeepSeek-R1 |
| R1-Distill-Llama-70B | — | — | 65.2 | — | — | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B |
| R1-Distill-Qwen-32B | — | — | 62.1 | — | — | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B |

*SimpleQA values >90 likely with-search variant; flag for exclusion in primary regression.

# Qwen

| Model | MMLU | MMLU-Pro | GPQA Diamond | SimpleQA | HLE | Source |
|---|---|---|---|---|---|---|
| Qwen2.5-72B-Instruct | — | 71.1 | 49.0 | — | — | https://qwenlm.github.io/blog/qwen2.5-llm/ |
| Qwen2.5-7B-Instruct | — | 56.3 | 36.4 | — | — | https://qwenlm.github.io/blog/qwen2.5-llm/ |
| Qwen3-32B (base) | — | 65.54 | — | — | — | https://arxiv.org/pdf/2505.09388 |
| Qwen3-30B-A3B (base) | — | 61.49 | 43.94 | — | — | https://arxiv.org/pdf/2505.09388 |
| Qwen3-235B-A22B-Instruct-2507 | 93.1 (Redux) | 83.0 | 77.5 | 54.3 | — | https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507 |
| Qwen3-235B-A22B-Thinking-2507 | 93.8 (Redux) | 84.4 | 81.1 | — | 18.2 | https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507 |
| Qwen3-Next-80B-A3B-Instruct | 90.9 (Redux) | 80.6 | 72.9 | — | — | https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct |

# Kimi

| Model | MMLU | MMLU-Pro | GPQA Diamond | SimpleQA | HLE | Source |
|---|---|---|---|---|---|---|
| Kimi-K2-Instruct (1T-A32B) | 89.5 | 81.1 | 75.1 | 31.0 | 4.7 | https://huggingface.co/moonshotai/Kimi-K2-Instruct |

# GLM

| Model | MMLU | MMLU-Pro | GPQA Diamond | SimpleQA | HLE | Source |
|---|---|---|---|---|---|---|
| GLM-4-32B-0414 | — | — | — | 88.1* | — | https://huggingface.co/zai-org/GLM-4-32B-0414 |

*GLM-4-32B SimpleQA 88.1 is with-search; exclude.

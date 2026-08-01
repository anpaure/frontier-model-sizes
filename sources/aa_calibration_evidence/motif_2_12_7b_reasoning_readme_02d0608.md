---
license: apache-2.0
language:
- en
- ko
base_model:
- Motif-Technologies/Motif-2-12.7B-Base
tags:
- text-generation-inference
- conversational
- custom_code
- text-generation
- Motif
library_name: transformers
---

Last update: 10 Dec. 2025

# Introduction

This is a reasoning enhanced version of **Motif-2-12.7B-Instruct**. Detailed information will be released later.

# Evaluation

|Benchmark|Evaluation setting|Motif-2-12.7B|Motif-2-12.7B|
|---|---|---|---|
|||Instruct|Reasoning|
|MMLU|0-shot|86.11|84.07
|MMLU-Redux|-|90.02|88.89|
|BBH|0-shot|85.78|78.34|
|GPQA-Diamond|0-shot, CoT|63.6|70|
|GSM8K|0-shot, CoT|96.13|95.53|
|MATH|0-shot|97|95.07|
|MBPP|3-shot|91|88.9|
|LiveBench 2024-11-25|-|33.8|49.9|
|IFEval|strict prompt|75.78|79.11|
|IFEval|0-shot|76.52|81.89|
|MATH-500|-|96.8|99.3|
|AIME24|-|72.3|88.3|
|AIME25|-|63.6|80|
|ZebraLogic|-|69.5|77|
|BFCL v3|-|55.34|60.2|
|LiveCodeBench v5 <br> (2024.10 - 2025.2)|-|50.03|65|
|LiveCodeBench v5 |0-shot, CoT|61.66|60.1|
|HumanEval|0-shot|93.2|93.2|
|**Average**|-|**75.45**|**79.71**|

## How to use in vllm
The [PR](https://github.com/vllm-project/vllm/pull/27396) adding support for the Motif model in the official vLLM package is currently under review.  
In the meantime, to use our model with vLLM, please use the following container [image](https://github.com/motiftechnologies/vllm/pkgs/container/vllm).  
Our model supports a sequence length of up to 64K tokens.
```bash
# run vllm api server
VLLM_ATTENTION_BACKEND=DIFFERENTIAL_FLASH_ATTN \
vllm serve Motif-Technologies/Motif-2-12.7B-Reasoning \
    --trust-remote-code \
    --max-model-len 65536 \
    --tensor-parallel-size 8

# sending requests with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital city of South Korea?"}
    ],
    "temperature": 0.6
  }'
```

## How to use advanced vllm options
For maximum performance, we highly recommend using the options below.  
```--compilation_config '{"full_cuda_graph": true}'``` : Activates cuda [full graph capture](https://docs.vllm.ai/en/stable/design/cuda_graphs/#cudagraphmodes)  
```--rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":65536}'```: Apply [yarn](https://arxiv.org/abs/2309.00071) to support 128K context length  
```--enable-auto-tool-choice --tool-call-parser hermes``` : Enables [tool calling](https://docs.vllm.ai/en/latest/features/tool_calling/)  
```--logits-processors logit_:WrappedPerReqLogitsProcessor```: Enables a ratio-based thinking budget and repetition-based auto-stop. The model is guided to think for ```(model_max_len - input_prompt_len) * VLLM_THINK_BUDGET_RATIO``` tokens, using the rest of the context window to generate the response  
```--reasoning-parser deepseek_r1``` : Parses [reasoning outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/)  

```bash
pip install -U "huggingface_hub[cli]"
hf download Motif-Technologies/Motif-2-12.7B-Reasoning \
  --include "logit_processors/*" \
  --local-dir ./

export PYTHONPATH="$PWD/logit_processors"
VLLM_ATTENTION_BACKEND=DIFFERENTIAL_FLASH_ATTN \
VLLM_THINK_BUDGET_RATIO=0.95 \ 
vllm serve Motif-Technologies/Motif-2-12.7B-Reasoning \
    --trust-remote-code \
    --compilation_config '{"full_cuda_graph": true}' \
    --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":65536}' \
    --max-model-len 131072 \
    --tensor-parallel-size 8 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --logits-processors logit_:WrappedPerReqLogitsProcessor \
    --reasoning-parser deepseek_r1
```

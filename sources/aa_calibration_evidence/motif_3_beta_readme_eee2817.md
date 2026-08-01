---
library_name: transformers
pipeline_tag: text-generation
language:
- en
- ko
tags:
- motif
- motif-3
- mixture-of-experts
- moe
- long-context
- multilingual
- preview
---

# Motif-3-Beta

> ⚠️ **Preview / beta checkpoint — not the final release.**
> This repository hosts an intermediate checkpoint of **Motif-3**. The **final checkpoint will be released soon.**

**Motif-3** is a large-scale Mixture-of-Experts (MoE) language model built from the ground up by
[Motif Technologies](https://motiftech.io) following a fully in-house, proprietary design — not a
re-parameterization of existing open-source architectures.

## Highlights

- 🧠 **~314B total parameters / ~13B active** per token (sparse MoE)
- 📏 **256K context length** (262,144 tokens), natively long-context
- ⚡ Sparse routing: 384 experts with 8 activated per token, plus 1 shared expert
- 🌐 **Multilingual**, general-purpose

## Model details

|                    |                                          |
| ------------------ | ---------------------------------------- |
| Model type         | Mixture-of-Experts causal language model |
| Total parameters   | ~314B                                    |
| Active parameters  | ~13B / token                             |
| Hidden size        | 4096                                     |
| Layers             | 53                                       |
| Routed experts     | 384 (top-8)                              |
| Shared experts     | 1                                        |
| Context length     | 262,144 (256K)                           |
| Vocabulary         | 220,160                                  |
| Tensor type        | bfloat16                                 |

## Architecture

Motif-3 is a fully in-house design and introduces several custom components:

- **Grouped Differential Latent Attention (GDLA)**
- **Grouped PolyNorm activation**, applied per expert
- **Modified mHC**
- **Multi-Token Prediction (MTP)** head (1 layer), enabling self-speculative decoding

## Benchmarks

**Artificial Analysis Intelligence Index (AAII): 44**

See [Artificial Analysis](https://artificialanalysis.ai/) for details.

## Usage - vLLM (Beta)

- Docker image: `ghcr.io/motiftechnologies/vllm:v0.20.2-motif3.rc2`
- Tested only on B200 and H200 GPUs.
- If you encounter any issues, please open an HF issue.
- The model ships with a built-in **MTP (multi-token prediction)** head (`num_nextn_predict_layers=1`), so it supports **self-speculative decoding** — add `--speculative-config` as shown below (`num_speculative_tokens: 1` is optimal for this model).
- Example vLLM command:
```
vllm serve "Motif-Technologies/Motif-3-Beta" \
    --served-model-name motif \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --data-parallel-size-local 8 \
    --enable-expert-parallel \
    --dtype bfloat16 \
    --quantization modelopt_blockfp8 \
    --speculative-config '{"model": "Motif-Technologies/Motif-3-Beta", "num_speculative_tokens": 1}' \
    --max-model-len 262144 \
    --generation-config auto \
    --reasoning-parser motif \
    --enable-auto-tool-choice \
    --tool-call-parser motif \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 --port 8080
```


## Usage - HF.generate

The shipped modeling code has been fixed and HF `.generate` now runs and produces coherent output. For production or high-throughput serving, vLLM (above) is recommended.

The model ships with custom modeling code, so load it with `trust_remote_code=True`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Motif-Technologies/Motif-3-Beta"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

messages = [{"role": "user", "content": "Hello!"}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## Access

This model is **openly available** — anyone can download the weights, no access request required.

## License

Permission is granted to use, modify, and redistribute this software
for personal, educational, and non-commercial research purposes only.

Commercial use is prohibited without prior written permission from
Motif Technologies.

---

© Motif Technologies. All rights reserved.

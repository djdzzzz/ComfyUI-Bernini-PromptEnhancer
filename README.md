# ComfyUI-Bernini-PromptEnhancer

[中文文档](README.zh-CN.md)

A ComfyUI custom node package that brings **ByteDance Bernini**'s MLLM-based semantic planning to local prompt enhancement. It combines image/video understanding with natural-language planning via **local GGUF inference** — no cloud API required — and outputs enhanced, high-quality prompts ready for video/image diffusion models (Wan2.2, etc.).

Based on [Bernini: Latent Semantic Planning for Video Diffusion](https://arxiv.org/abs/2605.22344) and the [Bernini-Diffusers](https://huggingface.co/ByteDance/Bernini-Diffusers) model.

## Features

- **Local GGUF inference** — runs entirely offline on consumer GPUs via llama.cpp
- **12 task types** — `t2v`, `t2i`, `v2v`, `mv2v`, `i2i`, `i2v`, `ads2v`, `vi2v`, `r2v`, `r2i`, `rv2v`, `vrc2v`
- **Zero VRAM residue** — subprocess-isolated inference, VRAM fully released after each run
- **Dual output** — `enhanced_prompt` + `structured_plan` (the reasoning trace)
- **Structured & official modes** — `template_mode` toggle on the Bernini node
- **Thinking-mode toggle** — Qwen3.5 node with `thinking_mode` switch (off by default)
- **Reference-slot alignment** — Ref-X labels map 1:1 to ComfyUI's `reference_image_X` inputs

### Task template overview

| Tasks | Template type | Output format |
|-------|--------------|---------------|
| r2v, r2i | built-in RULES five-section | `[OBSERVATION]` → `[FINAL_PROMPT]` |
| rv2v, vrc2v | built-in RULES five-section | same as above |
| i2v | separated system prompt | same as above |
| v2v, mv2v, i2i | short template + PLAN_SUFFIX | same as above |
| vi2v, ads2v | short template, no PLAN_SUFFIX | single-sentence output |
| t2v, t2i | plain-text template | direct enhancement |

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/djdzzzz/ComfyUI-Bernini-PromptEnhancer.git
cd ComfyUI-Bernini-PromptEnhancer
pip install -r requirements.txt
```

Or install via [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager) by searching "Bernini PromptEnhancer".

## Model setup

Download the Bernini-MLLM-Qwen2.5-VL-7B GGUF quantized models into `ComfyUI/models/clip/`:

- [Bernini-MLLM-Qwen2.5-VL-7B GGUF](https://huggingface.co/mradermacher/Bernini-MLLM-Qwen2.5-VL-7B-GGUF) — main model (`*.Q4_K_M.gguf` recommended)
- The matching `*.mmproj-*.gguf` from the same repo (required for image/video inputs)

### Qwen3.5 node (optional)

Supports the Qwen3.5-4B series. A prompt-engineering fine-tune is recommended:

- [Yusiko/qwen3.5-prompter](https://huggingface.co/Yusiko/qwen3.5-prompter) — Qwen3.5-4B, prompt-engineering tuned, multimodal

Place both `.gguf` and `.mmproj.gguf` into `ComfyUI/models/clip/`.

## Nodes

### Bernini MLLM Prompt Enhancer (GGUF)

Main node, under the `Bernini` category.

**Required inputs:**
- `model` — main GGUF model (e.g. `Q4_K_M.gguf`)
- `mmproj` — multimodal projection (e.g. `mmproj-Q8_0.gguf`)
- `task_type` — one of the 12 task types
- `template_mode` — `structured` (planned) or `official` (single-pass)
- `prompt` — your instruction

**Optional inputs:**
- `source_video`, `reference_video`, `reference_image_0/1/2`

**Outputs:**
- `enhanced_prompt` — the final enhanced prompt
- `structured_plan` — the semantic planning trace (structured mode)

### Qwen3.5 Prompt Enhancer (GGUF)

Next-generation node, under the `Bernini` category. Covers all 12 task types and suppresses thinking-mode output by default.

**Required inputs:** `model`, `mmproj`, `task_type`, `prompt`
**Optional inputs:** `thinking_mode`, `source_video`, `reference_video`, `reference_image_0/1/2`
**Outputs:** `enhanced_prompt`, `structured_plan`

## Example

**Input:** `Replace the person, change the background to a library, keep the original motion`

**structured_plan:**
```
[OBSERVATION]
The source video shows a young woman sitting on the floor of a bedroom, dressed in
an elegant blue gown and holding a glass slipper...

The reference image (Ref-0) depicts another young woman in a cozy room with
floral-patterned furniture, wearing a yellow dress with purple floral designs...

[UNDERSTAND]
The instruction requires replacing the original character with the one from the
reference image while keeping the glass-slipper pose. The setting changes to a library.

[EXECUTE]
1. Replace the woman in blue with the girl from Ref-0, keeping similar pose/expression.
2. Change the background to a library with bookshelves, a large window with natural
   light, and plants to preserve the original aesthetic.
```

**enhanced_prompt:**
```
In this filmic narrative, we find ourselves in a serene library. Soft warm light from
a large window bathes the scene, highlighting the intricate purple floral patterns on
the young woman's yellow dress. She sits gracefully on the floor, cradling a glass
slipper. Towering bookshelves and a classic wooden table adorned with books and plants
complete this quiet corner of knowledge.
```

## Models

| Quant | Size | Notes |
|-------|------|-------|
| Q4_K_M | ~4GB | Recommended, ~5GB VRAM |
| Q8_0 | ~8GB | Higher quality, ~9GB VRAM |
| bf16 | ~14GB | Full precision, best quality |

## Acknowledgements

- [Bernini-Diffusers](https://huggingface.co/ByteDance/Bernini-Diffusers) — ByteDance's semantic planner
- [Bernini MLLM GGUF](https://huggingface.co/mradermacher/Bernini-MLLM-Qwen2.5-VL-7B-GGUF) — GGUF quantization
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

## License

Apache-2.0 (see [LICENSE](LICENSE))

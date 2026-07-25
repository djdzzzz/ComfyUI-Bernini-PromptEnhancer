# ComfyUI-Bernini-PromptEnhancer

[中文文档](README.zh-CN.md)

A ComfyUI custom node package that pairs ByteDance Bernini's VLLM and semantic planning with image/video understanding and natural language reasoning for local prompt enhancement, powered by local GGUF inference — no cloud API required. Outputs high-quality enhanced prompts.

Based on [Bernini: Latent Semantic Planning for Video Diffusion](https://arxiv.org/abs/2605.22344) and the [Bernini-Diffusers](https://huggingface.co/ByteDance/Bernini-Diffusers) model.

## Features

- **13+n task types** — 13 core tasks + specialized lightweight tasks
- **Smart frame selection** — `smart_frames` toggle uses frame-difference sampling instead of uniform spacing, giving the MLLM higher information density
- **First/Last frame transition** — `fl2v` task generates a prompt describing the video between a start frame and end frame
- **GGUF local inference** — runs entirely offline on consumer GPUs via llama.cpp subprocess
- **Zero VRAM residue** — subprocess-isolated, VRAM fully released after each run
- **Dual output** — `enhanced_prompt` + `structured_plan`
- **Image passthrough** — `source_video`, `reference_video`, `reference_image_0/1/2` pass through unchanged
- **Dual template mode** — `structured` (five-section RULES) or `official` (single-pass), per the Bernini node
- **Thinking-mode toggle** — Qwen3.5 node with `thinking_mode` switch (off by default)

### Tasks

| Task | Inputs used | Description |
|------|-------------|-------------|
| t2v | prompt | Text to video |
| t2i | prompt | Text to image |
| v2v | source_video + prompt | Enhance/edit source video |
| mv2v | source_video + prompt | Multi-video edit (alias) |
| i2i | prompt | Image to image |
| i2v | ref_images + prompt | Image(s) to video |
| ads2v | source_video + prompt | Ad placement in video |
| vi2v | source_video + ref_images + prompt | Video + ref images edit |
| r2v | ref_images + prompt | Reference images to video |
| r2i | ref_images + prompt | Reference images to image |
| rv2v | ref_video + ref_images + prompt | Ref video + ref images edit |
| vrc2v | source_video + ref_images + prompt | Video + ref images edit (alias) |
| fl2v | ref_image_0 + ref_image_1 + prompt | First-to-last frame transition |

**Specialized tasks** — for specific mixed-source scenarios, with explicit role assignment per input:

| Task | Inputs used | Role assignment | Description |
|------|-------------|-----------------|-------------|
| r2v_motion | source_video + ref_images + prompt | ref=character, source=motion | Ref character performs source motion in described setting |

All tasks accept an optional user `prompt`. When left empty, a default instruction is used.

### Smart frame sampling

Default frame selection is uniform (evenly spaced). Enable `smart_frames` to use thumbnail-based frame-diff sampling: the first frame is always kept, remaining slots are filled by the highest inter-frame change magnitude. Thumbnail downscale (64x64) naturally resists noise and compression artifacts.

```
Uniform: 30 frames, pick 5 → frames 0, 7, 15, 22, 29
Smart:   30 frames, pick 5 → frame 0 (fixed) + the 4 frames with highest change
```

Controlled by `video_frames` (1–16).

### fl2v — First/Last frame transition

Generates a prompt describing the continuous video between a start frame (`reference_image_0`) and an end frame (`reference_image_1`). Reference images are labeled `START-FRAME` and `END-FRAME` explicitly in the model input.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/djdzzzz/ComfyUI-Bernini-PromptEnhancer.git
cd ComfyUI-Bernini-PromptEnhancer
pip install -r requirements.txt
```

## Model setup

Download GGUF quantized models into `ComfyUI/models/clip/`:

- [Bernini-MLLM-Qwen2.5-VL-7B GGUF](https://huggingface.co/mradermacher/Bernini-MLLM-Qwen2.5-VL-7B-GGUF) — `*.Q4_K_M.gguf` recommended
- Matching `*.mmproj-*.gguf` from the same repo

### Qwen3.5 node (optional)

- [Yusiko/qwen3.5-prompter](https://huggingface.co/Yusiko/qwen3.5-prompter) — Qwen3.5-4B, prompt-engineering tuned, multimodal

## Nodes

### Bernini MLLM Prompt Enhancer (GGUF)

**Category:** `Bernini`

| Input | Type | Default | Notes |
|-------|------|---------|-------|
| model | dropdown | — | `.gguf` file |
| mmproj | dropdown | — | `.mmproj.gguf` file |
| task_type | dropdown | v2v | 13 tasks |
| template_mode | dropdown | structured | `structured` or `official` |
| prompt | STRING | "" | multiline |
| temperature | FLOAT | 0.6 | 0.0–2.0 |
| video_frames | INT | 3 | 1–16, frames per video |
| smart_frames | BOOLEAN | false | frame-diff sampling |
| image_max_side | INT | 512 | 0=original, else max side |
| *source_video* | IMAGE | — | optional |
| *reference_video* | IMAGE | — | optional |
| *reference_image_0/1/2* | IMAGE | — | optional |

| Output | Type | Notes |
|--------|------|-------|
| enhanced_prompt | STRING | final prompt |
| structured_plan | STRING | RULES reasoning trace |
| source_video | IMAGE | passthrough |
| reference_video | IMAGE | passthrough |
| reference_image_0/1/2 | IMAGE | passthrough |

### Qwen3.5 Prompt Enhancer (GGUF)

**Category:** `Bernini`

Same inputs and outputs as the Bernini node, plus:

| Input | Type | Default | Notes |
|-------|------|---------|-------|
| thinking_mode | BOOLEAN | false | Qwen3.5 think mode |

## Models

| Quant | Size | VRAM |
|-------|------|------|
| Q4_K_M | ~4GB | ~5GB |
| Q8_0 | ~8GB | ~9GB |
| bf16 | ~14GB | ~14GB |

## Acknowledgements

- [Bernini-Diffusers](https://huggingface.co/ByteDance/Bernini-Diffusers) — semantic planner
- [Bernini MLLM GGUF](https://huggingface.co/mradermacher/Bernini-MLLM-Qwen2.5-VL-7B-GGUF) — GGUF quantization
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

## License

Apache-2.0 (see [LICENSE](LICENSE))

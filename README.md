# ComfyUI-Bernini-PromptEnhancer

[中文文档](README.zh-CN.md)

ComfyUI custom nodes for local prompt enhancement via GGUF multimodal models. Includes Bernini VLLM semantic planning, Qwen3.5 text enhancement, and MiniMax H3 video prompt planning (powered by Gemma 4 E4B) with multi-material context analysis. Zero cloud API.

Based on [Bernini: Latent Semantic Planning for Video Diffusion](https://arxiv.org/abs/2605.22344) (Bernini node) and the [MiniMax H3 prompt specification](https://www.minimax.io/blog/minimax-h3) (H3 node).

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
- **H3 video prompt planning** — multi-material context (video + image + audio), H3-standard three-section output, direct passthrough to official MiniMax H3 node

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

**H3 node tasks:** `t2v` / `i2v` / `h3_multi_ref` — see [H3 Prompt Planner](#h3-prompt-planner-gguf) section below for details.

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

### H3 node

- [Gemma 4 E4B GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF) (4.2 GB); `mmproj-F32.gguf` in same repo — recommended for H3 video prompt planning

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

### H3 Prompt Planner (GGUF)

![H3 Prompt Planner](assets/h3_node.png)

Video prompt planner for MiniMax H3. Accepts video, image, and audio as context, produces H3-standard three-section English prompts with shot-by-shot structure and integrated soundscape. Powered by Gemma 4 E4B GGUF.

**Tasks:** `t2v` / `i2v` / `h3_multi_ref`

**Materials:** Drag & drop videos, images, audio. Reference as `ref_video_1`, `ref_image_1`, `ref_audio_1` in prompt.

**FINAL_PROMPT format (H3 three-section standard):**
```
integrated_multimodal_description: Live-action cinematic...
[Shot 1] ...camera pushes in with small amplitude at slow speed...
At 00:03.500, the camera cuts to... (S1) says: <d>[English]...</d>...

overall_soundscape: Steady rain on cobblestone, distant traffic hum...

non_diegetic_music: N/A
```

**Passthrough:** `media_0`–`media_14` (JS renames to `ref_video_0`, `ref_image_0`, `ref_audio_0` in ComfyUI). Wire directly into official `MiniMax H3 Reference to Video` node inputs.

| Input | Type | Default | Notes |
|-------|------|---------|-------|
| model | dropdown | — | GGUF file, auto-scanned |
| mmproj | dropdown | `<none>` | Vision projector (required for image/video) |
| task_type | dropdown | `h3_multi_ref` | t2v / i2v / h3_multi_ref |
| prompt | STRING | "" | Multiline; use `ref_video_1` etc. |
| temperature | FLOAT | 0.6 | 0.0–2.0 |
| repeat_penalty | FLOAT | 1.15 | 1.0–2.0 |
| seed | INT | 0 | 0 = random |
| n_ctx | INT | 8192 | Context window |
| n_gpu_layers | INT | -1 | -1 = all GPU |
| max_tokens | INT | 2048 | Max output tokens |
| frame_mode | dropdown | `uniform` | uniform / smart |
| video_frames | INT | 5 | 1–64, frames per video |
| sample_fps | INT | 0 | 0 = use video_frames |
| sample_seconds | FLOAT | 5.0 | Max duration for sampling |
| image_max_side | INT | 512 | 64–4096 |
| thinking_mode | BOOLEAN | false | Qwen3.5 think mode |

| Output | Type | Notes |
|--------|------|-------|
| enhanced_prompt | STRING | H3-standard three-section prompt (FINAL_PROMPT only) |
| structured_plan | STRING | OBSERVATION–PRESERVE planning trace |
| media_0–14 | IMAGE/AUDIO | Material passthrough (JS renames to ref_video_N etc.) |

## Models

| Quant | Size | VRAM |
|-------|------|------|
| Q4_K_M | ~4GB | ~5GB |
| Q8_0 | ~8GB | ~9GB |
| bf16 | ~14GB | ~14GB |

## Acknowledgements

- [Bernini-Diffusers](https://huggingface.co/ByteDance/Bernini-Diffusers) — semantic planner
- [Bernini MLLM GGUF](https://huggingface.co/mradermacher/Bernini-MLLM-Qwen2.5-VL-7B-GGUF) — GGUF quantization
- [MiniMax H3](https://www.minimax.io/blog/minimax-h3) — omni-modal video generation model ([open weights](https://huggingface.co/MiniMaxAI/MiniMax-H3))
- [Gemma 4 E4B GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF) — H3 planner backbone
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

## License

Apache-2.0 (see [LICENSE](LICENSE))

# ComfyUI-Bernini-PromptEnhancer

GGUF 本地推理的 Bernini MLLM 语义规划器，将图像/视频理解与自然语言规划结合，输出增强后的高质量 prompt。

## 核心特性

- **GGUF 本地推理** — 无需联网，本地 GGUF 模型驱动
- **12 种任务类型** — t2v、t2i、v2v、mv2v、i2i、i2v、ads2v、vi2v、r2v、r2i、rv2v、vrc2v
- **VRAM 零残留** — 子进程隔离，推理后显存完全释放
- **双输出** — `enhanced_prompt`（增强 prompt）+ `structured_plan`（语义规划）
- **结构化与官方双模式** — `template_mode` 切换
- **语义规划器模式** — 模型先观察、理解、规划，再输出最终 prompt

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/djdzzzz/ComfyUI-Bernini-PromptEnhancer.git
cd ComfyUI-Bernini-PromptEnhancer
pip install -r requirements.txt
```

## 模型准备

下载 Bernini-MLLM-Qwen2.5-VL-7B GGUF 量化模型到 `ComfyUI/models/clip/`：

- [Bernini-MLLM-Qwen2.5-VL-7B.gguf](https://huggingface.co/mradermacher/Bernini-MLLM-Qwen2.5-VL-7B-GGUF)
- [Bernini-MLLM-Qwen2.5-VL-7B.mmproj.gguf](https://huggingface.co/mradermacher/Bernini-MLLM-Qwen2.5-VL-7B-GGUF)

## 节点

### Bernini MLLM Prompt Enhancer (GGUF)

主节点，位于 `Bernini` 分类下。

**必选输入：**
- `model` — 主模型 GGUF（如 Q4_K_M.gguf）
- `mmproj` — 多模态投影层（如 mmproj-Q8_0.gguf）
- `task_type` — 12 种任务之一
- `template_mode` — `structured`（结构化规划）或 `official`（单段输出）
- `prompt` — 用户指令

**可选输入：**
- `source_video`、`reference_video`、`reference_image_0/1/2`

**输出：**
- `enhanced_prompt` — 最终增强 prompt
- `structured_plan` — 语义规划过程（structured 模式下）

## 模型

| 模型 | 大小 | 说明 |
|------|------|------|
| Q4_K_M | ~4GB | 推荐，显存需求 ~5GB |
| Q8_0 | ~8GB | 更高质量，显存需求 ~9GB |
| bf16 | ~14GB | 原始精度，最佳效果 |


## 致谢
- [Bernini-Diffusers](https://huggingface.co/ByteDance/Bernini-Diffusers)
- [Bernini MLLM-GGUF](https://huggingface.co/mradermacher/Bernini-MLLM-Qwen2.5-VL-7B-GGUF) — ByteDance 的语义规划器的GGUF量化
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

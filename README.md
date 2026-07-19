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

> Q4 量化模型视觉理解能力有限，天气/细粒度属性提取可能不准确。追求质量建议用更高精度。

## 常见问题

### 云 GPU（仙宫云等）：libcudart.so 不匹配

若报 `libcudart.so.12: cannot open shared object file`，说明 llama-cpp-python 为 CUDA 12 编译但服务器是 CUDA 13.x。重装即可：

```bash
conda activate comfyui  # 或其他 ComfyUI 所在的虚拟环境
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir --force-reinstall
```

它会自动匹配当前 CUDA 版本编译。

## 致谢

- [Bernini MLLM](https://hf-mirror.com/attashe/Bernini-MLLM-Qwen2.5-VL-7B) — ByteDance 的语义规划器
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

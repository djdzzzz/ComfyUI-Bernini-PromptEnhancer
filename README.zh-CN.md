# ComfyUI-Bernini-PromptEnhancer

GGUF 本地推理的 Bernini MLLM 语义规划器，将图像/视频理解与自然语言规划结合，输出增强后的高质量 prompt。

## 核心特性

- **GGUF 本地推理** — 无需联网，本地 GGUF 模型驱动
- **12 种任务类型** — t2v、t2i、v2v、mv2v、i2i、i2v、ads2v、vi2v、r2v、r2i、rv2v、vrc2v
- **VRAM 零残留** — 子进程隔离，推理后显存完全释放
- **双输出** — `enhanced_prompt`（增强 prompt）+ `structured_plan`（语义规划）
- **结构化与官方双模式** — Bernini 节点支持 `template_mode` 切换
- **思考模式开关** — Qwen3.5 节点支持 `thinking_mode` 开关，默认关闭
- **参考图槽位对齐** — Ref-X 标签严格对应 ComfyUI 的 `reference_image_X` 端口，避免编号混

### 任务模板概述

| 任务 | 模板类型 | 输出格式 |
|------|---------|---------|
| r2v, r2i | 自带 RULES 五段 | `[OBSERVATION]` → `[FINAL_PROMPT]` |
| rv2v, vrc2v | 自带 RULES 五段 | 同上 |
| i2v | system prompt 分离 | 同上 |
| v2v, mv2v, i2i | 短模板 + PLAN_SUFFIX | 同上 |
| vi2v, ads2v | 短模板，无 PLAN_SUFFIX | 单句输出 |
| t2v, t2i | 纯文本模板 | 直接增强输出 |

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

### Qwen3.5 节点（可选）

支持 Qwen3.5-4B 系列模型，推荐使用专门为 prompt 工程微调的模型：

- [Yusiko/qwen3.5-prompter](https://huggingface.co/Yusiko/qwen3.5-prompter) — Qwen3.5-4B，prompt 工程微调，支持多模态

下载 `.gguf` 和 `.mmproj.gguf` 到 `ComfyUI/models/clip/` 即可。

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

### Qwen3.5 Prompt Enhancer (GGUF)

新一代节点，位于 `Bernini` 分类下。12 种任务全覆盖，自动抑制思考模式输出。

**必选输入：**
- `model` — 主模型 GGUF（如 Q4_K_M.gguf）
- `mmproj` — 多模态投影层
- `task_type` — 12 种任务之一
- `prompt` — 用户指令

**可选输入：**
- `thinking_mode` — 开关 Qwen3.5 思考模式（默认关闭）
- `source_video`、`reference_video`、`reference_image_0/1/2`

**输出：**
- `enhanced_prompt` — 最终增强 prompt
- `structured_plan` — 语义规划过程（结构化模式下）

## 示例

**输入：** `人物替换，背景改在图书馆，原视频动作不变`

**structured_plan：**
```
[OBSERVATION]
The source video shows a young woman sitting on the floor of what appears to be a bedroom.
She is dressed in an elegant blue gown and holding a glass slipper. The room has large
windows with natural light, a wooden bed with white bedding, and a small table beside it.

The reference image (Ref-0) depicts another young woman sitting in what looks like a cozy
room with floral-patterned furniture, wearing a yellow dress with purple floral designs.
The lighting is warm and soft, creating a welcoming atmosphere.

[UNDERSTAND]
The instruction requires replacing the original character with one from the reference image
while keeping her action of holding the glass slipper. The setting is to be changed to a
library instead of a bedroom.

[EXECUTE]
1. Replace the young woman in the blue dress with the girl from Ref-0, ensuring she maintains
   similar poses and expressions.
2. Change the background to a library with bookshelves filled with books, a large window
   allowing natural light, and possibly some green plants or floral arrangements to maintain
   some of the original's aesthetic elements.
```

**enhanced_prompt：**
```
In this filmic narrative, we find ourselves in a serene library setting. The soft, warm glow
from a large window bathes the scene in golden light, highlighting the intricate purple floral
patterns on the young woman's yellow dress. Her hair is styled in an elegant updo, complementing
her poised demeanor as she sits gracefully on the floor, cradling a glass slipper in her hands.
The background is filled with towering bookshelves, a classic wooden table adorned with books
and plants, adding to the enchanting ambiance of this quiet corner of knowledge.
```

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

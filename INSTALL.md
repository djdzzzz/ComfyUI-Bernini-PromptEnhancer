# 安装教程

## 环境要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| Python | 3.10 | 3.11+ |
| CUDA | 12.4 | 12.8 |
| 显存 | 6 GB | 8 GB+ |
| 磁盘 | 10 GB | SSD 20 GB+ |

## 第一步：克隆安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/djdzzzz/ComfyUI-Bernini-PromptEnhancer.git
cd ComfyUI-Bernini-PromptEnhancer
pip install -r requirements.txt
```

### 依赖列表

`requirements.txt` 包含：

```
llama-cpp-python>=0.3.0   # GGUF 推理引擎
soundfile                  # 音频文件读写
scipy                      # 音频重采样
```

ComfyUI 自带的（不需额外安装）：`torch`、`numpy`、`Pillow`。

可选依赖：
- `av`（PyAV）——视频抽帧需要，ComfyUI 已内置
- `pynvml`——显存监控，ComfyUI 已内置

## 第二步：下载模型

### H3 Prompt Planner（推荐：Gemma 4 E4B）

下载两个 GGUF 文件放到 `ComfyUI/models/clip/` 下的任意子目录：

| 文件 | 大小 | 说明 |
|------|------|------|
| `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` | 4.2 GB | LLM 主干（QAT 量化版） |
| `mmproj-F32.gguf` | 1.9 GB | 多模态投影器（视觉+音频） |

**下载方式 A：ModelScope（国内快）**
```bash
pip install modelscope

# LLM
modelscope download --model unsloth/gemma-4-E4B-it-qat-GGUF \
  gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  --local_dir ComfyUI/models/clip/gemma-4-E4B-it-qat-GGUF

# 视觉投影器（mmproj，在同一个 unsloth 仓库里）
modelscope download --model unsloth/gemma-4-E4B-it-qat-GGUF \
  mmproj-F32.gguf \
  --local_dir ComfyUI/models/clip/gemma-4-E4B-it-qat-GGUF
```

**下载方式 B：HuggingFace**
```bash
pip install huggingface_hub

huggingface-cli download unsloth/gemma-4-E4B-it-qat-GGUF \
  gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  --local-dir ComfyUI/models/clip/gemma-4-E4B-it-qat-GGUF

# 视觉投影器（同仓库，一起下载）
huggingface-cli download unsloth/gemma-4-E4B-it-qat-GGUF \
  mmproj-F32.gguf \
  --local-dir ComfyUI/models/clip/gemma-4-E4B-it-qat-GGUF
```

> **显存**: 模型 4.2G + mmproj ~2G + KV cache ~2G ≈ 6GB（RTX 3060 6GB 可跑）

### Bernini MLLM / Qwen3.5

任意兼容的 GGUF VLM 模型放到 `ComfyUI/models/clip/` 即可，节点启动时自动扫描。

## 第三步：验证

1. 重启 ComfyUI
2. 搜索节点 `H3 Prompt Planner`
3. 节点下拉中应能看到刚下载的 `gemma-4-E4B-it-qat-GGUF/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf`
4. 选择 mmproj 为 `gemma-4-E4B-it-qat-GGUF/mmproj-F32.gguf`
5. 输入简短指令（如"一只猫"），点 Queue
6. 如果正常输出五段规划 `[OBSERVATION] [UNDERSTAND] [EXECUTE] [PRESERVE] [FINAL_PROMPT]`，安装成功

## 常见问题

### Q: 模型下拉里没有我刚下载的 GGUF？

重启 ComfyUI。模型列表在启动时扫描 `ComfyUI/models/clip/` 及其子目录。

### Q: 报错 "No module named 'soundfile'"

```bash
pip install soundfile scipy
```

### Q: 显存不足（OOM）

- 降低 `n_gpu_layers`（如从 -1 改为 20）
- 降低 `n_ctx`（8192 → 4096）
- 降低 `image_max_side`（768 → 512）
- 减少 `video_frames`（5 → 3）

### Q: 模型加载太慢

首次从磁盘加载 4.2GB 约 20 秒，属正常。之后每次执行都会重新加载（用完即焚机制）。放到 SSD 上可显著加速。

### Q: 视频预览黑屏

浏览器不支持 H.265/HEVC 编码。不影响节点使用——PyAV 照样能解码抽帧。

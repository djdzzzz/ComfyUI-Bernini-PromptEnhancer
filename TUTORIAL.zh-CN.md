# 使用教程

## 目录

1. [文生视频 (t2v)](#文生视频-t2v)
2. [图生视频 (i2v)](#图生视频-i2v)
3. [全能参考 (h3_multi_ref)](#全能参考-h3_multi_ref)
4. [面板操作](#面板操作)
5. [参数详解](#参数详解)
6. [常见工作流](#常见工作流)
7. [性能调优](#性能调优)

---

## 文生视频 (t2v)

纯文本生成视频场景描述。

**步骤**：

1. 添加 **H3 Prompt Planner** 到工作流
2. `task_type` 选 `t2v`
3. 选择模型（Gemma 4 E4B + mmproj）
4. `prompt` 输入构思，如：
   ```
   一只橘猫在午后阳光下蜷缩在窗台上打盹，窗外是秋天的银杏树
   ```
5. Queue

**输出**（分两个 pin）：

`enhanced_prompt`（仅 FINAL_PROMPT 段）：
```
integrated_multimodal_description: Cinematic live-action, warm golden color palette...
[Shot 1] A medium shot of an orange tabby cat curled on the wooden windowsill...
At 00:04.000, the camera slowly pushes in...

overall_soundscape: Gentle breeze rustles through ginkgo leaves outside the window...

non_diegetic_music: N/A
```

`structured_plan`（规划过程）：
```
[OBSERVATION] 想象一只橘猫蜷缩在洒满金色阳光的窗台上，窗外银杏叶随风飘落...
[UNDERSTAND] 主体是橘猫，氛围是宁静慵懒的秋日午后...
[EXECUTE] 16:9画幅，15秒。一只橘色虎斑猫蜷缩在木窗台上...
[PRESERVE] 橘猫打盹、午后阳光、银杏树、宁静氛围
```

约 20-30 秒完成。

---

## 图生视频 (i2v)

图片作为视频的起始帧/首尾帧参考。

**步骤**：

1. `task_type` 选 `i2v`
2. 拖入 1-2 张图片到节点面板
3. `prompt` 中用 `<Picture 1>`、`<Picture 2>` 引用：
   ```
   <Picture 1> 作为起始帧，生成一段视频：女孩缓缓睁开眼睛，站起身走向窗边
   ```
4. Queue

模型会观察 `<Picture 1>` 的人物外观、场景色调、光线氛围，生成从该图出发的动作描述。

---

## 全能参考 (h3_multi_ref)

最强大的模式——自由组合视频、图片、音频作为多维参考。

**步骤**：

1. `task_type` 选 `h3_multi_ref`
2. 上传素材：
   - **源视频**：拖入一段视频（面板第一项）
   - **参考图**：拖入 1-9 张图片
   - **参考视频**：拖入 0-2 段视频
   - **音频**：拖入 0-3 段音频

3. 在 `prompt` 中说明各素材的用途：
   ```
   <Picture 1> 作为人物外观参考（古风服饰），
   <Video 1> 的动作和运镜节奏参考，
   <Audio 1> 的台词和情感氛围参考。
   生成一段仙侠风格的视频提示词。
   ```

4. Queue

**素材编号规则**：
- 源视频 → `<Video 1>`（抽帧后作为画面参考）
- 参考图 → `<Picture 1>`、`<Picture 2>`...
- 参考视频 → `<Video 2>`、`<Video 3>`（排在源视频之后）
- 音频 → `<Audio 1>`、`<Audio 2>`...

---

## 面板操作

H3 Prompt Planner 前端面板：

| 操作 | 方式 |
|------|------|
| 上传素材 | 点击 📎 按钮选择文件，或拖拽到按钮上 |
| 引用素材 | prompt 里输入 `<` 触发菜单，或点底部按钮 |
| 音频裁剪 | 点音频 chip 上的 ✂ 按钮，设为 2-15 秒片段 |
| 视频预览 | 点视频 chip 上的 ▶ 播放 |
| 图片预览 | 点图片 chip 放大查看 |
| 参数设置 | 点底部 ⚙ 按钮展开参数抽屉 |
| 删除素材 | 点 chip 上的 × |

---

## 参数详解

### 基础参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `model` | — | GGUF 模型，启动时自动扫描 `models/clip/` 子目录 |
| `task_type` | `h3_multi_ref` | t2v / i2v / h3_multi_ref |
| `prompt` | — | 用户指令，用 `<Video 1>`、`<Picture 1>`、`<Audio 1>` 引用素材 |
| `mmproj` | `<none>` | 视觉投影器。有图片/视频输入时必选 |

### 推理参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `temperature` | 0.6 | 采样温度。越高越随机（0.0-2.0），低值更稳定 |
| `seed` | 0 | 随机种子，0=每次随机 |
| `n_ctx` | 8192 | 上下文窗口。token 总数 = prompt + 图片编码 + 生成输出 |
| `n_gpu_layers` | -1 | GPU 层数。-1=全部放显存，显存不够可降到 20-30 |
| `max_tokens` | 2048 | 最大生成 token 数。FINAL_PROMPT 段通常 300-800 token，规划段另算 |
| `thinking_mode` | 关 | 开启后模型会先"思考"再输出，质量略高但慢 2-3 倍 |

### 抽帧参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `frame_mode` | `uniform` | `uniform`=均匀间隔；`smart`=智能关键帧（内容感知） |
| `video_frames` | 5 | 每个视频抽多少帧。2 个视频 = 10 张图片发给模型 |
| `sample_fps` | 0 | 按帧率采样。0=关闭；设为 16 则按 16fps×时长=帧数 |
| `sample_seconds` | 5.0 | 采样时长上限。与 sample_fps 配合，0=取整段视频 |

### 图片质量

| 参数 | 默认 | 说明 |
|------|------|------|
| `image_max_side` | 512 | 图片最长边像素。越大细节越多，但占用更多上下文（平方增长） |

768px vs 512px：图片 token 约 2.25 倍，推理时间增加 30-50%。

---

## 常见工作流

### 工作流 1：角色替换

```
任务：h3_multi_ref
素材：<Picture 1>（新角色外观图）+ <Video 1>（原角色动作视频）
prompt：<Picture 1> 的角色外观替换 <Video 1> 中的人物，保持 <Video 1> 的动作和场景不变
```

### 工作流 2：场景+氛围融合

```
任务：h3_multi_ref
素材：<Picture 1>（场景参考图）+ <Video 1>（动作参考）+ <Audio 1>（氛围音乐）
prompt：融合所有素材，生成一段电影感视频提示词
```

### 工作流 3：纯文本快速构思

```
task_type: t2v
prompt：一个穿红色连衣裙的舞者在雨中的霓虹灯街道独舞，慢镜头
```

---

## 性能调优

### 典型耗时（Gemma 4 E4B, RTX 3060 6GB, 768px）

| 素材 | 帧数 | 用时 |
|------|------|------|
| 纯文本 (t2v) | 0 | ~20s |
| 1 视频 + 1 图 + 1 音频 | 6 | ~78s |
| 2 视频 + 2 图 | 12 | ~90s |
| 2 视频 + 2 图 + 2 音频 | 14 | ~96s |

### 加速建议

| 场景 | 操作 |
|------|------|
| 素材太多 | `video_frames` 降到 3，`image_max_side` 降到 512 |
| 显存紧张 | `n_gpu_layers` 降到 20，`n_ctx` 降到 4096 |
| 多次连续跑 | 正常——每次 20s 加载是固定的 |

### 显存管理

- **用完即焚**：每次执行后 worker 子进程退出，模型显存完全释放
- **残留显存**：主进程 CUDA 缓存会自动回收（`torch.cuda.empty_cache()`）
- **0.8GB 占用**：正常——ComfyUI 自身 + PyTorch CUDA 上下文

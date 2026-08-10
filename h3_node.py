"""
MiniMax H3 Prompt Enhancer — GGUF planner node for MiniMax H3 (omni-modal video model).
Planner model: Gemma 4 12B / any llama.cpp GGUF multimodal model.
Output: H3-style prompts with multi-material references (image + video + audio).

Materials are UPLOADED in the node panel (js/h3_panel.js), not wired as inputs:
the front-end POSTs files to /upload/image and stores a JSON manifest in the
hidden `media_manifest` widget. This module resolves those filenames from the
ComfyUI input directory, decodes them (PIL / PyAV / torchaudio), feeds the
planner, and passes them through as compact media_0..N outputs.
"""

import base64, gc, json, os, re, subprocess, sys, threading, time, atexit
from io import BytesIO
from typing import Dict

import folder_paths
import numpy as np
import torch

# Clean up orphan llama-server processes from crashed sessions
try:
    subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'],
                   capture_output=True, timeout=5)
except Exception:
    pass
from PIL import Image


_MODEL_ROOTS = []
for key in ("clip", "text_encoders"):
    for dp in folder_paths.get_folder_paths(key):
        if os.path.isdir(dp):
            _MODEL_ROOTS.append(dp)
if not _MODEL_ROOTS:
    _MODEL_ROOTS = [os.path.join(folder_paths.models_dir, "clip"),
                    os.path.join(folder_paths.models_dir, "text_encoders")]

def _scan():
    m: Dict[str, str] = {}
    for root in _MODEL_ROOTS:
        if not os.path.isdir(root): continue
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.endswith(".gguf"):
                    rel = os.path.relpath(dp, root)
                    if rel == ".": rel = ""
                    display = f"{rel}\\{fn}" if rel else fn
                    m[display] = os.path.join(dp, fn)
    return m

def _list_models(mp):
    m = _scan()
    return sorted(k for k in m if ("mmproj" in k.lower()) == mp)

def _resolve(d):
    if os.path.isfile(d): return d
    return _scan().get(d, d)


_W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker", "h3_worker.py")


class _MLLM:
    def __init__(self, mp, mm, ctx, gpu, seed, thinking=False):
        self.mp = mp; self.mm = mm or ""; self.ctx = ctx; self.gpu = gpu; self.seed = seed
        self.thinking = thinking
        self.p = None; self.last_error = ''

    def _ensure(self):
        if self.p and self.p.poll() is None: return
        self.p = subprocess.Popen([sys.executable, "-u", _W], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace", bufsize=1, env={**os.environ})
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        print(f"[H3PE] worker pid={self.p.pid}", flush=True)

    def _pump_stderr(self):
        """Forward useful worker logs (handler, loading, errors) to the ComfyUI console.
        Suppresses llama.cpp chat template rendering noise."""
        _KEEP = ("using ", "loaded in", "handler=", "failed:", "error",
                 "Traceback", "File ", "line ", "Error", "Exception")
        _NOISE = ("add_text:", "add_media:", "image_tokens->", "audio_tokens->",
                  "batch_f32", "clip_image", "encoding ", "encoded in",
                  "image decoded", "decoding ", "slice",
                  "<|turn", "<image|>", "<audio|>", "llama_kv_cache",
                  "preproc_out", "grid_x", "mtmd_")
        try:
            for line in self.p.stderr:
                l = line.strip()
                if not l:
                    continue
                if any(n in l for n in _NOISE):
                    continue
                if any(k in l for k in _KEEP):
                    pfx = "" if l.startswith("[h3-worker]") else "[h3-worker] "
                    print(f"{pfx}{line}", end="", flush=True)
        except Exception:
            pass

    def _send(self, req, timeout=3600):
        self._ensure()
        self.p.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        box = {}
        def r():
            try: box["l"] = self.p.stdout.readline()
            except: box["l"] = ""
        t = threading.Thread(target=r, daemon=True); t.start(); t.join(timeout)
        if t.is_alive(): self._kill(); return None
        resp = json.loads(box.get("l") or "{}")
        if not resp.get("ok"):
            self.last_error = resp.get('error', 'unknown')
            print(f"[H3PE] worker error: {self.last_error}", flush=True)
            return None
        self.last_error = ''
        return resp.get("text")

    def _kill(self):
        if self.p and self.p.poll() is None:
            try: self.p.kill()
            except: pass
        self.p = None

    def _pay(self, **kw):
        return dict(model_path=self.mp, mmproj_path=self.mm, n_ctx=self.ctx,
                    n_gpu_layers=self.gpu, n_threads=0, use_mmap=True, use_mlock=False,
                    seed=self.seed, verbose=False, **kw)

    def chat(self, msgs, mt=8192, temp=0.6, rp=1.15):
        return self._send({"cmd": "run", "payload": self._pay(messages=msgs,
            max_tokens=mt, temperature=temp, top_p=0.9, repeat_penalty=rp,
            enable_thinking=self.thinking)})

    def unload(self):
        if self.p and self.p.poll() is None:
            try: self._send({"cmd": "exit"})
            except: pass
            try: self.p.wait(timeout=10)
            except: self._kill()
        self.p = None


def _t2p(img):
    """IMAGE (B,H,W,C float) -> list of PIL Images."""
    if img is None: return []
    t = img.detach().cpu().numpy()
    out = []
    for i in range(t.shape[0]):
        a = np.clip(t[i] * 255.0, 0, 255).astype(np.uint8)
        out.append(Image.fromarray(a))
    return out

def _p2t(pil):
    """Single PIL Image -> IMAGE tensor (1,H,W,C float 0-1)."""
    a = np.asarray(pil.convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(a).unsqueeze(0)

def _b64(pil, max_side):
    if max_side and max_side > 0:
        w, h = pil.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            pil = pil.resize((max(1, int(w*scale)), max(1, int(h*scale))), Image.LANCZOS)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def _sample(pils, n):
    if not pils: return []
    if len(pils) <= n: return pils
    idx = np.linspace(0, len(pils)-1, n).round().astype(int)
    return [pils[i] for i in idx]

def _smart_sample(pils, n):
    """智能均帧（与 Qwen3.5 节点一致）：64x64 缩略图做相邻帧差异，
    首帧必留，其余按画面变化幅度选变化最大的 n-1 帧（运动/切换处优先）。"""
    if not pils: return []
    if len(pils) <= n: return pils
    ts = 64
    thumbs = [np.asarray(im.resize((ts, ts), Image.NEAREST), np.float32) for im in pils]
    diffs = [np.abs(thumbs[i] - thumbs[i-1]).mean() for i in range(1, len(pils))]
    scores = [(max(diffs[i-1], diffs[i]), i) if i < len(pils)-1 else (diffs[i-1], i)
              for i in range(1, len(pils))]
    scores.sort(key=lambda x: x[0], reverse=True)
    picked = {0}
    for _, i in scores:
        if len(picked) >= n: break
        picked.add(i)
    return [pils[i] for i in sorted(picked)]


# ------------------------------------------------------- uploaded media I/O
_IMG_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
_AUD_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
_VID_EXTS = {'.mp4', '.webm', '.mov', '.mkv', '.avi', '.m4v'}


def _media_path(entry):
    """Manifest entry -> absolute path under the ComfyUI input directory."""
    if isinstance(entry, dict):
        name = entry.get('name', '')
        sub = entry.get('subfolder', '')
    else:
        name, sub = str(entry), ''
    if not name:
        return None
    rel = os.path.join(sub, name) if sub else name
    base = folder_paths.get_input_directory()
    path = os.path.abspath(os.path.join(base, os.path.normpath(rel)))
    if os.path.commonpath((os.path.abspath(base), path)) != os.path.abspath(base):
        return None
    return path if os.path.isfile(path) else None


def _load_image_pils(path):
    try:
        im = Image.open(path).convert('RGB')
        return [im]
    except Exception as e:
        print(f'[H3PE] image load failed {path}: {e}', flush=True)
        return []


def _load_video_pils_and_obj(path):
    """Decode video via PyAV (ComfyUI's VideoFromFile). Returns (pils, VideoFromFile, real_fps, seconds)."""
    try:
        from comfy_api.latest._input_impl.video_types import VideoFromFile
        vf = VideoFromFile(path)
        comp = vf.get_components()
        imgs = comp.images
        t = imgs.detach().cpu().numpy() if hasattr(imgs, 'detach') else np.asarray(imgs)
        pils = []
        for i in range(t.shape[0]):
            a = t[i]
            if a.dtype != np.uint8:
                a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
            pils.append(Image.fromarray(a))
        real_fps, seconds = 24.0, 0.0
        try:
            real_fps = float(comp.frame_rate) if comp.frame_rate else 24.0
            seconds = vf.get_duration()
        except Exception:
            seconds = len(pils) / real_fps if real_fps else 0.0
        if not seconds:
            seconds = len(pils) / (real_fps or 24.0)
        return pils, vf, (real_fps or 24.0), seconds
    except Exception as e:
        print(f'[H3PE] video load failed {path}: {e}', flush=True)
        return [], None, 24.0, 0.0


def _sample_by_fps(pils, fps, seconds, max_seconds):
    """Uniform sampling at `fps` fps, capped to max_seconds of content.
    Frame count = fps * min(seconds, max_seconds) + 1 (first frame inclusive)."""
    if not pils:
        return []
    total = len(pils)
    use_seconds = min(seconds, max_seconds) if (max_seconds and max_seconds > 0) else seconds
    n = max(1, int(round(fps * use_seconds)) + 1)
    n = min(n, total)
    idx = np.linspace(0, total - 1, n).round().astype(int)
    return [pils[i] for i in idx]


def _load_audio(path, trim=None):
    """Audio file -> ComfyUI AUDIO dict {waveform, sample_rate}. Optional trim {start,end} seconds.
    Loader chain: scipy(wav) -> soundfile(flac/ogg/mp3) -> torchaudio(fallback)."""
    wf, sr = None, None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == '.wav':
            import scipy.io.wavfile as _wv
            sr, data = _wv.read(path)
            a = data.astype(np.float32)
            if a.ndim == 2: a = a.T                      # (samples,ch) -> (ch,samples)
            if a.ndim == 1: a = a[None, :]
            mx = float(np.iinfo(data.dtype).max) if data.dtype.kind == 'i' else 1.0
            wf = torch.from_numpy(a / (mx if mx else 1.0))
        else:
            import soundfile as _sf
            a, sr = _sf.read(path, dtype='float32', always_2d=True)
            wf = torch.from_numpy(a.T)                   # (samples,ch) -> (ch,samples)
    except Exception:
        try:
            import torchaudio
            wf, sr = torchaudio.load(path)
        except Exception as e:
            print(f'[H3PE] audio load failed {path}: {e}', flush=True)
            return None
    try:
        wf = wf.float()
        if wf.ndim == 1: wf = wf.unsqueeze(0)
        if wf.shape[0] > 2: wf = wf[:2]
        if trim:
            s = max(0, int(float(trim.get('start', 0)) * sr))
            e0 = float(trim.get('end', 0))
            e = int(e0 * sr) if e0 > 0 else wf.shape[-1]
            e = min(e, wf.shape[-1])
            if e > s: wf = wf[..., s:e]
        return {'waveform': wf.unsqueeze(0).float(), 'sample_rate': int(sr)}
    except Exception as e:
        print(f'[H3PE] audio load failed {path}: {e}', flush=True)
        return None


def _parse_manifest(s):
    if not s:
        return {}
    try:
        d = json.loads(s)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------- H3 templates
#
# H3 specs: 5-15s output, 24FPS, native stereo audio, aspect 21:9/16:9/4:3/1:1/3:4/9:16
# Material limits: <=9 images + <=3 videos + <=3 audios (12 files max)

SYSTEM = ("You are an H3 video prompt planner. Observe materials, analyze roles, "
          "then output structured sections. The prompt text (EXECUTE, FINAL_PROMPT, "
          "soundscape, music) must be written entirely in English. ")

USER_TEMPLATE = """用户需求：{prompt}

{materials_block}{intent}按 H3 官方规范输出，分段如下：
[OBSERVATION] 逐一观察每个素材的真实画面和声音，只写观察到的不要臆造——把人物外观、衣着样式颜色、场景色调材质、动作节奏、音频的台词音色原样记录{asr_hint}
[UNDERSTAND] 判断每个素材的参考角色：哪几个提供人物外观、哪几个提供场景氛围、哪几个提供动作/运镜/节奏、哪几个提供音色/台词
[EXECUTE] 英文 detailed_description，300-500 words：
  开头声明 visual style + color palette + lighting
  用 [Shot N] 分镜，首镜无时间戳，后续用 At 00:00.000 标注切镜
  每镜头写明主体、构图、动作、光线、同步音效
  运镜：camera pushes in with small amplitude at slow speed 等自然入句
  对白：(Sx) says: <d>[Language] 台词 </d>；画外音补 lips remain closed
  素材融入：<Picture N> 的外观特征、<Video N> 的运镜节奏、<Audio N> 的台词音色
[PRESERVE] 必须保留的元素（要点列举）
[FINAL_PROMPT] 提炼为 H3 标准三段式最终提示词（全部英文）：
  integrated_multimodal_description: （EXECUTE 浓缩版，[Shot N]+At 00:00.000 分镜，300-500 words）
  overall_soundscape: 1-4句环境音/动作声/非语言人声
  non_diegetic_music: 1-3句配乐描述（无配乐则写 N/A）"""


def _render_template(tpl, prompt, materials_block, intent_text, asr_hint):
    """Render the user instruction template; unknown placeholders are dropped."""
    _map = {'prompt': prompt, 'materials_block': materials_block,
            'intent': intent_text, 'asr_hint': asr_hint}
    return re.sub(r'\{(\w+)\}', lambda m: _map.get(m.group(1), ''), tpl or USER_TEMPLATE)


def _build_msgs(sp, ut, imgs, labels=None):
    """imgs: list of base64 strings. labels: optional per-image labels."""
    if not imgs:
        return [{"role": "system", "content": sp}, {"role": "user", "content": ut}]
    content = []
    if labels is None:
        labels = [f"Ref-{i}" for i in range(len(imgs))]
    for i, b in enumerate(imgs):
        content.append({"type": "text", "text": f"[{labels[i]}]:\n"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}})
    content.append({"type": "text", "text": "\nAbove are the reference materials. Observe them."})
    return [{"role": "system", "content": sp},
            {"role": "user", "content": content},
            {"role": "user", "content": ut}]


def _route(tt, prompt, n_src=0, n_img=0, n_vid=0, audio_files=None, vid_nums=None, img_nums=None, aud_nums=None, system=None, user_template=None):
    """Builds dynamic template. Numbering matches official H3 Reference node: 1-based (Image 1, Video 1, Audio 1).
    vid_nums/img_nums/aud_nums are the user's manifest numbers (may have gaps after deletions).

    节点特色：先观察思考（OBSERVATION/UNDERSTAND）再规划增强（EXECUTE）。
    素材只按类型中性列出（Video N/Picture N/Audio N），角色分工不设死——
    模型根据实际画面内容和用户 @引用 在 UNDERSTAND 里自由判断。"""
    vnums = vid_nums or []
    inums = img_nums or []
    anums = aud_nums or (list(range(1, len(audio_files) + 1)) if audio_files else [])

    # ---- neutral material list with H3-style role hints ------------------------
    items = []
    if n_src:
        items.append(f"<Video {vnums[0]}>（{n_src} 帧，共 {n_src} 帧）")
    if n_vid:
        rv = vnums[1:] if n_src else vnums
        items.append(f"<Video {'>、<Video '.join(str(n) for n in rv)}>（共 {n_vid} 帧）")
    if n_img:
        items.append(f"<Picture {'>、<Picture '.join(str(n) for n in inums)}>")
    if audio_files:
        items.append(f"<Audio {'>、<Audio '.join(str(n) for n in anums)}>（可直接听到；若含语音尝试 ASR 转录）")
    material_text = ("已上传素材：" + "；".join(items) + "\n"
                     "H3 支持角色/场景/动作/镜头/音色多维度参考——"
                     "用户可指定哪个素材用作人物外观、场景氛围、动作节奏、镜头运镜或音色台词。\n") if items else ""

    # ---- ASR instruction (E4B supports speech recognition with explicit prompting)
    asr_hint = ""
    if audio_files:
        asr_hint = ("\n当素材包含音频时，请在 OBSERVATION 中对每段音频做语音识别（ASR）："
                    "若音频含人声，用 Transcription: ... 写出听到的文本；"
                    "若为音乐/环境声，描述其风格、节奏与情绪特征。\n")

    # ---- light task intent (one open-ended line each) ---------------------------
    intent = {
        "t2v":         "任务方向：纯文本生成视频，自由发挥画面与叙事。",
        "i2v":         "任务方向：图生视频——图片通常作为视频的起始帧/首尾帧，但也可按用户说明作其他用途。",
        "h3_multi_ref":"任务方向：全能参考——自由组合所有素材的人物、场景、动作、声音等元素生成视频。",
    }.get(tt, "")
    intent_text = (intent + "\n") if intent else ""

    # ---- assemble: render from editable template (default USER_TEMPLATE) --------
    materials_block = (material_text
        + "（用户在需求里可能用 &lt;Video N&gt;/&lt;Picture N&gt;/&lt;Audio N&gt; 引用素材——编号与上面一致；"
        + "也可自由指定素材用途，例如『<Picture 1> 的角色外观，<Video 1> 的场景运镜，<Audio 1> 的台词情感』。）\n\n"
        ) if items else ""
    ut = _render_template(user_template, prompt, materials_block, intent_text, asr_hint)

    if tt == "i2v": order = "video+ref"
    elif n_src and (n_img or n_vid): order = "video+ref"
    elif n_img or n_vid:        order = "ref"
    elif n_src:                 order = "video"
    else:                       order = "none"
    return (system or SYSTEM, ut, order)



import comfy.model_management
from comfy_api.latest import io, ComfyExtension

# Pre-build model lists once (mmproj scanning is expensive)
_H3_MODELS = ['<no .gguf>']
_H3_MMPROJS = ['<none>']

def _refresh_models():
    """Re-scan GGUF directories. Called on every execute so new files appear
    without restarting ComfyUI. Lists are mutated in-place so io.Combo.Input
    options (which hold references to these lists) stay in sync."""
    models = _list_models(False)
    _H3_MODELS[:] = models if models else ['<no .gguf>']
    mm = _list_models(True)
    _H3_MMPROJS[:] = ['<none>'] + mm
_refresh_models()  # initial scan on import

# Smart defaults: prefer E4B, fallback to first found
_H3_DEFAULT_MODEL = ''
_H3_DEFAULT_MMPROJ = '<none>'
for m in _H3_MODELS:
    if m == '<no .gguf>': continue
    ml = m.lower()
    if 'e4b' in ml:
        _H3_DEFAULT_MODEL = m
        for p in _H3_MMPROJS:
            if p != '<none>' and os.path.dirname(p) == os.path.dirname(m):
                _H3_DEFAULT_MMPROJ = p
                break
        break
if not _H3_DEFAULT_MODEL:
    for m in _H3_MODELS:
        if m == '<no .gguf>': continue
        ml = m.lower()
        if 'gemma' in ml:
            _H3_DEFAULT_MODEL = m
            for p in _H3_MMPROJS:
                if p != '<none>' and os.path.dirname(p) == os.path.dirname(m):
                    _H3_DEFAULT_MMPROJ = p
                    break
            break
if not _H3_DEFAULT_MODEL:
    for m in _H3_MODELS:
        if m != '<no .gguf>':
            _H3_DEFAULT_MODEL = m
            for p in _H3_MMPROJS:
                if p != '<none>' and os.path.dirname(p) == os.path.dirname(m):
                    _H3_DEFAULT_MMPROJ = p
                    break
            break


class H3PromptEnhancer(io.ComfyNode):
    TASKS = ['t2v', 'i2v', 'h3_multi_ref']

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3PromptEnhancer',
            display_name='H3 Prompt Planner (GGUF)',
            category='Bernini',
            description='Multi-modal prompt planner for MiniMax H3. Upload materials in the node panel.',
            inputs=[
                io.Combo.Input('model', options=_H3_MODELS, default=_H3_DEFAULT_MODEL),
                io.Combo.Input('mmproj', options=_H3_MMPROJS, default=_H3_DEFAULT_MMPROJ),
                io.Combo.Input('task_type', options=cls.TASKS, default='h3_multi_ref'),
                io.String.Input('prompt', default='', multiline=True),
                io.String.Input('media_manifest', default=''),
                io.Float.Input('temperature', default=0.6, min=0.0, max=2.0, step=0.05),
                io.Float.Input('repeat_penalty', default=1.15, min=1.0, max=2.0, step=0.01),
                io.Int.Input('seed', default=0, min=0, max=2**31 - 1),
                io.Int.Input('n_ctx', default=8192, min=2048, max=32768),
                io.Int.Input('n_gpu_layers', default=-1, min=-1, max=200),
                io.Int.Input('max_tokens', default=2048, min=128, max=16384),
                io.Combo.Input('frame_mode', options=['uniform', 'smart'], default='uniform',
                    tooltip='抽帧方式：uniform=均匀间隔抽帧；smart=智能（首/中/尾关键帧）'),
                io.Int.Input('video_frames', default=5, min=1, max=64,
                    tooltip='每个视频抽的帧数（采样帧率=0 时生效）'),
                io.Int.Input('sample_fps', default=0, min=0, max=60,
                    tooltip='采样帧率：0=关闭（用上面的每视频帧数）；>0 时按 帧率×时长 自动算帧数，如 16fps×5s=81帧'),
                io.Float.Input('sample_seconds', default=5.0, min=0.0, max=15.0, step=0.5,
                    tooltip='采样时长上限（秒）：与采样帧率配合，0=取整段视频'),
                io.Int.Input('image_max_side', default=512, min=64, max=4096, step=32,
                    tooltip='喂给 planner 的图片最长边（常用 384/512/768/1024/1536），直接输入任意值'),
                io.Boolean.Input('thinking_mode', default=False),
                # 追加在末尾：ComfyUI 按位置索引恢复 widgets_values，
                # 放在中间会让旧工作流的值错位（media_manifest JSON 串到 system_prompt）
                io.String.Input('system_prompt', default=SYSTEM, multiline=True,
                    tooltip='系统提示词，默认内置 H3 planner 系统提示，可自行编辑覆盖'),
                io.String.Input('user_template', default=USER_TEMPLATE, multiline=True,
                    tooltip='用户指令模板（输出规范），占位符：{prompt} {materials_block} {intent} {asr_hint}'),
            ],
            outputs=[
                io.String.Output(display_name='enhanced_prompt'),
                io.String.Output(display_name='structured_plan'),
            ] + [io.MatchType.Output(template=io.MatchType.Template(
                        f'h3_media_{i}', [io.Image, io.Video, io.Audio]),
                    display_name=f'media_{i}') for i in range(15)],
        )

    @classmethod
    def execute(cls, model, mmproj, task_type, prompt, media_manifest='',
            system_prompt=None, user_template=None, temperature=0.6, repeat_penalty=1.15, seed=0,
            n_ctx=8192, n_gpu_layers=-1, max_tokens=2048,
            frame_mode='uniform', video_frames=5, sample_fps=0, sample_seconds=5.0,
            image_max_side=512, thinking_mode=False):
        _refresh_models()  # pick up newly added GGUF files between runs
        raw = (prompt or '').strip()
        _defaults = {
            't2v': '\u751f\u6210\u4e00\u6bb5\u9ad8\u8d28\u91cf\u89c6\u9891',
            'i2v': '以图片为起始帧生成视频',
            'h3_multi_ref': '\u7ed3\u5408\u53c2\u8003\u7d20\u6750\u751f\u6210\u89c6\u9891',
        }
        if not raw:
            raw = _defaults.get(task_type, 'enhance')
            prompt = raw

        mp = _resolve(model)
        mm = _resolve(mmproj) if mmproj and mmproj != '<none>' else None
        if not os.path.isfile(mp): return tuple([f'ERROR: {mp}', ''] + [None] * 15)

        # ---------------- resolve uploaded materials from manifest ----------------
        man = _parse_manifest(media_manifest)
        # frame_mode: uniform (均匀) / smart (智能均帧，与 Qwen3.5 同款内容感知算法)
        # 智能均帧 = 只选"每视频帧数"帧（首帧+变化最大），永远少量；
        # 采样帧率(fps×秒) 只作用于均匀模式——是省 token 的工具
        smart = (frame_mode == 'smart')
        fps_val = sample_fps if (sample_fps and sample_fps > 0) else 0
        side_val = int(image_max_side or 512)

        def _pick(pils, seconds):
            if not pils: return []
            if smart:
                return _smart_sample(pils, min(video_frames, len(pils)))
            return _sample_by_fps(pils, fps_val, seconds, sample_seconds) if fps_val else _sample(pils, video_frames)

        src_pils_all, src_vf = [], None
        src_fps, src_sec = 24.0, 0.0
        if man.get('source_video'):
            p = _media_path(man['source_video'])
            if p: src_pils_all, src_vf, src_fps, src_sec = _load_video_pils_and_obj(p)
        src = _pick(src_pils_all, src_sec)

        rvid_pils, rvid_objs = [], []
        rvid_full = []  # full decoded frames for passthrough
        for e in (man.get('reference_videos') or [])[:3]:
            p = _media_path(e)
            if not p: continue
            pls, vf, rfps, rsec = _load_video_pils_and_obj(p)
            if pls and vf is not None:
                rvid_full.append(pls)
                rvid_pils.append(_pick(pls, rsec)); rvid_objs.append(vf)
        rvid = [f for pls in rvid_pils for f in pls]

        ref = []
        for e in (man.get('images') or [])[:9]:
            p = _media_path(e)
            if p: ref += _load_image_pils(p)[:1]
        ref_slots = list(range(len(ref)))

        audio_paths, audio_trims = [], []
        for e in (man.get('audios') or [])[:3]:
            p = _media_path(e)
            if p:
                audio_paths.append(p)
                audio_trims.append(e.get('trim') if isinstance(e, dict) else None)
        audio_objs = [a for a in (_load_audio(p, t) for p, t in zip(audio_paths, audio_trims)) if a]
        audio_files = audio_paths
        has_trim = any(audio_trims)

        # Numbering: 1-based to match official H3 Reference node ("Image 1"/"Video 1"/"Audio 1")
        vid_nums = list(man.get('vid_nums') or [])
        img_nums = list(man.get('img_nums') or [])
        aud_nums = list(man.get('aud_nums') or [])
        if not vid_nums:
            vid_nums = list(range(1, (1 if src_vf is not None else 0) + len(rvid_objs) + 1))
        if not img_nums:
            img_nums = list(range(1, len(ref) + 1))
        if not aud_nums:
            aud_nums = list(range(1, len(audio_files) + 1))

        sp, ut, order = _route(task_type, prompt, system=system_prompt, user_template=user_template, n_src=len(src), n_img=len(ref), n_vid=len(rvid),
                               audio_files=audio_files, vid_nums=vid_nums, img_nums=img_nums, aud_nums=aud_nums)
        if audio_files:
            parts = '; '.join(f'<Audio {aud_nums[i]}>:{os.path.basename(ap)}' for i, ap in enumerate(audio_files))
            ut = f'{ut}\n\n音频素材：{parts}。按编号引用。'
        src_num = vid_nums[0] if vid_nums else 1
        rv_nums = vid_nums[1:] if (src_vf is not None) else vid_nums

        # Base64 audio payloads for native input_audio (Gemma 4 E4B/12B audio support).
        # Trimmed clips are re-encoded to a temp WAV so the planner hears the cut, not the full file.
        audio_b64, _tmp_wavs = [], []
        if audio_files:
            import base64 as _b64mod, tempfile, scipy.io.wavfile as _wav
            for ap, tr in zip(audio_paths, audio_trims):
                try:
                    if tr:
                        obj = _load_audio(ap, tr)
                        if obj is None: audio_b64.append(None); continue
                        wf = obj['waveform'].squeeze(0).cpu().numpy()
                        if wf.ndim == 2 and wf.shape[0] > wf.shape[-1]: wf = wf.T
                        mono = wf[0] if wf.ndim == 2 else wf
                        tf = tempfile.NamedTemporaryFile(suffix='_h3trim.wav', delete=False)
                        tf.close()
                        _wav.write(tf.name, obj['sample_rate'], (mono * 32767).astype('int16'))
                        _tmp_wavs.append(tf.name)
                        with open(tf.name, 'rb') as f:
                            audio_b64.append(_b64mod.b64encode(f.read()).decode())
                    else:
                        with open(ap, 'rb') as f:
                            audio_b64.append(_b64mod.b64encode(f.read()).decode())
                except Exception:
                    audio_b64.append(None)

        # Build messages at a given image side. On ctx overflow we retry with a
        # smaller side (frame count preserved, resolution drops — token cost
        # scales with side^2, so this fits huge frame counts into n_ctx).
        def _build_all(side):
            imgs_src  = [_b64(p, side) for p in src]
            imgs_ref  = [_b64(p, side) for p in ref]
            imgs_rvid = [_b64(p, side) for p in rvid]

            if order == 'video+ref':
                sc, rc = [], []
                for i, b in enumerate(imgs_src):
                    sc += [{'type':'text','text':f'[<Video {src_num}> 第{i+1}帧]:\n'},{'type':'image_url','image_url':{'url':f'data:image/png;base64,{b}'}}]
                sc.append({'type':'text','text':f'\n以上是源视频（<Video {src_num}>）的抽帧。'})
                for i, b in enumerate(imgs_ref):
                    rc += [{'type':'text','text':f'[<Picture {img_nums[i]}>]:\n'},{'type':'image_url','image_url':{'url':f'data:image/png;base64,{b}'}}]
                for i, b in enumerate(imgs_rvid):
                    vn = rv_nums[min(i, max(0, len(rv_nums)-1))] if rv_nums else (src_num + 1)
                    rc += [{'type':'text','text':f'[<Video {vn}> 第{i+1}帧]:\n'},{'type':'image_url','image_url':{'url':f'data:image/png;base64,{b}'}}]
                rc.append({'type':'text','text':'\n以上是参考素材。'})
                return ([{'role':'system','content':sp},{'role':'user','content':sc},{'role':'user','content':rc},{'role':'user','content':ut}],
                        len(imgs_src) + len(imgs_ref) + len(imgs_rvid))

            labels = None
            if order == 'ref':
                imgs = imgs_ref + imgs_rvid
                labels = [f'<Picture {img_nums[i]}>' for i in range(len(imgs_ref))] + \
                         [f'<Video {rv_nums[min(i, max(0, len(rv_nums)-1))] if rv_nums else 1}> 第{i+1}帧' for i in range(len(imgs_rvid))]
            elif order == 'video':
                imgs = imgs_src + imgs_rvid
                labels = [f'<Video {src_num}> 第{i+1}帧' for i in range(len(imgs_src))] + \
                         [f'<Video {rv_nums[min(i, max(0, len(rv_nums)-1))] if rv_nums else src_num+1}> 第{i+1}帧' for i in range(len(imgs_rvid))]
            else:
                imgs = {'none':[],'ref_or_first':(imgs_ref or imgs_src[:1])+imgs_rvid}[order]
                if order == 'ref_or_first':
                    if imgs_ref:
                        labels = [f'<Picture {img_nums[i]}>' for i in range(min(len(imgs_ref), len(imgs)))]
                    elif imgs_src:
                        labels = [f'<Video {src_num}> 第1帧']
            return (_build_msgs(sp, ut, imgs, labels), len(imgs))

        def _attach_audio(msgs):
            if not audio_b64: return msgs
            last = msgs[-1]
            if isinstance(last['content'], str):
                last['content'] = [{'type': 'text', 'text': last['content']}]
            for i, b64 in enumerate(audio_b64):
                if b64:
                    last['content'].append({'type': 'text', 'text': f'\n[<Audio {aud_nums[i]}>] 参考音频:'})
                    last['content'].append({'type': 'input_audio', 'input_audio': {'data': b64, 'format': 'wav'}})
            return msgs

        t0 = time.time()
        mllm = _MLLM(mp, mm, n_ctx, n_gpu_layers, seed, thinking_mode)
        result, imgs_used, side_used = None, 0, side_val
        try:
            # Retry ladder on ctx overflow: shrink image side (not frame count).
            sides = sorted(set([side_val, 512, 384, 256, 192, 128]), reverse=True)
            for attempt, side in enumerate(sides):
                msgs, imgs_used = _build_all(side)
                msgs = _attach_audio(msgs)
                if attempt == 0:
                    print(f'[H3PE] {task_type} imgs={imgs_used} side={side} audio={len(audio_files)}', flush=True)
                else:
                    print(f'[H3PE] ctx overflow -> side {side_val}->{side} retry', flush=True)
                result = mllm.chat(msgs, mt=max_tokens, temp=temperature, rp=repeat_penalty)
                if result is not None:
                    side_used = side
                    break
                err = getattr(mllm, 'last_error', '') or ''
                mllm._kill()
                if 'n_ctx' not in err and 'exceeds' not in err and attempt > 0:
                    # non-context failure after a retry — one more try at same side
                    result = mllm.chat(msgs, mt=max_tokens, temp=temperature, rp=repeat_penalty)
                    side_used = side
                    break
        finally:
            mllm.unload()
            for tw in _tmp_wavs:
                try: os.unlink(tw)
                except: pass
            gc.collect()
            try: torch.cuda.empty_cache()
            except: pass
        print(f'[H3PE] {task_type} imgs={imgs_used} side={side_used} {time.time()-t0:.1f}s', flush=True)
        result = result or ''
        plan, desc = '', result
        m = re.search(r'(?:^|\n)\*{0,2}\[?\*{0,2}\s*FINAL[\s_]PROM\w+T\]?\*{0,2}\s*:?\s*', result, re.IGNORECASE)
        if m: plan = result[:m.start()].strip(); desc = result[m.end():].strip()
        else: desc = result; plan = '[Direct]'
        desc = re.sub(r'^\s*assistant\s*\n*', '', desc, flags=re.IGNORECASE).strip()
        desc = desc.lstrip('*').lstrip('-').strip()
        # Compact passthrough: all IMAGE batches (video frames stacked, images single),
        media = []
        if src_pils_all:
            media.append(torch.cat([_p2t(p) for p in src_pils_all], dim=0))
        for pls in rvid_full:
            media.append(torch.cat([_p2t(p) for p in pls], dim=0))
        for p in ref:
            media.append(_p2t(p))
        media += audio_objs
        return tuple([desc or '', plan or ''] + (media + [None] * 15)[:15])


NODE_CLASS_MAPPINGS = {'H3PromptEnhancer': H3PromptEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {'H3PromptEnhancer': 'H3 Prompt Planner (GGUF)'}

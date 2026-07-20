"""
Qwen3.5 Prompt Enhancer — separate GGUF node for Qwen3.5-VL models.
Simpler than Bernini node: direct prompt output, no structured format.
"""

import base64, gc, json, os, re, subprocess, sys, threading, time
from io import BytesIO
from typing import Dict

import folder_paths
import numpy as np
import torch
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


_W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker", "qwen35_worker.py")


class _MLLM:
    def __init__(self, mp, mm, ctx, gpu, seed, thinking=False):
        self.mp = mp; self.mm = mm or ""; self.ctx = ctx; self.gpu = gpu; self.seed = seed
        self.thinking = thinking; self.p = None

    def _ensure(self):
        if self.p and self.p.poll() is None: return
        self.p = subprocess.Popen([sys.executable, "-u", _W], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=sys.stderr, text=True, encoding="utf-8",
            errors="replace", bufsize=1, env={**os.environ})
        print(f"[Qwen35PE] worker pid={self.p.pid}", flush=True)

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
            print(f"[Qwen35PE] worker error: {resp.get('error', 'unknown')}", flush=True)
            return None
        return resp.get("text")

    def _kill(self):
        p, self.p = self.p, None
        if p:
            try: p.stdin.close()
            except: pass
            try: p.terminate(); p.wait(5)
            except:
                try: p.kill()
                except: pass
        gc.collect()

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
        self._kill()


def _t2p(t):
    if t is None: return []
    a = t.detach().cpu().float().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, np.float32)
    if a.ndim == 3: a = a[None]
    return [Image.fromarray((np.clip(a[i], 0, 1) * 255).round().astype(np.uint8)) for i in range(a.shape[0])]

def _sample(p, k):
    n = len(p)
    if n < 2: return p
    return [p[max(0, min(n-1, round(i*(n-1)/max(k-1,1))))] for i in range(k)]

def _b64(img, ms=0):
    if ms and max(img.size) > ms:
        s = ms / max(img.size)
        img = img.resize((max(1, int(img.size[0]*s)), max(1, int(img.size[1]*s))), Image.LANCZOS)
    b = BytesIO(); img.convert("RGB").save(b, "PNG")
    return base64.b64encode(b.getvalue()).decode()

def _build_msgs(sp, ut, imgs, labels=None):
    if not imgs: return [{"role": "system", "content": sp}, {"role": "user", "content": ut}]
    if labels is None: labels = [str(i) for i in range(len(imgs))]
    c = []
    for i, b in enumerate(imgs):
        c.append({"type": "text", "text": f"[{labels[i]}]:\n"})
        c.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}})
    c.append({"type": "text", "text": "\n" + ut})
    return [{"role": "system", "content": sp}, {"role": "user", "content": c}]


SYSTEM = "You are a semantic planner and prompt enhancer. Reply in natural language only. Never output coordinates, bounding boxes, HTML tags, or think/reasoning tags."

# Structured templates (same as Bernini node — Qwen3.5 should handle reliably)
R2V = """You are a subject-driven video generation planner. {image_num} reference image(s) of subjects provided.

Original description: {original_text}

RULES:
- [OBSERVATION]: Describe each reference image in detail — subject appearance, clothing, features. Note ONLY what you actually see.
- [UNDERSTAND]: Explain what the original description means.
- [EXECUTE]: Plan a video featuring these subjects (reference as "the subject from image0"). Include scene, environment, temporal action sequence, lighting, camera.
- [PRESERVE]: Subject identity and appearance must match reference images exactly.
- [FINAL_PROMPT]: Write a descriptive video paragraph (4-8 sentences). Describe the final scene as if narrating a film — NOT instructions, NOT a checklist. Only describe what the viewer sees.

[OBSERVATION]
[UNDERSTAND]
[EXECUTE]
[PRESERVE]
[FINAL_PROMPT]"""

R2I = """You are a subject-driven image generation planner. {image_num} reference image(s) of subjects provided.

Original description: {original_text}

RULES:
- [OBSERVATION]: Describe each reference image in detail — subject appearance, clothing, features. Note ONLY what you actually see.
- [UNDERSTAND]: Explain what the original description means.
- [EXECUTE]: Plan an image featuring these subjects (reference as "the subject from image0"). Include scene, environment, lighting, composition, framing.
- [PRESERVE]: Subject identity and appearance must match reference images exactly.
- [FINAL_PROMPT]: Write a descriptive video paragraph (4-8 sentences). Describe the final scene as if narrating a film — NOT instructions, NOT a checklist. Only describe what the viewer sees.

[OBSERVATION]
[UNDERSTAND]
[EXECUTE]
[PRESERVE]
[FINAL_PROMPT]"""

VR2V = """You are a video editing planner. Source video frames and {image_num} reference image(s) provided.

User instruction: {original_text}

RULES:
- [OBSERVATION]: Describe the source video (scene, subjects, actions, lighting, weather, camera, mood). Then describe EACH reference image independently — use the Ref-X label shown with each image. Include what it shows, lighting, colors, atmosphere, style, any notable details. NEVER write "similar to Ref-X".
- [UNDERSTAND]: Explain what the instruction means based on your observations. What changes? What stays?
- [EXECUTE]: Plan the visual changes step by step. Be specific.
- [PRESERVE]: List everything that stays exactly the same.
- [FINAL_PROMPT]: Write a descriptive video paragraph (4-8 sentences). Describe the final scene as if narrating a film — NOT instructions, NOT a checklist. Only describe what the viewer sees.

[OBSERVATION]
[UNDERSTAND]
[EXECUTE]
[PRESERVE]
[FINAL_PROMPT]"""

PLAN_SUFFIX = """

RULES:
- [OBSERVATION]: Describe the source (scene, subjects, lighting, camera). Then describe each reference image — what it shows, color, material, shape, text.
- [UNDERSTAND]: What the user instruction means for these specific subjects.
- [EXECUTE]: Step-by-step visual changes — which subjects, how they change, new positions, appearances, motions.
- [PRESERVE]: Everything that stays exactly the same.
- [FINAL_PROMPT]: Write a descriptive video paragraph (4-8 rich sentences). Describe the final scene narratively — NOT instructions or a checklist. Only describe what the viewer sees.

[OBSERVATION]
[UNDERSTAND]
[EXECUTE]
[PRESERVE]
[FINAL_PROMPT]"""


def _route(tt, prompt, nr):
    n = max(nr, 1)
    ds = SYSTEM
    r = {
        "t2v":   (ds, f'Prompt: "{prompt}". Add cinematic details. Output English prompt only.', "none"),
        "t2i":   (ds, f'Prompt: "{prompt}". Add cinematic details. Output English prompt only.', "none"),
        "v2v":   (ds, V2V.format(user_prompt=prompt), "video"),
        "mv2v":  (ds, V2V.format(user_prompt=prompt), "video"),
        "i2i":   (ds, I2I.format(user_prompt=prompt), "ref"),
        "i2v":   (I2V_SP, I2V.format(user_prompt=prompt, image_num=n), "ref_or_first"),
        "ads2v": (ds, f'Video frames. Ad instruction: "{prompt}". Output ONE English ad placement sentence.', "video"),
        "vi2v":  (ds, f'Video + {n} reference image(s). Instruction: "{prompt}". Output ONE editing sentence.', "video+ref"),
        "r2v":   (ds, R2V.format(image_num=n, original_text=prompt), "ref"),
        "r2i":   (ds, R2I.format(image_num=n, original_text=prompt), "ref"),
        "rv2v":  (ds, VR2V.format(image_num=n, image_num_1=n-1, original_text=prompt), "video+ref"),
        "vrc2v": (ds, VR2V.format(image_num=n, image_num_1=n-1, original_text=prompt), "video+ref"),
    }
    sp, ut, order = r.get(tt, (ds, f'Instruction: "{prompt}". Generate an enhanced prompt.', "none"))
    if order != "none" and tt not in ("r2v", "r2i", "rv2v", "vrc2v", "vi2v", "ads2v", "i2v"):
        ut += PLAN_SUFFIX
    return sp, ut, order


# Short task templates (without built-in plan — PLAN_SUFFIX appended)
V2V = """Source video frames provided.
User instruction: "{user_prompt}"
Reply in descriptive English text. Do NOT output coordinates or bounding boxes."""

I2I = """Source image provided.
User instruction: "{user_prompt}"
Reply in descriptive English text. Do NOT output coordinates or bounding boxes."""

I2V = """{image_num} reference image(s) provided.
User intent: "{user_prompt}" """

I2V_SP = ("You are a video generation planner. Reply in natural language. "
    "Never output coordinates, bounding boxes, or HTML tags. "
    "Output using these EXACT sections: "
    "[OBSERVATION] — describe the reference image; "
    "[UNDERSTAND] — what the user intent means; "
    "[EXECUTE] — plan the video step by step; "
    "[PRESERVE] — key elements that stay consistent; "
    "[FINAL_PROMPT] — descriptive video paragraph (4-8 sentences, film-narrative style).")


class Qwen35PromptEnhancer:
    CATEGORY = "Bernini"
    FUNCTION = "enhance"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "structured_plan")
    TASKS = ["t2v","t2i","v2v","mv2v","i2i","i2v","ads2v","vi2v","r2v","r2i","rv2v","vrc2v"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (_list_models(False) or ["<no .gguf>"],),
            "mmproj": (["<none>"] + _list_models(True),),
            "task_type": (cls.TASKS, {"default": "v2v"}),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
            "repeat_penalty": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            "n_ctx": ("INT", {"default": 8192, "min": 2048, "max": 32768}),
            "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200}),
            "max_tokens": ("INT", {"default": 4096, "min": 128, "max": 16384}),
            "video_frames": ("INT", {"default": 3, "min": 1, "max": 16}),
            "image_max_side": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 32}),
            "thinking_mode": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "source_video": ("IMAGE",), "reference_video": ("IMAGE",),
            "reference_image_0": ("IMAGE",), "reference_image_1": ("IMAGE",), "reference_image_2": ("IMAGE",),
        }}

    def enhance(self, model, mmproj, task_type, prompt,
                temperature, repeat_penalty, seed,
                n_ctx, n_gpu_layers, max_tokens, video_frames, image_max_side,
                thinking_mode,
                source_video=None, reference_video=None,
                reference_image_0=None, reference_image_1=None, reference_image_2=None, **kw):
        raw = prompt.strip()
        _defaults = {
            "v2v":"enhance the video","mv2v":"enhance the video",
            "i2i":"enhance the image","i2v":"generate a video",
            "ads2v":"identify ad placement","vi2v":"edit with reference",
            "rv2v":"edit with reference","vrc2v":"edit with reference",
            "r2v":"generate a video","r2i":"generate an image",
        }
        if not raw:
            raw = _defaults.get(task_type, "describe and enhance")
            prompt = raw

        mp = _resolve(model)
        mm = _resolve(mmproj) if mmproj and mmproj != "<none>" else None
        if not os.path.isfile(mp): return (f"ERROR: {mp}", "")

        src = _sample(_t2p(source_video), video_frames)
        rvid = _sample(_t2p(reference_video), video_frames)
        ref = []
        ref_slots = []
        for idx, ri in enumerate((reference_image_0, reference_image_1, reference_image_2)):
            p = _t2p(ri)
            if p:
                ref.append(p[0])
                ref_slots.append(idx)

        sp, ut, order = _route(task_type, prompt, len(ref))
        labels = None
        imgs_src  = [_b64(p, image_max_side) for p in src]
        imgs_ref  = [_b64(p, image_max_side) for p in ref]
        imgs_rvid = [_b64(p, image_max_side) for p in rvid]
        imgs = {"none": [], "video": imgs_src + imgs_rvid,
                "ref": imgs_ref + imgs_rvid,
                "video+ref": imgs_src + imgs_ref + imgs_rvid,
                "ref_or_first": (imgs_ref or imgs_src[:1]) + imgs_rvid}[order]
        if order == "video+ref":
            labels = [f"Source-{i}" for i in range(len(imgs_src))] + \
                     [f"Ref-{ref_slots[i]}"   for i in range(len(imgs_ref))]  + \
                     [f"RefVid-{i}" for i in range(len(imgs_rvid))]
        msgs = _build_msgs(sp, ut, imgs, labels)

        v0 = (torch.cuda.memory_allocated() / 1048576) if torch.cuda.is_available() else 0
        t0 = time.time()
        mllm = _MLLM(mp, mm, n_ctx, n_gpu_layers, seed, thinking_mode)
        try:
            result = mllm.chat(msgs, mt=max_tokens, temp=temperature, rp=repeat_penalty)
            if result is None:
                print("[Qwen35PE] retrying...", flush=True)
                mllm._kill()
                result = mllm.chat(msgs, mt=max_tokens, temp=temperature, rp=repeat_penalty)
        finally:
            mllm.unload()
        v1 = (torch.cuda.memory_allocated() / 1048576) if torch.cuda.is_available() else 0
        if result is None:
            print("[Qwen35PE] ERROR: worker returned None", flush=True)
        print(f"[Qwen35PE] {task_type} {time.time()-t0:.1f}s VRAM {v0:.0f}->{v1:.0f}MB", flush=True)

        # Split structured output
        result = result or ""
        plan, desc = "", result
        m = re.search(r'(?:^|\n)(?:\d+\.\s*)?\[?\*{0,2}\s*FINAL[\s_]PROM\w+T\]?\s*:?\*{0,2}', result, re.IGNORECASE)
        if m:
            plan = result[:m.start()].strip()
            desc = result[m.end():].strip()
            desc = re.sub(r'^\s*assistant\s*\n*', '', desc, flags=re.IGNORECASE).strip()
        else:
            desc = result
            plan = "[Direct]"
        return (desc or "", plan or "")


NODE_CLASS_MAPPINGS = {"Qwen35PromptEnhancer": Qwen35PromptEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"Qwen35PromptEnhancer": "Qwen3.5 Prompt Enhancer (GGUF)"}

"""
ComfyUI-Bernini-PromptEnhancer — GGUF MLLM prompt enhancer.
Subprocess isolation, VRAM=0 after every generation. Raw pass-through output.
"""

import base64, gc, json, os, re, subprocess, sys, threading, time
from io import BytesIO
from typing import Dict, List, Optional

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

def _scan_models():
    """Rescan model directories — called on every INPUT_TYPES so new GGUF files appear on refresh."""
    models: Dict[str, str] = {}
    for root in _MODEL_ROOTS:
        if not os.path.isdir(root): continue
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.endswith(".gguf"):
                    rel = os.path.relpath(dp, root)
                    if rel == ".": rel = ""
                    display = f"{rel}\\{fn}" if rel else fn
                    models[display] = os.path.join(dp, fn)
    return models

def _list_models(mp):
    models = _scan_models()
    return sorted(k for k in models if ("mmproj" in k.lower()) == mp)

def _resolve(d):
    if os.path.isfile(d):
        return d
    return _scan_models().get(d, d)


class _MLLM:
    _W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker", "bernini_pe_worker.py")

    def __init__(self, mp, mm, ctx, gpu, seed):
        self.mp = mp; self.mm = mm or ""; self.ctx = ctx; self.gpu = gpu; self.seed = seed; self.p = None

    def _ping(self):
        """Ping worker with timeout. Returns True if alive, False if unresponsive."""
        try:
            self.p.stdin.write('{"cmd":"ping"}\n'); self.p.stdin.flush()
            box = {}
            def r(): 
                try: box["l"] = self.p.stdout.readline()
                except: box["l"] = ""
            t = threading.Thread(target=r, daemon=True); t.start(); t.join(5)
            if t.is_alive(): return False
            return bool(box.get("l") and json.loads(box["l"]).get("ok"))
        except:
            return False

    def _ensure(self):
        if self.p and self.p.poll() is None: return
        self.p = subprocess.Popen([sys.executable, "-u", self._W], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=sys.stderr, text=True, encoding="utf-8",
            errors="replace", bufsize=1, env={**os.environ})
        print(f"[BerniniPE] worker pid={self.p.pid}", flush=True)

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
            err = resp.get("error", "unknown worker error")
            print(f"[BerniniPE] worker error: {err}", flush=True)
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
            max_tokens=mt, temperature=temp, top_p=0.9, repeat_penalty=rp)})

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


def _route(tt, prompt, nr):
    n = max(nr, 1); ds = "You are a semantic planner and prompt enhancer. Reply in natural language text only. Never output bounding boxes, coordinates, <loc_*> tokens, or HTML tags."
    r = {
        "t2v":   (T2V, prompt, "none"),
        "t2i":   (T2I, prompt, "none"),
        "v2v":   (ds, V2V.format(user_prompt=prompt), "video"),
        "mv2v":  (ds, V2V.format(user_prompt=prompt), "video"),
        "i2i":   (ds, I2I.format(user_prompt=prompt), "ref"),
        "i2v":   (I2V_SP, I2V.format(user_prompt=prompt, image_num=n), "ref_or_first"),
        "ads2v": (ds, ADS2V.format(user_prompt=prompt), "video"),
        "vi2v":  (ds, VI2V.format(user_prompt=prompt, image_num=n), "video+ref"),
        "r2v":   (ds, R2V.format(image_num=n, original_text=prompt), "ref"),
        "r2i":   (ds, R2I.format(image_num=n, original_text=prompt), "ref"),
        "rv2v":  (ds, VR2V.format(image_num=n, image_num_1=n-1, original_text=prompt), "video+ref"),
        "vrc2v": (ds, VR2V.format(image_num=n, image_num_1=n-1, original_text=prompt), "video+ref"),
    }
    sp, ut, order = r.get(tt, (ds, prompt, "none"))
    # For visual tasks: append structured plan request with delimiter.
    # Skip tasks that already have [FINAL_PROMPT] built into their template.
    if order != "none" and tt not in ("r2v", "r2i", "rv2v", "vrc2v", "vi2v", "ads2v", "i2v"):
        ut += PLAN_SUFFIX
    return sp, ut, order


def _route_official(tt, prompt, nr):
    """Official Bernini templates — single paragraph output, no structured plan."""
    n = max(nr, 1); ds = "You are a semantic planner and prompt enhancer. Reply in natural language text only. Never output bounding boxes, coordinates, <loc_*> tokens, or HTML tags."
    r = {
        "t2v":   (OT2V, prompt, "none"),
        "t2i":   (OT2I, prompt, "none"),
        "v2v":   (ds, OV2V.format(user_prompt=prompt), "video"),
        "mv2v":  (ds, OV2V.format(user_prompt=prompt), "video"),
        "i2i":   (ds, OI2I.format(user_prompt=prompt), "ref"),
        "i2v":   (ds, OI2V.format(user_prompt=prompt, image_num=n), "ref_or_first"),
        "ads2v": (ds, OADS2V.format(user_prompt=prompt), "video"),
        "vi2v":  (ds, OVI2V.format(user_prompt=prompt, image_num=n), "video+ref"),
        "r2v":   (ds, OR2V.format(image_num=n, original_text=prompt), "ref"),
        "r2i":   (ds, OR2I.format(image_num=n, original_text=prompt), "ref"),
        "rv2v":  (ds, OVR2V.format(image_num=n, original_text=prompt), "video+ref"),
        "vrc2v": (ds, OVR2V.format(image_num=n, original_text=prompt), "video+ref"),
    }
    return r.get(tt, (ds, prompt, "none"))


class BerniniMLLMPromptEnhancer:
    CATEGORY = "Bernini"
    FUNCTION = "enhance"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "structured_plan")
    TASKS = ["t2v", "t2i", "v2v", "mv2v", "i2i", "i2v", "ads2v", "vi2v", "r2v", "r2i", "rv2v", "vrc2v"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (_list_models(False) or ["<no .gguf>"],),
            "mmproj": (["<none>"] + _list_models(True),),
            "task_type": (cls.TASKS, {"default": "v2v"}),
            "template_mode": (["official", "structured"], {"default": "structured"}),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
            "repeat_penalty": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            "n_ctx": ("INT", {"default": 8192, "min": 2048, "max": 32768}),
            "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200}),
            "max_tokens": ("INT", {"default": 4096, "min": 128, "max": 16384}),
            "video_frames": ("INT", {"default": 3, "min": 1, "max": 16}),
            "image_max_side": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 32}),
        }, "optional": {
            "source_video": ("IMAGE",), "reference_video": ("IMAGE",),
            "reference_image_0": ("IMAGE",), "reference_image_1": ("IMAGE",), "reference_image_2": ("IMAGE",),
        }}

    def enhance(self, model, mmproj, task_type, template_mode, prompt,
                temperature, repeat_penalty, seed,
                n_ctx, n_gpu_layers, max_tokens, video_frames, image_max_side,
                source_video=None, reference_video=None,
                reference_image_0=None, reference_image_1=None, reference_image_2=None, **kw):
        raw = prompt.strip()
        # Default instructions when user leaves prompt empty
        _defaults = {
            "v2v": "enhance the video", "mv2v": "enhance the video",
            "i2i": "enhance the image", "i2v": "generate a video",
            "ads2v": "identify ad placement", "vi2v": "edit with reference",
            "rv2v": "edit with reference", "vrc2v": "edit with reference",
            "r2v": "generate a video", "r2i": "generate an image",
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

        sp, ut, order = (_route_official if template_mode == "official" else _route)(task_type, prompt, len(ref))
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
        mllm = _MLLM(mp, mm, n_ctx, n_gpu_layers, seed)
        try:
            result = mllm.chat(msgs, mt=max_tokens, temp=temperature, rp=repeat_penalty)
            if result is None:
                print(f"[BerniniPE] retrying with fresh worker...", flush=True)
                mllm._kill()
                result = mllm.chat(msgs, mt=max_tokens, temp=temperature, rp=repeat_penalty)
        finally:
            mllm.unload()
        v1 = (torch.cuda.memory_allocated() / 1048576) if torch.cuda.is_available() else 0
        if result is None:
            print(f"[BerniniPE] ERROR: worker returned None — check stderr above for traceback", flush=True)
            print(f"[BerniniPE]   model={mp} mmproj={mm}", flush=True)
        print(f"[BerniniPE] {task_type} {time.time()-t0:.1f}s VRAM {v0:.0f}->{v1:.0f}MB", flush=True)

        if template_mode == "official":
            return (result or "", result or "")

        # Split output: everything before FINAL marker = plan, after = prompt
        result = result or ""
        plan, desc = "", result
        # match header like [FINAL_PROMPT], **FINAL_PROMPT**:, FINAL PROMPT: (same line or new line)
        m = re.search(r'(?:^|\n)(?:\d+\.\s*)?\[?\*{0,2}\s*FINAL[\s_]PROM\w+T\]?\s*:?\*{0,2}', result, re.IGNORECASE)
        if m:
            plan = result[:m.start()].strip()
            desc = result[m.end():].strip()
            # clean up trailing "assistant" noise
            desc = re.sub(r'^\s*assistant\s*\n*', '', desc, flags=re.IGNORECASE).strip()
        else:
            # No structured format — model wrote direct output
            desc = result
            plan = "[Direct]"
        return (desc or "", plan or "")


# ═══════════════════════════════════════════════════════════════════════════════
# MLLM as SEMANTIC PLANNER — observe, analyze, plan, then output.
# The model SEES source video/images → UNDERSTANDS the scene → PLANS changes.
# ═══════════════════════════════════════════════════════════════════════════════

T2V = """你是一位电影导演。根据用户的文本描述，从电影美学角度规划画面：
1. 确定时间(Day/Night/Dawn/Sunrise)、光源类型和方向、光线强度和色调
2. 确定镜头大小和拍摄角度、构图方式
3. 细化主体外貌、表情、姿态、数量
4. 规划运动过程（人物动作、背景动态如云飘风吹）
输出：一段60-200字的英文prompt，包含全部上述规划。只输出prompt。"""

T2I = """你是一位电影导演。根据用户的文本描述，从电影美学角度规划静态画面：
1. 确定时间、光源、光线强度、色调、镜头、构图
2. 细化主体外貌、表情、姿态
3. 注意：静态图像，不规划运动/摄像机运动，只规划静态状态
输出：一段60-200字的英文prompt。只输出prompt。"""

V2V = """You are a video editing planner. Source video frames provided.
User instruction: "{user_prompt}"
Reply in descriptive English text. Do NOT output coordinates or bounding boxes."""  # noqa: E501

I2I = """You are an image editing planner. Source image provided.
User instruction: "{user_prompt}"
Reply in descriptive English text. Do NOT output coordinates or bounding boxes."""  # noqa: E501

I2V = """{image_num} reference image(s) provided.
User intent: "{user_prompt}" """  # noqa: E501

I2V_SP = ("You are a video generation planner. Reply in natural language. "
    "Never output bounding boxes, coordinates, or HTML tags. "
    "Output using these EXACT sections: "
    "[OBSERVATION] — describe the reference image; "
    "[UNDERSTAND] — what the user intent means; "
    "[EXECUTE] — plan the video step by step; "
    "[PRESERVE] — key elements that stay consistent; "
    "[FINAL_PROMPT] — descriptive video paragraph (4-8 sentences, film-narrative style).")

VI2V = """Task: Video Editing with Reference Image. 3 source video frames + {image_num} reference image(s) provided.
User instruction: "{user_prompt}"

Determine the task type and output ONE sentence:
- propagation: "edit the video following the first frame."
- insertion: "Integrate the [object] from the reference image into the video."
- replacement: describe replacing an element with the reference subject.

Output ONLY the final sentence. Do NOT output coordinates or bounding boxes. Stop immediately."""  # noqa: E501

ADS2V = """Task: Ads Insertion. 3 source video frames provided.
User instruction: "{user_prompt}"

Output ONE concise English sentence like:
"Add Starbucks Latte wallpaper on the second floor across the street"

Do NOT output coordinates or bounding boxes. Stop immediately after the sentence."""  # noqa: E501

R2V = """You are a subject-driven video generation planner. {image_num} reference image(s) of subjects provided.

Original description: {original_text}

RULES:
- [OBSERVATION]: Describe each reference image in detail — subject appearance, clothing, features. Note ONLY what you actually see.
- [UNDERSTAND]: Explain what the original description means.
- [EXECUTE]: Plan a video featuring these subjects (reference as "the subject from image0"). Include scene, environment, temporal action sequence, lighting, camera.
- [PRESERVE]: Subject identity and appearance must match reference images exactly.
- [FINAL_PROMPT]: Write a descriptive video paragraph (4-8 sentences). Describe the final scene as if narrating a film — NOT instructions, NOT a checklist, NOT step-by-step execution. Only describe what the viewer sees.

[OBSERVATION]
[UNDERSTAND]
[EXECUTE]
[PRESERVE]
[FINAL_PROMPT]"""  # noqa: E501

R2I = """You are a subject-driven image generation planner. {image_num} reference image(s) of subjects provided.

Original description: {original_text}

RULES:
- [OBSERVATION]: Describe each reference image in detail — subject appearance, clothing, features. Note ONLY what you actually see.
- [UNDERSTAND]: Explain what the original description means.
- [EXECUTE]: Plan an image featuring these subjects (reference as "the subject from image0"). Include scene, environment, lighting, composition, framing.
- [PRESERVE]: Subject identity and appearance must match reference images exactly.
- [FINAL_PROMPT]: Write a descriptive video paragraph (4-8 sentences). Describe the final scene as if narrating a film — NOT instructions, NOT a checklist, NOT step-by-step execution. Only describe what the viewer sees.

[OBSERVATION]
[UNDERSTAND]
[EXECUTE]
[PRESERVE]
[FINAL_PROMPT]"""  # noqa: E501

VR2V = """You are a video editing planner. Source video frames and {image_num} reference image(s) provided.

User instruction: {original_text}

RULES:
- [OBSERVATION]: Describe the source video (scene, subjects, actions, lighting, weather, camera, mood). Then describe EACH reference image independently — use the Ref-X label shown with each image. Include what it shows, lighting, colors, atmosphere, style, any notable details. NEVER write "similar to Ref-X".
- [UNDERSTAND]: Explain what the instruction means based on your observations. What changes? What stays?
- [EXECUTE]: Plan the visual changes step by step. Be specific.
- [PRESERVE]: List everything that stays exactly the same.
- [FINAL_PROMPT]: Write a descriptive video paragraph (4-8 sentences). Describe the final scene as if narrating a film — NOT instructions, NOT a checklist, NOT step-by-step execution. Only describe what the viewer sees.

[OBSERVATION]
[UNDERSTAND]
[EXECUTE]
[PRESERVE]
[FINAL_PROMPT]"""  # noqa: E501



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
[FINAL_PROMPT]"""  # noqa: E501


# ═══════════════════════════════════════════════════════════════════════════════
# OFFICIAL TEMPLATES — verbatim from ByteDance Bernini prompt_enhancer.py
# ═══════════════════════════════════════════════════════════════════════════════

OT2V = """你是一位电影导演，旨在为用户输入的原始prompt添加电影元素，改写为优质（英文）Prompt，使其完整、具有表现力注意，输出必须是英文！
任务要求：
1. 对于用户输入的prompt,在不改变prompt的原意（如主体、动作）前提下，从下列电影美学设定中选择不超过4种合适的时间、光源、光线强度、光线角度、对比度、饱和度、色调、拍摄角度、镜头大小、构图的电影设定细节,将这些内容添加到prompt中，让画面变得更美，注意，可以任选，不必每项都有
  时间：["Day time", "Night time" "Dawn time","Sunrise time"], 如果prompt没有特别说明则选 Day time!!!
  光源：["Daylight", "Artificial lighting", "Moonlight", "Practical lighting", "Firelight","Fluorescent lighting", "Overcast lighting" "Sunny lighting"]
  光线强度：["Soft lighting", "Hard lighting"], 色调：["Warm colors","Cool colors", "Mixed colors"]
  光线角度：["Top lighting", "Side lighting", "Underlighting", "Edge lighting"]
  镜头尺寸：["Medium shot", "Medium close-up shot", "Wide shot","Medium wide shot","Close-up shot", "Extreme close-up shot", "Extreme wide shot"]
  拍摄角度：["Over-the-shoulder shot", "Low angle shot", "High angle shot","Dutch angle shot", "Aerial shot","Overhead shot"]
  构图：["Center composition","Balanced composition","Right-heavy composition","Left-heavy composition","Symmetrical composition","Short-side composition"]
2. 完善用户描述中出现的主体特征（如外貌、表情，数量、种族、姿态等），确保不要添加原始prompt中不存在的主体
3. 不要输出关于氛围、感觉等文学描写
4. 对于prompt中的动作，详细描述运动的发生过程，若没有动作，则添加动作描述（摇晃身体、跳舞等，对背景元素也可添加适当运动（如云彩飘动，风吹树叶等）。
5. 若原始prompt中没有风格，则不添加风格描述，若有风格描述，则将风格描述放于首位
6. 若prompt出现天空的描述，则改为湛蓝色的天空相关描述，避免曝光
7. 输出必须是全英文，改写后的prompt字数控制在60-200字左右, 不要输出类似"改写后prompt:"这样的输出"""  # noqa: E501
OT2I = OT2V.replace("动作", "静态状态")

OV2V = """Task: Video Editing. Source video frames provided. User instruction: "{user_prompt}"
Generate a detailed English editing prompt with modifications and preservations:
- Modifications: describe what changes — appearance, location, lighting, motion
- Preservations: describe what stays — camera, background, other objects
- Concretize vague references to specific named instances
Describe naturally. Output ONLY the final prompt. No explanations."""  # noqa: E501

OI2I = """Task: Image Editing. Source image provided. User instruction: "{user_prompt}"
Generate a detailed English editing prompt describing modifications and preservations. Output ONLY the final prompt. No explanations."""  # noqa: E501

OI2V = """Task: Image-to-Video. {image_num} reference image(s). User: "{user_prompt}"
Generate video description in English. Output ONLY final prompt."""  # noqa: E501

OVI2V = """Task: Video Editing with Reference Image. 3 source frames + {image_num} reference image(s). User: "{user_prompt}"
Determine type (propagation/insertion/replacement) and output ONE English prompt sentence."""  # noqa: E501

OADS2V = """Task: Ads Insertion. User: "{user_prompt}". Output concise one-sentence English ad insertion instruction."""  # noqa: E501

OR2V = """Subject-driven video. {image_num} reference image(s). Original: {original_text}
Write instruction + detailed "Generate a video where..." paragraph. Reference via "image0"/"image1".
Return: {{"rewritten_text":"..."}}"""  # noqa: E501

OR2I = """Subject-driven image. {image_num} reference image(s). Original: {original_text}
Write instruction + detailed "Generate an image where..." paragraph. Reference via "image0"/"image1".
Return: {{"rewritten_text":"..."}}"""  # noqa: E501

OVR2V = """Reference-image-guided video editing. Source frames + {image_num} ref images. Original: {original_text}
Write ONE English paragraph: editing instruction + target description.
Return: {{"rewritten_text":"..."}}"""  # noqa: E501


from .qwen35_node import Qwen35PromptEnhancer

NODE_CLASS_MAPPINGS = {
    "BerniniMLLMPromptEnhancer": BerniniMLLMPromptEnhancer,
    "Qwen35PromptEnhancer": Qwen35PromptEnhancer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BerniniMLLMPromptEnhancer": "Bernini MLLM Prompt Enhancer (GGUF)",
    "Qwen35PromptEnhancer": "Qwen3.5 Prompt Enhancer (GGUF)",
}

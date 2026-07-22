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

def _smart_sample(p, k):
    n = len(p)
    if n <= k: return p
    diffs = [0.0]
    arr = [np.asarray(im, np.float32) for im in p]
    for i in range(1, n):
        diffs.append(np.abs(arr[i] - arr[i-1]).mean())
    ranked = sorted(range(n), key=lambda i: diffs[i], reverse=True)
    picked = sorted(ranked[:k])
    return [p[i] for i in picked]

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
        "fl2v":  (ds, FL2V.format(original_text=prompt), "ref"),
    }
    sp, ut, order = r.get(tt, (ds, f'Instruction: "{prompt}". Generate an enhanced prompt.', "none"))
    if order != "none" and tt not in ("r2v", "r2i", "rv2v", "vrc2v", "vi2v", "ads2v", "i2v", "fl2v"):
        ut += PLAN_SUFFIX
    return sp, ut, order


# Short task templates (without built-in plan — PLAN_SUFFIX appended)
V2V = """Source video frames provided.

User instruction: "{user_prompt}"

Watch the source video. Identify specific physical details:
1. WHAT MOVES: subjects, objects, background elements in motion vs stationary
2. HOW IT MOVES: direction, speed, pattern of each moving element
3. POSITIONS: spatial arrangement of subjects and objects
4. CAMERA: movement type, speed, angle
5. TIMING: rhythm, pace of actions and transitions

Then enhance or edit based on the user's instruction. Add, modify, or emphasize specific elements while preserving the physical patterns.

IMPORTANT: Focus on physical actions and visual details. NO vague adjectives or mood descriptions.

RULES:
- [OBSERVATION]: Describe the source video — ONLY physical details: what is in frame, what moves, how it moves, camera work, positions, timing. NO style/color/mood.
- [UNDERSTAND]: What the user wants to change, add, or emphasize. Map the instruction to specific physical modifications.
- [EXECUTE]: Plan the enhanced video. Keep source's physical patterns. Apply user's changes as specific physical actions.
- [PRESERVE]: Key physical elements from source that must stay consistent. Changes from user instruction.
- [FINAL_PROMPT]: Descriptive video paragraph (4-8 sentences). Describe the enhanced video with specific physical actions and details. NO vague style or mood language.

[OBSERVATION]
[UNDERSTAND]
[EXECUTE]
[PRESERVE]
[FINAL_PROMPT]"""

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


# Imitation video templates
IM2V = """Imitation video frames provided.

User request: "{user_prompt}"

Watch the imitation video. Identify specific physical details:
1. WHAT MOVES / WHAT IS STILL: Which parts are in motion? Which stay fixed?
2. HOW IT MOVES: Direction, speed, pattern (continuous/rhythmic/intermittent)
3. POSITIONS & LAYOUT: Where is each subject in frame? Spatial arrangement
4. CAMERA: Is it moving? Static? Push/pull? Pan/tilt? Speed of any movement?
5. TIMING: Rhythm, pace, duration of motions. Frame-to-frame changes.

Then TRANSFER these exact physical patterns to the user's subjects.

IMPORTANT: Do NOT describe colors, lighting, mood, atmosphere, or "style". Describe only what PHYSICALLY happens — which parts move HOW, WHERE, and at WHAT pace.

EXAMPLES:
- Flowers sway left-right, heads move, stems still → cats sway left-right, heads move, bodies still
- Camera pushes in over 5s at constant speed → same push-in on new subjects
- Three subjects center-left, one right-center → same arrangement

RULES:
- [OBSERVATION]: Physically describe the imitation video. ONLY: what moves, how (direction/speed/pattern), what is stationary, where subjects are in frame, how camera moves. NO style/mood/color/lighting.
- [UNDERSTAND]: Map each physical pattern to the user's request. "Heads that sway left-right → the user's subjects' heads should sway left-right. Stalks that stay still → bodies stay still. Static camera → same."
- [EXECUTE]: Plan video applying extracted motions, positions, camera, timing to user's subjects. Same movements, same layout, same pace.
- [PRESERVE]: Movement patterns, spatial positions, camera behavior, timing from the imitation.
- [FINAL_PROMPT]: Video paragraph (4-8 sentences). Describe user's subjects with imitation's physical actions. Say WHAT moves, HOW, WHERE, at WHAT pace. NO style/mood.

[OBSERVATION]
[UNDERSTAND]
[EXECUTE]
[PRESERVE]
[FINAL_PROMPT]"""

RIM2V = """{image_num} reference image(s) and imitation video frames provided.

Original description: "{original_text}"

Reference images = WHO (subject identity, appearance). Imitation video = the physical template (actions, camera, layout, timing).

Watch the imitation video for what moves, how, where, camera, timing — then transfer these physical patterns onto the reference subjects.

IMPORTANT: Do NOT describe style/mood/color/lighting. Physical actions only.

RULES:
- [OBSERVATION]: Describe each ref image (Ref-X) — subject appearance only. Then physically describe imitation video: what moves, how, where, camera, timing. NO style/color/lighting/mood.
- [UNDERSTAND]: Map imitation's physical patterns to reference subjects. Concrete mapping per body part/object.
- [EXECUTE]: Reference subjects performing imitation's movements. Same positions, same camera, same timing.
- [PRESERVE]: Reference subject appearance/identity. Imitation's movement patterns, layout, camera, timing.
- [FINAL_PROMPT]: Video paragraph (4-8 sentences). Reference subjects with imitation's physical actions. WHAT moves HOW WHERE at WHAT pace. NO style/mood.

[OBSERVATION]
[UNDERSTAND]
[EXECUTE]
[PRESERVE]
[FINAL_PROMPT]"""

RVIM2V = """Reference video frames and imitation video frames provided.

Original description: "{original_text}"

Reference video = scene/subjects to keep. Imitation video = physical template (actions, camera, layout, timing).

Analyze imitation for what moves/how/where/camera/timing. Transfer to reference. Physical actions only. NO style/mood/color/lighting.

RULES:
- [OBSERVATION]: Describe reference video — scene, subjects, actions, positions, camera. Then physically describe imitation video: what moves, how, where, camera, timing. NO style.
- [UNDERSTAND]: Map imitation's physical patterns to reference scene. Concrete mapping.
- [EXECUTE]: Reference video's scene/subjects + imitation's movements, camera, layout, timing.
- [PRESERVE]: Reference subjects/identities. Imitation's movement patterns, positions, camera, timing.
- [FINAL_PROMPT]: Video paragraph (4-8 sentences). Reference scene with imitation's physical actions. WHAT moves HOW WHERE at WHAT pace. NO style.

[OBSERVATION]
[UNDERSTAND]
[EXECUTE]
[PRESERVE]
[FINAL_PROMPT]"""

VIIM2V = """Source video frames and imitation video frames provided.

User instruction: "{user_prompt}"

Source video = scene/subjects to keep. Imitation video = physical template (actions, camera, layout, timing).

Analyze imitation for what moves/how/where/camera/timing. Transfer to source. Physical actions only. NO style/mood/color/lighting.

RULES:
- [OBSERVATION]: Describe source video — scene, subjects, actions, positions, camera. Then physically describe imitation video: what moves, how, where, camera, timing. NO style.
- [UNDERSTAND]: Map imitation's physical patterns to source. Concrete mapping.
- [EXECUTE]: Source video's scene/subjects + imitation's movements, camera, layout, timing.
- [PRESERVE]: Source subjects/identities. Imitation's movement patterns, positions, camera, timing.
- [FINAL_PROMPT]: Video paragraph (4-8 sentences). Source scene with imitation's physical actions. WHAT moves HOW WHERE at WHAT pace. NO style.

[OBSERVATION]
[UNDERSTAND]
[EXECUTE]
[PRESERVE]
[FINAL_PROMPT]"""

TIM2V = """Imitation video frames provided.

User request: "{user_prompt}"

Watch the imitation video CLOSELY. Identify specific physical details:

1. WHAT MOVES: Which body parts or objects are in motion? Which are stationary?
2. HOW IT MOVES: Direction (left-right? up-down? circular?) Speed (fast? slow? rhythmic?) Pattern (continuous? intermittent? random?)
3. POSITIONS: Where is each subject in frame? Their spatial arrangement and distances.
4. CAMERA: Is the camera moving? How? Static? Pushing in/out? Panning? Tilting?
5. TIMING: What is the rhythm and pace of the motion? How long do movements last?

Then TRANSFER these specific physical patterns onto the user's request.

IMPORTANT — do NOT describe "style" or "mood". Describe what PHYSICALLY happens in the video, then physically apply it to new subjects.

EXAMPLES of correct content mapping:
- Flowers sway left-to-right, flower heads move but stems are still → user's subjects should sway left-to-right, their heads move but bodies stay still
- Camera slowly pushes in at constant speed over 5 seconds → same push-in speed and duration
- Three subjects clustered center-left, one offset right-center → same spatial arrangement
- Subjects bob up-down once every 2 seconds in a rhythmic cycle → same bobbing rhythm

RULES:
- [OBSERVATION]: Physically describe the imitation video. ONLY talk about what you SEE: which objects move, HOW they move (direction/speed/pattern), WHERE they are in frame, HOW the camera moves. DO NOT describe colors, lighting, mood, or atmosphere.
- [UNDERSTAND]: Map each physical pattern from the imitation onto the user's request. "The flower heads that sway left-right → the cats' heads should sway left-right. The stems that stay still → the cats' bodies stay still. The stationary camera → same stationary camera."
- [EXECUTE]: Plan the new video by applying the extracted physical patterns to the user's subjects. Same motions, same positions, same camera, same timing.
- [PRESERVE]: Movement patterns, positions, camera behavior and timing from the imitation.
- [FINAL_PROMPT]: Write a descriptive video paragraph (4-8 sentences). Describe the user's subjects with the imitation's exact physical actions. Say WHAT moves, HOW, WHERE, and at WHAT pace. DO NOT describe style or mood.

[OBSERVATION]
[UNDERSTAND]
[EXECUTE]
[PRESERVE]
[FINAL_PROMPT]"""

FL2V = """Two reference images — START and END frames.
Original description: "{original_text}"
START-FRAME = first. END-FRAME = last. Plan transition.

RULES:
- [OBSERVATION]: START-FRAME details, then END-FRAME. Note changes.
- [UNDERSTAND]: Transition from start to end.
- [EXECUTE]: Video starts at START-FRAME, ends at END-FRAME.
- [PRESERVE]: START-FRAME as beginning, END-FRAME as ending.
- [FINAL_PROMPT]: 4-8 sentences. Begins with START-FRAME scene, ends with END-FRAME scene.

[OBSERVATION]
[UNDERSTAND]
[EXECUTE]
[PRESERVE]
[FINAL_PROMPT]"""


class Qwen35PromptEnhancer:
    CATEGORY = "Bernini"
    FUNCTION = "enhance"
    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("enhanced_prompt", "structured_plan",
                    "source_video", "reference_video",
                    "reference_image_0", "reference_image_1", "reference_image_2")
    TASKS = ["t2v","t2i","v2v","mv2v","i2i","i2v","ads2v","vi2v","r2v","r2i","rv2v","vrc2v","fl2v"]

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
            "smart_frames": ("BOOLEAN", {"default": False}),
            "thinking_mode": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "source_video": ("IMAGE",), "reference_video": ("IMAGE",),
            "reference_image_0": ("IMAGE",), "reference_image_1": ("IMAGE",), "reference_image_2": ("IMAGE",),
        }}

    def enhance(self, model, mmproj, task_type, prompt,
                temperature, repeat_penalty, seed,
                n_ctx, n_gpu_layers, max_tokens, video_frames, image_max_side, smart_frames,
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
            "fl2v":"generate video from start to end frame",
        }
        if not raw:
            raw = _defaults.get(task_type, "describe and enhance")
            prompt = raw

        mp = _resolve(model)
        mm = _resolve(mmproj) if mmproj and mmproj != "<none>" else None
        if not os.path.isfile(mp): return (f"ERROR: {mp}", "")

        sf = _smart_sample if smart_frames else _sample
        src = sf(_t2p(source_video), video_frames)
        rvid = sf(_t2p(reference_video), video_frames)
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
        if order == "ref":
            imgs = imgs_ref
            if task_type == "fl2v" and len(imgs_ref) >= 2:
                labels = ["START-FRAME", "END-FRAME"]
            else:
                labels = [f"Ref-{ref_slots[i]}" for i in range(len(imgs_ref))]
        else:
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
        print(f"[Qwen35PE] {task_type} order={order} n_imgs={len(imgs)} prompt_len={len(prompt)}", flush=True)
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
        return (desc or "", plan or "",
                source_video, reference_video,
                reference_image_0, reference_image_1, reference_image_2)


NODE_CLASS_MAPPINGS = {"Qwen35PromptEnhancer": Qwen35PromptEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"Qwen35PromptEnhancer": "Qwen3.5 Prompt Enhancer (GGUF)"}

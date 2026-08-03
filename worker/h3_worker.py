"""
H3 prompt planner worker — isolated llama.cpp subprocess for Gemma 4 / any GGUF.
Kills on exit -> OS frees 100% VRAM. Protocol: stdin/stdout JSON lines.
"""

import json, re, sys, time


def _log(msg):
    print(f"[h3-worker] {msg}", file=sys.stderr, flush=True)


def _respond(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


AUDIO_MAGIC = b"\x00H3AUDIO\x00"


class Gemma4AudioChatHandler:
    """MTMDChatHandler subclass that also processes input_audio content parts.

    Uses the SAME C bindings (mtmd_bitmap_init_from_audio / AUDIO chunk type) that
    llama-cpp-python 0.3.34 already ships, so NO llama-cpp upgrade is required.
    Audio is sent by the node as {"type":"input_audio","input_audio":{"data":<b64 wav>,"format":"wav"}}.
    """

    @classmethod
    def _make(cls, clip_model_path, verbose=False, use_gpu=True):
        from llama_cpp.llama_chat_format import MTMDChatHandler

        class _Gemma4AudioImpl(MTMDChatHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)

            def get_image_urls(self, messages):
                urls = []
                for message in messages:
                    if message["role"] == "user":
                        content = message["content"]
                        if isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict):
                                    if part.get("type") == "image_url":
                                        u = part["image_url"]
                                        urls.append(u["url"] if isinstance(u, dict) else u)
                                    elif part.get("type") == "input_audio":
                                        ia = part["input_audio"]
                                        data = ia.get("data", "") if isinstance(ia, dict) else ""
                                        urls.append("audio://" + data)
                return urls

            @staticmethod
            def _convert_content_part_for_template(part, media_marker):
                if isinstance(part, dict) and part.get("type") in ("image_url", "input_audio"):
                    return {"type": "text", "text": media_marker}
                return part

            def load_image(self, image_url):
                if isinstance(image_url, str) and image_url.startswith("audio://"):
                    import base64
                    return AUDIO_MAGIC + base64.b64decode(image_url[len("audio://"):])
                return super().load_image(image_url)

            def _create_bitmap_from_bytes(self, data):
                if isinstance(data, bytes) and data.startswith(AUDIO_MAGIC):
                    import ctypes as C
                    import io
                    import numpy as np
                    import soundfile as sf
                    wav_bytes = data[len(AUDIO_MAGIC):]
                    buf = io.BytesIO(wav_bytes)
                    sr = sf.info(buf).samplerate
                    buf.seek(0)
                    samples, _ = sf.read(buf, dtype='float32')
                    if samples.ndim > 1:
                        samples = samples.mean(axis=1)
                    target_sr = self._mtmd_cpp.mtmd_get_audio_sample_rate(self.mtmd_ctx)
                    if sr != target_sr and len(samples) > 0:
                        n_out = max(1, int(len(samples) * target_sr / sr))
                        x_old = np.linspace(0, 1, len(samples), endpoint=False)
                        x_new = np.linspace(0, 1, n_out, endpoint=False)
                        samples = np.interp(x_new, x_old, samples).astype(np.float32)
                    arr = (C.c_float * len(samples))(*samples.tolist())
                    bitmap = self._mtmd_cpp.mtmd_bitmap_init_from_audio(len(samples), arr)
                    if bitmap is None:
                        raise ValueError("Failed to create audio bitmap")
                    return bitmap
                return super()._create_bitmap_from_bytes(data)

        return _Gemma4AudioImpl(clip_model_path, verbose=verbose, use_gpu=use_gpu)


def _load(payload):
    from llama_cpp import Llama
    kw = dict(
        model_path=payload["model_path"], n_ctx=payload["n_ctx"],
        n_gpu_layers=payload["n_gpu_layers"], n_threads=payload["n_threads"],
        use_mmap=payload.get("use_mmap", True),
        use_mlock=payload.get("use_mlock", False),
        seed=payload["seed"], verbose=payload.get("verbose", False),
    )
    mm = payload.get("mmproj_path")
    if mm:
        # Gemma 4 first (audio+vision mmproj) via audio-capable handler, then fallbacks.
        for handler_path in [
            "Gemma4AudioChatHandler",
            "llama_cpp.llama_chat_format.Gemma4ChatHandler",
            "llama_cpp.llama_chat_format.Gemma3nChatHandler",
            "llama_cpp.llama_chat_format.Qwen3VLHandler",
            "llama_cpp.llama_chat_format.Qwen25VLChatHandler",
        ]:
            try:
                if handler_path == "Gemma4AudioChatHandler":
                    handler_cls = Gemma4AudioChatHandler._make
                else:
                    mod, cls = handler_path.rsplit(".", 1)
                    handler_cls = __import__(mod, fromlist=[cls]).__dict__[cls]
                kw["chat_handler"] = handler_cls(clip_model_path=mm)
                _log(f"using {handler_path.rsplit('.',1)[-1]}")
                break
            except Exception as e:
                _log(f"{handler_path.rsplit('.',1)[-1]} failed: {e}")
    return Llama(**kw)


def _strip_input_audio(msgs):
    """Remove input_audio parts for handlers that cannot process audio."""
    out = []
    for m in msgs:
        content = m.get("content")
        if isinstance(content, list):
            content = [p for p in content if not (isinstance(p, dict) and p.get("type") == "input_audio")]
            if not content:
                continue
            m = {**m, "content": content}
        out.append(m)
    return out


def _chat(llm, payload):
    msgs = payload["messages"]
    if not payload.get("enable_thinking", False):
        # Merge the no-thinking directive into the existing system message when
        # present — prepending a second system message breaks Gemma templates
        # ("System message must be at the beginning").
        if msgs and msgs[0].get("role") == "system":
            first = msgs[0]
            if isinstance(first.get("content"), str):
                first["content"] += "\nRespond directly without thinking or reasoning tags."
            else:
                msgs = [{"role": "system", "content": "Respond directly without thinking or reasoning tags."}] + msgs
        else:
            msgs = [{"role": "system", "content": "Respond directly without thinking or reasoning tags."}] + msgs
    handler = getattr(llm, "chat_handler", None)
    handler_name = type(handler).__name__ if handler is not None else ""
    if "Gemma4AudioImpl" not in handler_name:
        msgs = _strip_input_audio(msgs)
    n_img = sum(1 for m in msgs if isinstance(m.get("content"), list)
                for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url")
    _log(f"handler={handler_name} images={n_img} audio_parts={sum(1 for m in msgs if isinstance(m.get('content'), list) for p in m['content'] if isinstance(p, dict) and p.get('type') == 'input_audio')}")
    resp = llm.create_chat_completion(
        messages=msgs,
        max_tokens=payload["max_tokens"],
        temperature=payload["temperature"],
        top_p=payload["top_p"],
        repeat_penalty=payload["repeat_penalty"],
    )
    text = resp["choices"][0]["message"]["content"] or ""
    text = text.lstrip()
    while text.startswith("assistant"):
        text = text[len("assistant"):].lstrip("\n\r ").lstrip()
    # Strip <think>...</think> blocks that leak through
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    return text


def main():
    llm = None
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try: req = json.loads(line)
        except: continue

        cmd = req.get("cmd")
        if cmd == "ping":
            _respond({"ok": True, "text": "pong"})
            continue
        if cmd == "exit":
            _respond({"ok": True, "text": "bye"})
            break
        if cmd != "run":
            _respond({"ok": False, "error": f"unknown: {cmd}"})
            continue

        try:
            p = req["payload"]
            if llm is None:
                t0 = time.time()
                llm = _load(p)
                _log(f"loaded in {time.time()-t0:.1f}s")
            text = _chat(llm, p)
            _respond({"ok": True, "text": text})
        except Exception as e:
            _respond({"ok": False, "error": str(e)})


if __name__ == "__main__":
    main()

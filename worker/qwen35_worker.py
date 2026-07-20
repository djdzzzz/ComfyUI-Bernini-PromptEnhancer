"""
Qwen3.5-VL worker — isolated llama.cpp subprocess.
Kills on exit -> OS frees 100% VRAM. Protocol: stdin/stdout JSON lines.
"""

import json, re, sys, time


def _log(msg):
    print(f"[qwen35-worker] {msg}", file=sys.stderr, flush=True)


def _respond(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _load(payload):
    from llama_cpp import Llama
    kw = dict(
        model_path=payload["model_path"], n_ctx=payload["n_ctx"],
        n_gpu_layers=payload["n_gpu_layers"], n_threads=payload["n_threads"],
        use_mmap=payload.get("use_mmap", True),
        use_mlock=payload.get("use_mlock", False),
        seed=payload["seed"], verbose=payload.get("verbose", False),
    )
    if payload.get("mmproj_path"):
        handlers_ok = False
        for handler_path in [
            "llama_cpp.llama_chat_format.Qwen3VLHandler",
            "llama_cpp.llama_chat_format.Qwen25VLChatHandler",
        ]:
            try:
                mod, cls = handler_path.rsplit(".", 1)
                handler_cls = __import__(mod, fromlist=[cls]).__dict__[cls]
                kw["chat_handler"] = handler_cls(clip_model_path=payload["mmproj_path"])
                _log(f"using {cls}")
                handlers_ok = True
                break
            except Exception as e:
                _log(f"{cls} failed: {e}")
        if not handlers_ok:
            raise RuntimeError("No Qwen3.5/VL handler available. Upgrade llama-cpp-python or use without mmproj.")
    return Llama(**kw)


def _chat(llm, payload):
    msgs = payload["messages"]
    if not payload.get("enable_thinking", False):
        msgs = [{"role": "system", "content": "Respond directly without thinking or reasoning tags."}] + msgs
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

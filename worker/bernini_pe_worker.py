"""Bernini PE worker — isolated llama.cpp subprocess.
Kills on exit → OS frees 100% VRAM. Protocol: stdin/stdout JSON lines.
"""

import json, sys, time


def _log(msg):
    print(f"[worker] {msg}", file=sys.stderr, flush=True)


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
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        kw["chat_handler"] = Qwen25VLChatHandler(clip_model_path=payload["mmproj_path"])
    return Llama(**kw)


def _chat(llm, payload):
    resp = llm.create_chat_completion(
        messages=payload["messages"],
        max_tokens=payload["max_tokens"],
        temperature=payload["temperature"],
        top_p=payload["top_p"],
        repeat_penalty=payload["repeat_penalty"],
    )
    text = resp["choices"][0]["message"]["content"] or ""
    # Strip leading "assistant\n" blocks — chat handler format leak
    text = text.lstrip()
    while text.startswith("assistant"):
        text = text[len("assistant"):].lstrip("\n\r ").lstrip()
    return text


def main():
    llm = None
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try: req = json.loads(line)
        except: continue

        cmd = req.get("cmd")
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

import { app } from "../../scripts/app.js";

// ---------------------------------------------------------------------------
// H3PromptEnhancer — MiniMax H3 panel with built-in media upload.
// Materials are uploaded to ComfyUI's input directory (POST /upload/image),
// tracked in the hidden `media_manifest` widget (JSON), and passed through to
// compact media_0..N outputs that sync with the manifest.
// ---------------------------------------------------------------------------

const TASK_LABELS = {
    "t2v": "文生视频 t2v",
    "i2v": "图生视频 i2v",
    "h3_multi_ref": "全能参考 multi-ref",
};

const PANEL_WIDGETS = [
    "model", "mmproj", "task_type", "prompt", "media_manifest", "system_prompt",
    "user_template",
    "temperature", "repeat_penalty", "seed", "n_ctx", "n_gpu_layers",
    "max_tokens", "frame_mode", "video_frames", "sample_fps", "sample_seconds",
    "image_max_side", "thinking_mode",
];

const MAX_IMGS = 9, MAX_VIDS = 3, MAX_AUDS = 3;

const H3_CSS = `
.h3p { display:flex; flex-direction:column; gap:8px; width:100%; height:100%;
       box-sizing:border-box; background:#191919; border:1px solid #2b2b2b;
       border-radius:10px; padding:10px; font-family:inherit;
       color:#d5d5d5; font-size:12px; overflow:auto; min-width:0; }
.h3p * { box-sizing:border-box; min-width:0; }
.h3p-top { display:flex; gap:10px; flex:1 1 auto; min-height:110px; width:100%; min-width:0; }
.h3p-media { width:112px; flex:0 0 112px; border:1px dashed #3a3a3a; border-radius:8px;
             background:#202020; overflow-y:auto; overflow-x:hidden;
             padding:4px; scrollbar-width:thin; scrollbar-color:#3a3a3a transparent; }
.h3p-media.h3p-drag { border-color:#9ed94a; background:#22291a; }
.h3p-addbtn { display:flex; flex-direction:column; align-items:center; justify-content:center;
              gap:3px; min-height:52px; color:#8a8a8a; cursor:pointer; border-radius:6px; }
.h3p-addbtn:hover { background:#262626; color:#b5b5b5; }
.h3p-addbtn .h3p-plus { font-size:20px; line-height:1; }
.h3p-addbtn .h3p-mtitle { font-size:11px; }
.h3p-chip { position:relative; margin-bottom:4px; border-radius:6px; background:#2a2a2a;
            height:44px; display:flex; align-items:center; justify-content:center;
            overflow:hidden; }
.h3p-chip img { width:100%; height:100%; object-fit:cover; display:block; }
.h3p-chip .h3p-cname { font-size:10px; color:#c8c8c8; padding:0 16px 0 6px;
                       white-space:nowrap; overflow:hidden; text-overflow:ellipsis; width:100%; }
.h3p-chip .h3p-cicon { font-size:18px; }
.h3p-chip .h3p-x { position:absolute; top:1px; right:3px; width:14px; height:14px;
                   border-radius:50%; background:rgba(0,0,0,.65); color:#ddd;
                   font-size:10px; line-height:14px; text-align:center; cursor:pointer; }
.h3p-chip .h3p-x:hover { background:#a33; }
.h3p-chip .h3p-ctag { position:absolute; left:2px; bottom:1px; font-size:9px; color:#9ed94a;
                      background:rgba(0,0,0,.6); padding:0 3px; border-radius:3px; }
.h3p-prompt { flex:1 1 auto; resize:none; background:transparent; border:none; outline:none;
              color:#e8e8e8; font-size:13px; line-height:1.5; font-family:inherit;
              min-width:0; min-height:60px; }
.h3p-prompt::placeholder { color:#5f5f5f; }
.h3p-bar { display:flex; align-items:center; gap:6px; flex:0 0 auto; flex-wrap:wrap; }
.h3p-sel { background:#242424; color:#cfcfcf; border:1px solid #303030; border-radius:999px;
           padding:4px 8px; font-size:11px; outline:none; cursor:pointer; }
.h3p-sel:hover { background:#2d2d2d; }
.h3p-btn { background:#242424; color:#cfcfcf; border:1px solid #303030; border-radius:999px;
           padding:4px 10px; font-size:11px; cursor:pointer; }
.h3p-btn:hover { background:#2d2d2d; }
.h3p-btn.h3p-on { background:#33401f; border-color:#5b7a24; color:#c9e784; }
.h3p-count { margin-left:auto; color:#7d7d7d; font-size:11px; user-select:none; }
.h3p-adv { display:none; flex:0 0 auto; background:#1f1f1f; border:1px solid #2b2b2b;
           border-radius:8px; padding:8px; overflow-y:auto; max-height:300px; min-width:0;
           scrollbar-width:thin; scrollbar-color:#3a3a3a transparent; }
.h3p-adv.h3p-open { display:grid; grid-template-columns:1fr 1fr; gap:6px 14px; }
.h3p-row { display:flex; align-items:center; gap:6px; min-width:0; }
.h3p-row label { flex:0 0 92px; color:#909090; font-size:11px; white-space:nowrap;
                 overflow:hidden; text-overflow:ellipsis; }
.h3p-row input[type=number], .h3p-row select { flex:1 1 auto; min-width:0; background:#262626;
                 border:1px solid #333; border-radius:6px; color:#d5d5d5;
                 padding:3px 6px; font-size:11px; outline:none; }
.h3p-row input[type=checkbox] { accent-color:#9ed94a; }
/* @ mention menu */
.h3p-mention { position:absolute; z-index:9999; background:#222; border:1px solid #3a3a3a;
               border-radius:8px; min-width:180px; max-height:220px; overflow-y:auto;
               box-shadow:0 6px 24px rgba(0,0,0,.5); padding:4px; }
.h3p-mention .h3p-mhead { font-size:10px; color:#888; padding:3px 8px; }
.h3p-mention .h3p-mitem { display:flex; align-items:center; gap:8px; padding:5px 8px;
                          border-radius:6px; cursor:pointer; font-size:12px; color:#ddd; }
.h3p-mention .h3p-mitem:hover, .h3p-mention .h3p-mitem.h3p-sel { background:#33401f; color:#e8f5c8; }
.h3p-mention .h3p-mitem img { width:28px; height:28px; object-fit:cover; border-radius:4px; }
.h3p-mention .h3p-mitem .h3p-micon { width:28px; height:28px; display:flex; align-items:center;
                                     justify-content:center; background:#2c2c2c; border-radius:4px; font-size:14px; }
/* audio trim dialog */
.h3p-overlay { position:fixed; inset:0; z-index:10000; background:rgba(0,0,0,.55);
               display:flex; align-items:center; justify-content:center; }
.h3p-dialog { background:#1d1d1d; border:1px solid #333; border-radius:12px; padding:16px;
              width:380px; color:#ddd; font-size:13px; }
.h3p-dialog h4 { margin:0 0 12px; font-size:14px; color:#eee; font-weight:600; }
.h3p-dialog .h3p-trow { display:flex; align-items:center; gap:8px; margin:8px 0; }
.h3p-dialog .h3p-trow label { flex:0 0 52px; color:#999; font-size:12px; }
.h3p-dialog input[type=number] { flex:1; background:#262626; border:1px solid #333; border-radius:6px;
                                 color:#ddd; padding:5px 8px; font-size:12px; outline:none; }
.h3p-dialog .h3p-tinfo { font-size:12px; color:#9ed94a; margin:6px 0; }
.h3p-dialog .h3p-tinfo.h3p-terr { color:#ff7a7a; }
.h3p-dialog .h3p-tbtns { display:flex; gap:8px; justify-content:flex-end; margin-top:14px; }
.h3p-dialog button { background:#242424; color:#cfcfcf; border:1px solid #333; border-radius:8px;
                     padding:6px 16px; font-size:12px; cursor:pointer; }
.h3p-dialog button.h3p-primary { background:#9ed94a; color:#161616; border-color:#9ed94a; font-weight:600; }
.h3p-dialog button:hover { filter:brightness(1.15); }
.h3p-chip .h3p-trim { position:absolute; top:1px; left:3px; width:14px; height:14px;
                      border-radius:50%; background:rgba(0,0,0,.65); font-size:9px; line-height:14px;
                      text-align:center; cursor:pointer; }
.h3p-chip .h3p-trim:hover { background:#33401f; }
.h3p-chip .h3p-trimmed { position:absolute; right:18px; top:1px; font-size:9px; color:#9ed94a;
                         background:rgba(0,0,0,.6); padding:0 3px; border-radius:3px; }
/* video preview player */
.h3p-player { position:fixed; z-index:10001; background:#000; border:1px solid #3a3a3a;
              border-radius:10px; box-shadow:0 10px 40px rgba(0,0,0,.7); overflow:hidden; }
.h3p-player video { display:block; max-width:520px; max-height:70vh; background:#000; }
.h3p-player .h3p-pclose { position:absolute; top:6px; right:6px; width:24px; height:24px;
                          border-radius:50%; background:rgba(0,0,0,.7); color:#fff; border:none;
                          font-size:13px; cursor:pointer; z-index:2; }
.h3p-player .h3p-pclose:hover { background:#a33; }
.h3p-chip .h3p-play { position:absolute; inset:0; display:flex; align-items:center;
                      justify-content:center; background:rgba(0,0,0,.25); color:#fff;
                      font-size:16px; cursor:pointer; opacity:0; transition:opacity .15s; }
.h3p-chip:hover .h3p-play { opacity:1; }
`;

let cssInjected = false;
function injectCss() {
    if (cssInjected) return;
    cssInjected = true;
    const st = document.createElement("style");
    st.textContent = H3_CSS;
    document.head.appendChild(st);
}

function hideWidget(w) {
    if (!w) return;
    w.hidden = true;
    w.type = "hidden";
    w.computeSize = () => [0, -4];
}

// ---------------------------------------------------------------- media state
function emptyMedia() {
    return { source_video: null, reference_videos: [], images: [], audios: [],
             vid_nums: [], img_nums: [], aud_nums: [] };
}
function getMedia(node) {
    const w = node._h3W?.media_manifest;
    if (!w || !w.value) return emptyMedia();
    try {
        const d = JSON.parse(w.value);
        return {
            source_video: d.source_video || null,
            reference_videos: Array.isArray(d.reference_videos) ? d.reference_videos : [],
            images: Array.isArray(d.images) ? d.images : [],
            audios: Array.isArray(d.audios) ? d.audios : [],
            vid_nums: Array.isArray(d.vid_nums) ? d.vid_nums : [],
            img_nums: Array.isArray(d.img_nums) ? d.img_nums : [],
            aud_nums: Array.isArray(d.aud_nums) ? d.aud_nums : [],
        };
    } catch { return emptyMedia(); }
}
function setMedia(node, m) {
    node._h3W.media_manifest.value = JSON.stringify(m);
}
function entryName(e) { return typeof e === "string" ? e : (e?.name || ""); }
function mediaCount(m) {
    return (m.source_video ? 1 : 0) + m.reference_videos.length + m.images.length + m.audios.length;
}
// smallest positive int not in arr (1-based numbering)
function nextNum(arr) {
    let n = 1;
    const s = new Set(arr);
    while (s.has(n)) n++;
    return n;
}

// ---------------------------------------------------------------- uploads
const IMG_EXT = /\.(png|jpe?g|webp|bmp|gif)$/i;
const VID_EXT = /\.(mp4|webm|mov|mkv|avi|m4v)$/i;
const AUD_EXT = /\.(wav|mp3|flac|ogg|m4a|aac)$/i;

async function uploadFile(file) {
    const fd = new FormData();
    fd.append("image", file, file.name);
    fd.append("type", "input");
    const resp = await fetch("/upload/image", { method: "POST", body: fd });
    if (!resp.ok) throw new Error(`upload ${file.name}: HTTP ${resp.status}`);
    return await resp.json(); // {name, subfolder, type}
}

async function addFiles(node, files) {
    const m = getMedia(node);
    const rejected = { img: 0, vid: 0, aud: 0 };
    for (const f of files) {
        const n = f.name;
        const isImg = IMG_EXT.test(n), isVid = VID_EXT.test(n), isAud = AUD_EXT.test(n);
        if (!isImg && !isVid && !isAud) continue;
        if (isImg && m.images.length >= MAX_IMGS) { rejected.img++; continue; }
        if (isVid && (m.reference_videos.length + (m.source_video ? 1 : 0)) >= MAX_VIDS) { rejected.vid++; continue; }
        if (isAud && m.audios.length >= MAX_AUDS) { rejected.aud++; continue; }
        try {
            const r = await uploadFile(f);
            const entry = { name: r.name, subfolder: r.subfolder || "" };
            if (isVid) {
                const num = nextNum(m.vid_nums);
                if (!m.source_video) { m.source_video = entry; m.vid_nums.unshift(num); }
                else { m.reference_videos.push(entry); m.vid_nums.push(num); }
            } else if (isImg) { m.images.push(entry); m.img_nums.push(nextNum(m.img_nums)); }
            else { m.audios.push(entry); m.aud_nums.push(nextNum(m.aud_nums)); }
        } catch (e) { console.error("[H3Panel]", e); }
    }
    setMedia(node, m);
    const parts = [];
    if (rejected.img) parts.push(`图片最多${MAX_IMGS}张`);
    if (rejected.vid) parts.push(`视频最多${MAX_VIDS}个`);
    if (rejected.aud) parts.push(`音频最多${MAX_AUDS}个`);
    showHint(node, parts.length ? `超出上限：${parts.join('，')}，已忽略 ${rejected.img + rejected.vid + rejected.aud} 个文件` : "");
    refreshPanel(node);
}

// transient red hint on the count badge
let hintTimer = null;
function showHint(node, msg) {
    const cnt = node._h3ui?.root?.querySelector(".h3p-count");
    if (!cnt) return;
    if (hintTimer) { clearTimeout(hintTimer); hintTimer = null; }
    if (!msg) { renderCount(node); return; }
    cnt.textContent = `⚠ ${msg}`;
    cnt.style.color = "#ff7a7a";
    hintTimer = setTimeout(() => { hintTimer = null; renderCount(node); }, 4000);
}

// ---------------------------------------------------------------- outputs
function mediaSlots(m) {
    const slots = [];
    if (m.source_video) slots.push({ label: `ref_video_0`, type: "IMAGE" });
    m.reference_videos.forEach((_, i) => slots.push({ label: `ref_video_${i + 1}`, type: "IMAGE" }));
    m.images.forEach((_, i) => slots.push({ label: `ref_image_${i}`, type: "IMAGE" }));
    m.audios.forEach((_, i) => slots.push({ label: `ref_audio_${i}`, type: "AUDIO" }));
    return slots;
}

function syncOutputs(node) {
    const slots = mediaSlots(getMedia(node));
    const target = slots.length;
    // NOTE: addOutput/removeOutput mutate node.outputs in place — never keep a
    // separate array copy, it would double-push / double-splice and corrupt slots.
    // 1) ensure media ports exist up to target (append only, indexes stay stable)
    while (node.outputs.length - 2 < target) {
        const s = slots[node.outputs.length - 2];
        if (!s) break;
        node.addOutput(s.label, s.type);
        const o = node.outputs[node.outputs.length - 1];
        if (s.type === "VIDEO") o.color_on = o.color_off = "#66a3ff";
        else if (s.type === "AUDIO") o.color_on = o.color_off = "#ffa04d";
    }
    // 2) trim tail beyond target (removeOutput = disconnect + splice, real delete);
    //    NEVER drop a connected port (keeps links alive across saves/reloads).
    //    Deleting from the tail keeps indexes valid as the array shrinks.
    for (let i = node.outputs.length - 1; i >= target + 2; i--) {
        const o = node.outputs[i];
        if (o && !o.links?.length) node.removeOutput(i);
    }
    // 3) refresh labels on retained ports
    for (let i = 0; i < target && i + 2 < node.outputs.length; i++) {
        const o = node.outputs[i + 2];
        const s = slots[i];
        if (o && s && o.name !== s.label) o.name = s.label;
    }
    node.setSize([node.size[0], node.computeSize()[1]]);
    app.graph?.setDirtyCanvas(true, true);
}

// ---------------------------------------------------------------- panel UI
function buildPanel(node) {
    const W = {};
    for (const nm of PANEL_WIDGETS) W[nm] = node.widgets?.find(w => w.name === nm);
    if (!W.prompt || !W.task_type || !W.model || !W.media_manifest) return;
    node._h3W = W;
    for (const nm of PANEL_WIDGETS) hideWidget(W[nm]);

    const root = document.createElement("div");
    root.className = "h3p";
    root.innerHTML = `
        <div class="h3p-top">
            <div class="h3p-media" title="点击或拖拽上传：视频 ≤3 / 图片 ≤9 / 音频 ≤3（混合总上限 12 个）"></div>
            <textarea class="h3p-prompt" spellcheck="false"
                placeholder="描述你想要的视频：画面内容、运镜、氛围…… 可上传视频/图片/音频作参考，在文中用 <Video 1>、<Picture 1>、<Audio 1> 引用它们"></textarea>
        </div>
        <div class="h3p-bar">
            <select class="h3p-sel h3p-task"></select>
            <button class="h3p-btn h3p-at" type="button" title="引用素材（输入 @ 也可唤起）">@</button>
            <button class="h3p-btn h3p-gear" type="button">⚙ 参数</button>
            <span class="h3p-count"></span>
        </div>
        <div class="h3p-adv"></div>
        <input type="file" class="h3p-file" multiple style="display:none"
               accept="image/*,video/*,audio/*,.wav,.mp3,.flac,.ogg,.m4a,.aac,.mp4,.webm,.mov,.mkv,.avi"/>`;

    const ta       = root.querySelector(".h3p-prompt");
    const selTask  = root.querySelector(".h3p-task");
    const atBtn    = root.querySelector(".h3p-at");
    const gearBtn  = root.querySelector(".h3p-gear");
    const advBox   = root.querySelector(".h3p-adv");
    const mediaBx  = root.querySelector(".h3p-media");
    const fileInp  = root.querySelector(".h3p-file");
    node._h3ui = { mediaBx, root, ta };

    // task select
    for (const v of (W.task_type.options?.values || Object.keys(TASK_LABELS))) {
        const o = document.createElement("option");
        o.value = v; o.textContent = TASK_LABELS[v] || v;
        selTask.appendChild(o);
    }
    selTask.addEventListener("change", () => { W.task_type.value = selTask.value; });

    // upload interactions
    mediaBx.addEventListener("click", (e) => {
        if (e.target.closest(".h3p-x")) return;
        fileInp.click();
    });
    fileInp.addEventListener("change", () => {
        if (fileInp.files?.length) addFiles(node, Array.from(fileInp.files));
        fileInp.value = "";
    });
    mediaBx.addEventListener("dragover", (e) => {
        e.preventDefault(); e.stopPropagation();
        mediaBx.classList.add("h3p-drag");
    });
    mediaBx.addEventListener("dragleave", () => mediaBx.classList.remove("h3p-drag"));
    mediaBx.addEventListener("drop", (e) => {
        e.preventDefault(); e.stopPropagation();
        mediaBx.classList.remove("h3p-drag");
        if (e.dataTransfer?.files?.length) addFiles(node, Array.from(e.dataTransfer.files));
    });

    // advanced drawer (model lives here now)
    let advOpen = false;
    const advCtrls = [];
    function row(label) {
        const r = document.createElement("div"); r.className = "h3p-row";
        const lb = document.createElement("label"); lb.textContent = label;
        r.appendChild(lb); advBox.appendChild(r); return r;
    }
    function numRow(label, w, step) {
        const r = row(label);
        const ip = document.createElement("input");
        ip.type = "number"; if (step) ip.step = String(step);
        ip.addEventListener("change", () => {
            const v = parseFloat(ip.value);
            if (!Number.isNaN(v)) w.value = v;
        });
        r.appendChild(ip);
        advCtrls.push(() => { ip.value = w.value; });
    }
    function chkRow(label, w) {
        const r = row(label);
        const ip = document.createElement("input"); ip.type = "checkbox";
        ip.addEventListener("change", () => { w.value = ip.checked; });
        r.appendChild(ip);
        advCtrls.push(() => { ip.checked = !!w.value; });
    }
    const SEL_LABELS = {
        frame_mode: { uniform: "均匀抽帧", smart: "智能(首/中/尾)" },
    };
    function selRow(label, w) {
        const r = row(label);
        const se = document.createElement("select");
        const fill = () => {
            se.innerHTML = "";
            const map = SEL_LABELS[w.name] || {};
            for (const v of (w.options?.values || [])) {
                const o = document.createElement("option");
                o.value = v; o.textContent = map[v] || v; o.title = v; se.appendChild(o);
            }
            se.value = w.value;
        };
        fill();
        se.addEventListener("change", () => { w.value = se.value; });
        r.appendChild(se);
        advCtrls.push(fill);
    }

    selRow("model", W.model);
    if (W.mmproj)         selRow("mmproj", W.mmproj);
    if (W.system_prompt) {
        const r = row("系统提示词");
        r.style.gridColumn = "1 / -1";
        const ta = document.createElement("textarea");
        ta.rows = 5;
        ta.value = W.system_prompt.value ?? "";
        ta.style.cssText = "width:100%;background:#262626;color:#cfcfcf;border:1px solid #303030;"
            + "border-radius:6px;padding:6px;font-size:11px;font-family:inherit;resize:vertical;min-width:0;";
        ta.addEventListener("input", () => { W.system_prompt.value = ta.value; });
        r.appendChild(ta);
        advCtrls.push(() => { ta.value = W.system_prompt.value ?? ""; });
    }
    if (W.user_template) {
        const r2 = row("输出规范模板");
        r2.style.gridColumn = "1 / -1";
        const ta2 = document.createElement("textarea");
        ta2.rows = 14;
        ta2.value = W.user_template.value ?? "";
        ta2.style.cssText = "width:100%;background:#262626;color:#cfcfcf;border:1px solid #303030;"
            + "border-radius:6px;padding:6px;font-size:11px;font-family:monospace;line-height:1.4;resize:vertical;min-width:0;";
        ta2.addEventListener("input", () => { W.user_template.value = ta2.value; });
        r2.appendChild(ta2);
        advCtrls.push(() => { ta2.value = W.user_template.value ?? ""; });
    }
    if (W.temperature)    numRow("temperature", W.temperature, 0.05);
    if (W.repeat_penalty) numRow("repeat_penalty", W.repeat_penalty, 0.01);
    if (W.seed)           numRow("seed", W.seed);
    if (W.n_ctx)          numRow("n_ctx", W.n_ctx, 1024);
    if (W.n_gpu_layers)   numRow("n_gpu_layers", W.n_gpu_layers);
    if (W.max_tokens)     numRow("max_tokens", W.max_tokens, 128);
    if (W.frame_mode)     selRow("抽帧方式", W.frame_mode);
    if (W.video_frames)   numRow("每视频帧数", W.video_frames);
    if (W.sample_fps)     numRow("采样帧率(0关)", W.sample_fps);
    if (W.sample_seconds) numRow("采样时长(秒)", W.sample_seconds, 0.5);
    if (W.image_max_side) numRow("图片边长", W.image_max_side, 32);
    if (W.thinking_mode)  chkRow("thinking_mode", W.thinking_mode);
    node._h3ui.advCtrls = advCtrls;

    // widget -> DOM
    ta.addEventListener("input", () => {
        W.prompt.value = ta.value;
        renderCount(node);
        // "@" opens the mention menu; range = the @ being typed
        if (ta.value[ta.selectionStart - 1] === "@") {
            openMention(node, caretRect(ta), { start: ta.selectionStart - 1, end: ta.selectionStart });
        } else if (mentionEl) {
            // keep menu open while typing a short filter after @; close on space/newline
            const before = ta.value.slice(0, ta.selectionStart);
            const at = before.lastIndexOf("@");
            if (at < 0 || /[\s@]/.test(before.slice(at + 1)) || before.slice(at + 1).length > 6) closeMention();
        }
    });
    ta.addEventListener("keydown", (e) => {
        if (mentionEl && ["ArrowDown", "ArrowUp", "Enter", "Tab", "Escape"].includes(e.key)) {
            mentionEl._onKey?.(e);
        }
    });
    ta.addEventListener("blur", () => setTimeout(() => { if (!mentionEl?.matches(":hover")) closeMention(); }, 150));
    atBtn.addEventListener("click", () => {
        ta.focus();
        openMention(node, caretRect(ta), { start: ta.selectionStart, end: ta.selectionStart });
    });
    node._h3ui.refreshFromWidgets = () => {
        if (ta.value !== (W.prompt.value ?? "")) ta.value = W.prompt.value ?? "";
        if (selTask.value !== W.task_type.value) selTask.value = W.task_type.value;
        for (const f of advCtrls) f();
    };

    // DOM widget
    const BASE_H = 280, ADV_H = 310;
    const domWidget = node.addDOMWidget("h3panel", "h3panel", root, {
        serialize: false, hideOnZoom: false,
        selectOn: ["click", "focus"],
        // ComfyUI 0.30 DOM widgets size via these callbacks (computeLayoutSize),
        // a plain computeSize override is ignored by the new frontend.
        getMinHeight: () => 200,
        getMaxHeight: () => 900,
        getHeight: () => (advOpen ? BASE_H + ADV_H : BASE_H),
        afterResize: () => {
            // issue #12443 workaround: never let a stale widget.width win over node.size[0]
            const _dw = node.widgets?.find(w => w.name === "h3panel");
            if (_dw && typeof _dw.width === "number") delete _dw.width;
        },
    });
    gearBtn.addEventListener("click", () => {
        advOpen = !advOpen;
        advBox.classList.toggle("h3p-open", advOpen);
        gearBtn.classList.toggle("h3p-on", advOpen);
        if (advOpen) {
            for (const f of advCtrls) f();  // refresh all drawer controls incl. template editors
        }
        node.setSize([node.size[0], node.computeSize()[1]]);
        app.graph?.setDirtyCanvas(true, true);
    });

    node.setSize([Math.max(node.size[0], 600), Math.max(node.size[1], node.computeSize()[1])]);

    // lifecycle
    const onConfigure = node.onConfigure;
    node.onConfigure = function () {
        const r = onConfigure?.apply(this, arguments);
        // rAF: the new frontend creates widgets asynchronously, so media_manifest
        // is only available after configure() fully settles. Re-sync ports then.
        requestAnimationFrame(() => {
            // Clear widget.width polluted by WidgetLegacy.vue (ComfyUI issue #12443):
            // it freezes the DOM widget width to a stale value and squeezes content.
            const _dw = node.widgets?.find(w => w.name === "h3panel");
            if (_dw && typeof _dw.width === "number") delete _dw.width;
            node._h3ui?.refreshFromWidgets?.();
            refreshPanel(node);
        });
        return r;
    };
    // live link/unlink from other nodes -> keep manifest-driven slots consistent
    const occ = node.onConnectionsChange;
    node.onConnectionsChange = function () {
        const r = occ?.apply(this, arguments);
        // During configure/link restore the manifest widget may not be populated yet;
        // touching ports then can corrupt slot layout. Skip until it has a value.
        const w = node._h3W?.media_manifest;
        if (!w || !w.value) return r;
        const want = mediaSlots(getMedia(node)).length + 2;
        if ((this.outputs?.length || 0) !== want) refreshPanel(this);
        return r;
    };

    node._h3ui.refreshFromWidgets();
    refreshPanel(node);
}

// render chips + count + outputs
function refreshPanel(node) {
    const ui = node._h3ui;
    if (!ui) return;
    const m = getMedia(node);
    // migrate manifests saved before numbering existed
    let dirty = false;
    const nVid = (m.source_video ? 1 : 0) + m.reference_videos.length;
    while (m.vid_nums.length < nVid) { m.vid_nums.push(nextNum(m.vid_nums)); dirty = true; }
    while (m.img_nums.length < m.images.length) { m.img_nums.push(nextNum(m.img_nums)); dirty = true; }
    while (m.aud_nums.length < m.audios.length) { m.aud_nums.push(nextNum(m.aud_nums)); dirty = true; }
    if (dirty) setMedia(node, m);
    const bx = ui.mediaBx;
    bx.innerHTML = "";

    const addBtn = document.createElement("div");
    addBtn.className = "h3p-addbtn";
    addBtn.innerHTML = `<div class="h3p-plus">+</div><div class="h3p-mtitle">参考 (${mediaCount(m)})</div>`;
    bx.appendChild(addBtn);

    const mkChip = (entry, kind, tag, onRemove, extra) => {
        const c = document.createElement("div");
        c.className = "h3p-chip";
        const nm = entryName(entry);
        const sub = entry?.subfolder ? `&subfolder=${encodeURIComponent(entry.subfolder)}` : "";
        if (kind === "img") {
            const im = document.createElement("img");
            im.src = `/view?filename=${encodeURIComponent(nm)}${sub}&type=input`;
            im.alt = nm;
            c.appendChild(im);
        } else {
            const ic = document.createElement("span");
            ic.className = "h3p-cicon";
            ic.textContent = kind === "vid" ? "🎬" : "🔊";
            c.appendChild(ic);
            const cn = document.createElement("span");
            cn.className = "h3p-cname"; cn.textContent = nm; cn.title = nm;
            c.appendChild(cn);
        }
        const tg = document.createElement("span");
        tg.className = "h3p-ctag"; tg.textContent = tag;
        c.appendChild(tg);
        if (extra?.onTrim) {
            const tb = document.createElement("span");
            tb.className = "h3p-trim"; tb.textContent = "✂"; tb.title = "剪辑音频";
            tb.addEventListener("click", (e) => { e.stopPropagation(); extra.onTrim(); });
            c.appendChild(tb);
        }
        if (extra?.trim) {
            const tm = document.createElement("span");
            tm.className = "h3p-trimmed";
            tm.textContent = `${extra.trim.start}s`;
            tm.title = `已裁剪 ${extra.trim.start}s – ${extra.trim.end}s`;
            c.appendChild(tm);
        }
        if (kind === "vid") {
            const pv = document.createElement("span");
            pv.className = "h3p-play"; pv.textContent = "▶"; pv.title = "预览视频";
            pv.addEventListener("click", (e) => {
                e.stopPropagation();
                openPlayer(entry, c.getBoundingClientRect(), tag, "vid");
            });
            c.appendChild(pv);
        }
        if (kind === "img") {
            c.addEventListener("click", (e) => {
                if (e.target.closest(".h3p-x")) return;
                e.stopPropagation();
                openPlayer(entry, c.getBoundingClientRect(), tag, "img");
            });
        }
        const x = document.createElement("span");
        x.className = "h3p-x"; x.textContent = "×"; x.title = "移除";
        x.addEventListener("click", (e) => { e.stopPropagation(); onRemove(); });
        c.appendChild(x);
        bx.appendChild(c);
    };

    if (m.source_video) mkChip(m.source_video, "vid", `<Video ${m.vid_nums[0] ?? 1}>·源`, () => {
        const mm = getMedia(node); mm.source_video = null; mm.vid_nums.shift(); setMedia(node, mm); refreshPanel(node);
    });
    m.reference_videos.forEach((e, i) => mkChip(e, "vid", `<Video ${m.vid_nums[i + 1] ?? i + 1}>`, () => {
        const mm = getMedia(node); mm.reference_videos.splice(i, 1); mm.vid_nums.splice(i + 1, 1); setMedia(node, mm); refreshPanel(node);
    }));
    m.images.forEach((e, i) => mkChip(e, "img", `<Picture ${m.img_nums[i] ?? i + 1}>`, () => {
        const mm = getMedia(node); mm.images.splice(i, 1); mm.img_nums.splice(i, 1); setMedia(node, mm); refreshPanel(node);
    }));
    m.audios.forEach((e, i) => mkChip(e, "aud", `<Audio ${m.aud_nums[i] ?? i + 1}>`, () => {
        const mm = getMedia(node); mm.audios.splice(i, 1); mm.aud_nums.splice(i, 1); setMedia(node, mm); refreshPanel(node);
    }, {
        trim: e.trim ? { start: e.trim.start, end: e.trim.end } : null,
        onTrim: () => openTrimDialog(node, i),
    }));

    renderCount(node);
    syncOutputs(node);
}

// ------------------------------------------------------------- @ mention menu
function mediaItems(node) {
    const m = getMedia(node);
    const items = [];
    const sub = (e) => e?.subfolder ? `&subfolder=${encodeURIComponent(e.subfolder)}` : "";
    if (m.source_video) items.push({ label: `<Video ${m.vid_nums[0] ?? 1}>`, kind: "vid", entry: m.source_video });
    m.reference_videos.forEach((e, i) => items.push({ label: `<Video ${m.vid_nums[i + 1] ?? i + 1}>`, kind: "vid", entry: e }));
    m.images.forEach((e, i) => items.push({ label: `<Picture ${m.img_nums[i] ?? i + 1}>`, kind: "img", entry: e }));
    m.audios.forEach((e, i) => items.push({ label: `<Audio ${m.aud_nums[i] ?? i + 1}>`, kind: "aud", entry: e }));
    return items;
}

let mentionEl = null;
function closeMention() {
    if (mentionEl) { mentionEl.remove(); mentionEl = null; }
    document.removeEventListener("mousedown", mentionOutside, true);
}
function mentionOutside(e) {
    if (mentionEl && !mentionEl.contains(e.target)) closeMention();
}
function openMention(node, anchorRect, replaceRange) {
    closeMention();
    const items = mediaItems(node);
    if (!items.length) return;
    const ta = node._h3ui.ta;
    mentionEl = document.createElement("div");
    mentionEl.className = "h3p-mention";
    mentionEl.innerHTML = `<div class="h3p-mhead">素材引用</div>`;
    let selIdx = 0;
    const rows = [];
    items.forEach((it, i) => {
        const r = document.createElement("div");
        r.className = "h3p-mitem";
        if (it.kind === "img") {
            const im = document.createElement("img");
            const s = it.entry?.subfolder ? `&subfolder=${encodeURIComponent(it.entry.subfolder)}` : "";
            im.src = `/view?filename=${encodeURIComponent(entryName(it.entry))}${s}&type=input`;
            r.appendChild(im);
        } else {
            const ic = document.createElement("span");
            ic.className = "h3p-micon";
            ic.textContent = it.kind === "vid" ? "🎬" : "🔊";
            r.appendChild(ic);
        }
        const lb = document.createElement("span"); lb.textContent = it.label;
        r.appendChild(lb);
        r.addEventListener("mousedown", (e) => { e.preventDefault(); pick(i); });
        mentionEl.appendChild(r); rows.push(r);
    });
    const pick = (i) => {
        const it = items[i];
        const insert = it.label;  // e.g. "<Video 1>"
        const start = replaceRange ? replaceRange.start : ta.selectionStart;
        const end = replaceRange ? replaceRange.end : ta.selectionStart;
        ta.value = ta.value.slice(0, start) + insert + ta.value.slice(end);
        ta.selectionStart = ta.selectionEnd = start + insert.length;
        ta.dispatchEvent(new Event("input", { bubbles: true }));
        closeMention(); ta.focus();
    };
    const hl = (i) => { rows.forEach((r, j) => r.classList.toggle("h3p-sel", j === i)); selIdx = i; };
    hl(0);
    mentionEl._onKey = (e) => {
        if (e.key === "ArrowDown") { e.preventDefault(); hl((selIdx + 1) % items.length); }
        else if (e.key === "ArrowUp") { e.preventDefault(); hl((selIdx - 1 + items.length) % items.length); }
        else if (e.key === "Enter" || e.key === "Tab") { e.preventDefault(); pick(selIdx); }
        else if (e.key === "Escape") { e.preventDefault(); closeMention(); }
    };
    document.body.appendChild(mentionEl);
    const x = Math.min(anchorRect.left, window.innerWidth - 200);
    const y = anchorRect.bottom + 4;
    mentionEl.style.left = Math.max(4, x) + "px";
    mentionEl.style.top = y + "px";
    setTimeout(() => document.addEventListener("mousedown", mentionOutside, true), 0);
}

function caretRect(ta) {
    const r = ta.getBoundingClientRect();
    return { left: r.left + 14, bottom: r.bottom - 8, top: r.top };
}

// ------------------------------------------------------------- audio trim
function viewUrl(entry) {
    const s = entry?.subfolder ? `&subfolder=${encodeURIComponent(entry.subfolder)}` : "";
    return `/view?filename=${encodeURIComponent(entryName(entry))}${s}&type=input`;
}

// floating media preview player (video + image)
let playerEl = null;
function closePlayer() {
    if (playerEl) {
        const v = playerEl.querySelector("video");
        if (v) { v.pause(); v.src = ""; }
        playerEl.remove(); playerEl = null;
    }
    document.removeEventListener("mousedown", playerOutside, true);
}
function playerOutside(e) {
    if (playerEl && !playerEl.contains(e.target)) closePlayer();
}
function openPlayer(entry, anchorRect, tag, kind) {
    const url = viewUrl(entry);
    playerEl = document.createElement("div");
    playerEl.className = "h3p-player";
    let mediaHtml;
    if (kind === "img") {
        mediaHtml = `<img src="${url}" alt="${tag||''}" style="display:block;max-width:520px;max-height:70vh"/>`;
    } else {
        mediaHtml = `<video controls autoplay preload="metadata" ${tag?`title="${tag}"`:""} style="display:block;max-width:520px;max-height:70vh"></video>`;
    }
    playerEl.innerHTML = `<button class="h3p-pclose" type="button" title="关闭">×</button>${mediaHtml}<div class="h3p-perr" style="display:none;position:absolute;bottom:6px;left:8px;color:#ff7a7a;font-size:11px">编码不支持？<a href="${url}" target="_blank" style="color:#9ed94a">新标签打开</a></div>`;
    const media = playerEl.querySelector("video") || playerEl.querySelector("img");
    if (kind === "vid" && media) {
        media.src = url;
        let t = setTimeout(() => { const ee = playerEl?.querySelector(".h3p-perr"); if(ee) ee.style.display="block"; }, 3000);
        media.addEventListener("canplay", () => { clearTimeout(t); const e = playerEl?.querySelector(".h3p-perr"); if(e) e.style.display="none"; }, {once:true});
    }
    playerEl.querySelector(".h3p-pclose").addEventListener("click", (e) => { e.stopPropagation(); closePlayer(); });
    document.body.appendChild(playerEl);
    const w = 520, h = 360;
    const x = Math.min(Math.max(8, anchorRect.right + 8), window.innerWidth - w - 8);
    const y = Math.min(Math.max(8, anchorRect.top), window.innerHeight - h - 8);
    playerEl.style.left = x + "px";
    playerEl.style.top = y + "px";
    setTimeout(() => document.addEventListener("mousedown", playerOutside, true), 0);
}

function openTrimDialog(node, audIdx) {
    const m = getMedia(node);
    const entry = m.audios[audIdx];
    if (!entry) return;
    const num = m.aud_nums[audIdx] ?? audIdx + 1;
    const overlay = document.createElement("div");
    overlay.className = "h3p-overlay";
    const trim = entry.trim || null;
    overlay.innerHTML = `
        <div class="h3p-dialog">
            <h4>剪辑音频 · Audio ${num}</h4>
            <audio class="h3p-audio" controls preload="metadata" style="width:100%;height:32px"></audio>
            <div class="h3p-trow"><label>开始(秒)</label><input type="number" class="h3p-ts" min="0" step="0.1" value="${trim ? trim.start : 0}"></div>
            <div class="h3p-trow"><label>结束(秒)</label><input type="number" class="h3p-te" min="0" step="0.1" value="${trim ? trim.end : 0}"></div>
            <div class="h3p-tinfo"></div>
            <div class="h3p-trow" style="color:#888;font-size:11px">每段至少 2 秒，总时长不超过 15 秒；结束填 0 表示到文件末尾</div>
            <div class="h3p-tbtns">
                <button class="h3p-tclear" type="button">清除裁剪</button>
                <button class="h3p-tcancel" type="button">取消</button>
                <button class="h3p-primary h3p-tok" type="button">确认</button>
            </div>
        </div>`;
    document.body.appendChild(overlay);
    const audio = overlay.querySelector(".h3p-audio");
    audio.src = viewUrl(entry);
    const ts = overlay.querySelector(".h3p-ts");
    const te = overlay.querySelector(".h3p-te");
    const info = overlay.querySelector(".h3p-tinfo");
    let dur = 0;
    audio.addEventListener("loadedmetadata", () => {
        dur = audio.duration || 0;
        if (!trim) te.value = dur ? dur.toFixed(1) : 0;
        upd();
    });
    function upd() {
        const s = parseFloat(ts.value) || 0;
        const e0 = parseFloat(te.value) || 0;
        const e = e0 <= 0 ? dur : e0;
        const len = e - s;
        if (!dur) { info.textContent = "读取音频中…"; info.className = "h3p-tinfo"; return; }
        if (len < 2 || len > 15 || s < 0 || e > dur + 0.01) {
            info.textContent = `当前片段 ${len.toFixed(1)} 秒（要求 2–15 秒，且不超出总时长 ${dur.toFixed(1)} 秒）`;
            info.className = "h3p-tinfo h3p-terr";
        } else {
            info.textContent = `当前片段 ${len.toFixed(1)} 秒 ✓`;
            info.className = "h3p-tinfo";
        }
    }
    ts.addEventListener("input", upd); te.addEventListener("input", upd);
    const close = () => { audio.pause(); overlay.remove(); };
    overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) close(); });
    overlay.querySelector(".h3p-tcancel").addEventListener("click", close);
    overlay.querySelector(".h3p-tclear").addEventListener("click", () => {
        const mm = getMedia(node);
        if (mm.audios[audIdx]) delete mm.audios[audIdx].trim;
        setMedia(node, mm); refreshPanel(node); close();
    });
    overlay.querySelector(".h3p-tok").addEventListener("click", () => {
        const s = parseFloat(ts.value) || 0;
        const e0 = parseFloat(te.value) || 0;
        const e = e0 <= 0 ? (dur || 0) : e0;
        if (e - s < 2 || e - s > 15 || s < 0) { upd(); return; }
        const mm = getMedia(node);
        if (mm.audios[audIdx]) mm.audios[audIdx].trim = { start: s, end: e };
        setMedia(node, mm); refreshPanel(node); close();
    });
}

// count badge + dangling reference check (<Picture N>/<Video N>/<Audio N>)
function renderCount(node) {    const ui = node._h3ui;
    const cnt = ui?.root?.querySelector(".h3p-count");
    if (!cnt) return;
    const m = getMedia(node);
    const prompt = node._h3W?.prompt?.value || "";
    const dangling = [];
    const scan = (re, nums, label) => {
        for (const mt of prompt.matchAll(re)) {
            const n = parseInt(mt[1], 10);
            if (!nums.includes(n) && !dangling.includes(`<${label} ${n}>`)) dangling.push(`<${label} ${n}>`);
        }
    };
    scan(/<Picture (\d+)>/g, m.img_nums, "Picture");
    scan(/<Video (\d+)>/g, m.vid_nums, "Video");
    scan(/<Audio (\d+)>/g, m.aud_nums, "Audio");
    if (dangling.length) {
        cnt.textContent = `⚠ 未上传：${dangling.join(" ")}`;
        cnt.style.color = "#ff7a7a";
    } else {
        cnt.textContent = `🖼 ${m.images.length}  🎬 ${m.reference_videos.length + (m.source_video ? 1 : 0)}  🔊 ${m.audios.length}`;
        cnt.style.color = "";
    }
}

app.registerExtension({
    name: "Bernini.H3Panel",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3PromptEnhancer" && nodeData.name !== "H3OmniPromptEnhancer") return;
        injectCss();
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            try { buildPanel(this); } catch (e) { console.error("[H3Panel]", e); }
            return r;
        };
    },
});

import { app } from "../../scripts/app.js";

const VARIANTS = {
    "r2v": [{ value: "none", text: "none" },
            { value: "motion (r2v)", text: "motion — ref=character, source=motion" }],
    "v2v": [{ value: "none", text: "none" },
            { value: "storyboard (v2v)", text: "storyboard — 3 camera angles from 1 action" }],
    "t2v": [{ value: "none", text: "none" },
            { value: "cinematic (t2v)", text: "cinematic — film aesthetics" },
            { value: "anime (t2v)", text: "anime — Japanese animation style" },
            { value: "realistic (t2v)", text: "realistic — photorealism" },
            { value: "director (t2v)", text: "director — precise camera/shot control" }],
    "rv2v": [{ value: "none", text: "none" },
             { value: "3dreal (rv2v)", text: "3dreal — 3D render → realistic" }],
};

function updateVariantWidget(widget, parentTask) {
    const opts = VARIANTS[parentTask] || [{ value: "none", text: "none" }];
    widget.options.values = opts.map(o => o.value);
    widget.options.texts = opts.map(o => o.text);
    if (!opts.find(o => o.value === widget.value)) {
        widget.value = "none";
    }
}

app.registerExtension({
    name: "Bernini.Variant",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "BerniniMLLMPromptEnhancer" &&
            nodeData.name !== "Qwen35PromptEnhancer") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const taskW = this.widgets.find(w => w.name === "task_type");
            const varW = this.widgets.find(w => w.name === "variant");
            if (!taskW || !varW) return r;

            updateVariantWidget(varW, taskW.value);
            const prev = taskW.callback;
            taskW.callback = function (v) {
                prev?.call(this, v);
                updateVariantWidget(varW, v);
            };
            return r;
        };
    },
});

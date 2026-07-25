import { app } from "../../scripts/app.js";

const VARIANTS = {
    "r2v": [{ value: "none", text: "none (default)" },
            { value: "motion", text: "motion — ref=character, source=motion" }],
};

function updateVariantWidget(widget, parentTask) {
    const opts = VARIANTS[parentTask] || [{ value: "none", text: "none (default)" }];
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

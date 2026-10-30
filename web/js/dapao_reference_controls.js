export function installCanvasTitleButton(node) {
    const linkBox = () => [node.size[0] - 92, -28, 82, 24];
    const openCanvas = () => window.open("https://paopao.dapaoai.com/", "_blank", "noopener,noreferrer");
    const titleButton = node.addTitleButton?.({name: "dapao_infinite_canvas", text: "无限画布 ↗", fontSize: 12, xOffset: -6});
    if (titleButton?.draw && titleButton?.getWidth) {
        // Retain native title layout and hit testing, with padding included in both.
        const measure = titleButton.getWidth;
        const drawButton = titleButton.draw;
        titleButton.getWidth = function (ctx) {
            return this.visible ? measure.call(this, ctx) + 16 : 0;
        };
        titleButton.draw = function (ctx, x, y) {
            if (!this.visible) return;
            ctx.save();
            ctx.fillStyle = "#244b70";
            ctx.strokeStyle = "#71a9dc";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(x + this.xOffset, y + this.yOffset, this.getWidth(ctx), this.height, 5);
            ctx.fill(); ctx.stroke();
            ctx.fillStyle = "#f0f7ff";
            drawButton.call(this, ctx, x, y);
            ctx.restore();
        };
    }
    const titleClick = node.onTitleButtonClick;
    node.onTitleButtonClick = function (button) {
        if (button.name === "dapao_infinite_canvas") { openCanvas(); return; }
        return titleClick?.apply(this, arguments);
    };
    return { titleButton, linkBox, openCanvas };
}

// Shared local-gallery controls. No generation or workflow submission is needed.
const livePreviews = new WeakMap();
export function showConnectedPreview(target, image) {
    if (!livePreviews.has(target)) {
        const state = { image: null };
        livePreviews.set(target, state);
        const background = target.onDrawBackground;
        target.onDrawBackground = function () {
            const result = background?.apply(this, arguments);
            // A previously executed preview may still be decoding asynchronously.
            if (state.image) { this.imgs = [state.image]; this.imageIndex = 0; }
            return result;
        };
        const executed = target.onExecuted;
        target.onExecuted = function () {
            state.image = null; // Normal workflow output takes precedence when run.
            return executed?.apply(this, arguments);
        };
        const configure = target.onConfigure;
        target.onConfigure = function () { state.image = null; return configure?.apply(this, arguments); };
    }
    livePreviews.get(target).image = image;
    target.imgs = [image]; target.imageIndex = 0;
    target.animatedImages = false; target.previewMediaType = "image";
    target.setSize?.([target.size[0], Math.max(target.size[1], target.computeSize?.()[1] || 0, 300)]);
    target.setDirtyCanvas?.(true, true);
}

export function referenceControls(node, root, { items, current, select, imageUrl, blocked, textOutputs = () => [] }) {
    const message = root.querySelector(".dpn-message");
    let alive = true, rolling = false, epoch = 0;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const diceBox = [14, 8, 44, 40];
    const { titleButton, linkBox, openCanvas } = installCanvasTitleButton(node);
    const inside = (p, box) => p && p[0] >= box[0] && p[0] <= box[0] + box[2] && p[1] >= box[1] && p[1] <= box[1] + box[3];
    const draw = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
        draw?.apply(this, arguments);
        if (this.flags?.collapsed) return;
        ctx.save();
        ctx.fillStyle = "#303b4c"; ctx.beginPath(); ctx.roundRect(...diceBox, 8); ctx.fill();
        ctx.translate(36, 28);
        if (rolling && !reducedMotion.matches) ctx.rotate(performance.now() / 95);
        ctx.font = '28px "Segoe UI Emoji", sans-serif'; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.globalAlpha = blocked() ? .45 : 1; ctx.fillText("🎲", 0, 0);
        ctx.restore();
        if (titleButton) return;
        ctx.save();
        ctx.fillStyle = "#244b70"; ctx.strokeStyle = "#71a9dc"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.roundRect(...linkBox(), 5); ctx.fill(); ctx.stroke();
        ctx.fillStyle = "#f0f7ff"; ctx.font = '12px "Microsoft YaHei", sans-serif';
        ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText("无限画布 ↗", linkBox()[0] + 6, -16); ctx.restore();
    };
    async function roll() {
        if (!alive || rolling || blocked() || !items()?.length) return;
        const token = epoch, previous = current();
        rolling = true;
        const start = performance.now();
        await new Promise(resolve => {
            const tick = () => {
                node.setDirtyCanvas(true, true);
                if (!alive || epoch !== token || performance.now() - start >= 480) return resolve();
                requestAnimationFrame(tick);
            }; tick();
        });
        try {
            if (!alive || token !== epoch || blocked() || current() !== previous) return;
            const candidates = items().filter(item => item.id !== current());
            const pool = candidates.length ? candidates : items();
            const item = pool[Math.floor(Math.random() * pool.length)];
            await select(item);
            if (alive && token === epoch && current() === item.id) await previewConnected(item, token);
        } finally { rolling = false; node.setDirtyCanvas(true, true); }
    }
    const down = node.onMouseDown;
    node.onMouseDown = function (event, position) {
        if (!this.flags?.collapsed && (event.button === undefined || event.button === 0)) {
            if (inside(position, diceBox)) { void roll(); return true; }
            if (!titleButton && inside(position, linkBox())) { openCanvas(); return true; }
        }
        return down?.apply(this, arguments);
    };
    // Keyboard-accessible alternatives also provide discoverable labels.
    const actions = document.createElement("div"); actions.className = "dpn-accessible-actions";
    const dice = document.createElement("button"); dice.type = "button"; dice.textContent = "🎲 随机选择"; dice.onclick = roll;
    const link = document.createElement("a"); link.textContent = "无限画布";
    link.href = "https://paopao.dapaoai.com/"; link.target = "_blank"; link.rel = "noopener noreferrer";
    actions.append(dice, link); root.prepend(actions);
    const configure = node.onConfigure;
    node.onConfigure = function () { epoch++; return configure?.apply(this, arguments); };
    const remove = node.onRemoved;
    node.onRemoved = function () { alive = false; epoch++; return remove?.apply(this, arguments); };
    // Only direct IMAGE consumers of type PreviewImage are updated. Never queue a prompt.
    function previewTargets() {
        const graph = node.graph;
        return [...new Set((node.outputs?.[0]?.links || []).map(id => {
            const edge = graph?.links?.get?.(id) || graph?.links?.[id];
            if (!edge || edge.origin_id !== node.id || edge.origin_slot !== 0) return null;
            const target = graph.getNodeById(edge.target_id);
            return target && (target.comfyClass || target.type) === "PreviewImage" && edge.target_slot === 0 ? target : null;
        }).filter(Boolean))];
    }
    function ensurePreviewTargets() {
        const connected = previewTargets();
        if (connected.length) return connected;
        // Respect an existing text-only (or other) downstream layout.
        if (node.outputs?.some(output => output.links?.length)) return [];
        const graph = node.graph;
        if (!graph) throw new Error("素材节点已不在画布中");
        const target = globalThis.LiteGraph?.createNode("PreviewImage");
        if (!target) throw new Error("无法创建预览图像节点，请刷新页面后重试");
        target.size = [320, 360];
        const x = node.pos[0] + node.size[0] + 80;
        let y = node.pos[1];
        // Keep existing nodes in place; find room to the right of this node.
        const others = graph.nodes || graph._nodes || [];
        for (let attempt = 0; attempt <= others.length; attempt++) {
            const overlap = others.find(other => other !== node && other.pos && other.size &&
                x < other.pos[0] + other.size[0] + 24 && x + target.size[0] + 24 > other.pos[0] &&
                y < other.pos[1] + other.size[1] + 40 && y + target.size[1] + 40 > other.pos[1]);
            if (!overlap) break;
            y = overlap.pos[1] + overlap.size[1] + 60;
        }
        target.pos = [x, y];
        graph.beforeChange?.();
        try {
            graph.add(target);
            const connection = node.connect(0, target, 0);
            if (!connection) throw new Error("预览图像自动连线失败，请重试");
        } catch (error) {
            // Roll back only the node created by this attempt.
            if (target.graph === graph) graph.remove(target);
            throw error;
        } finally { graph.afterChange?.(); }
        graph.setDirtyCanvas?.(true, true);
        return [target];
    }
    function previewText(item) {
        const supported = new Set(["PreviewAny", "ShowText|pysssss", "easy showAnything"]);
        const values = textOutputs(item);
        const skipped = new Set();
        for (let slot = 1; slot < values.length; slot++) {
            if (typeof values[slot] !== "string") continue;
            for (const id of node.outputs?.[slot]?.links || []) {
                const edge = node.graph?.links?.get?.(id) || node.graph?.links?.[id];
                if (!edge || edge.origin_id !== node.id || edge.origin_slot !== slot) continue;
                const target = node.graph.getNodeById(edge.target_id);
                if (!target) continue;
                if (!supported.has(target.comfyClass || target.type) || edge.target_slot !== 0 || !target.onExecuted) {
                    skipped.add(target.title || target.type); continue;
                }
                target.onExecuted({text: [values[slot]]});
                target.setDirtyCanvas?.(true, true);
            }
        }
        return skipped;
    }
    async function previewConnected(item, token) {
        const image = new Image(); image.src = imageUrl(item);
        try {
            await image.decode();
            if (!alive || token !== epoch || current() !== item.id) return;
            const targets = ensurePreviewTargets();
            for (const target of targets) {
                showConnectedPreview(target, image);
            }
            const skipped = previewText(item);
            if (message) message.textContent = skipped.size ? `素材已更新；${[...skipped].join("、")}需正常运行工作流后更新。` : "";
        } catch (error) {
            if (alive && token === epoch && message) message.textContent = `预览失败：${error.message || "图片读取失败，请重试"}`;
        }
    }
    return { update() { dice.disabled = blocked() || !items()?.length; } };
}

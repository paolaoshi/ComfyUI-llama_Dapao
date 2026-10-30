import { referenceControls } from "./dapao_reference_controls.js";
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const ROOT = "/dapao/pose-gallery";
const ANGLES = ["正面", "斜侧", "侧面", "俯视", "仰视"];
const url = item => api.apiURL(`${ROOT}/image/${encodeURIComponent(item.id)}`);
const key = item => `${item.system}|${item.number}`;
const el = (tag, cls, text) => {
    const element = document.createElement(tag);
    if (cls) element.className = cls;
    if (text) element.textContent = text;
    return element;
};
const button = (text, fn) => {
    const b = el("button", "", text); b.type = "button"; b.onclick = fn; return b;
};
function styles() {
    if (document.getElementById("dapao-pose-style")) return;
    const link = el("link"); link.id = "dapao-pose-style"; link.rel = "stylesheet";
    link.href = new URL("./dapao_pose_reference.css", import.meta.url).href;
    document.head.append(link);
}
function groups(items) {
    const map = new Map();
    for (const item of items) {
        if (!map.has(key(item))) map.set(key(item), []);
        map.get(key(item)).push(item);
    }
    return [...map.values()].map(g => g.sort((a, b) => ANGLES.indexOf(a.angle) - ANGLES.indexOf(b.angle)));
}
let catalogPromise;
async function catalog(reload = false) {
    if (reload) catalogPromise = null;
    if (!catalogPromise) catalogPromise = api.fetchApi(`${ROOT}/catalog`).then(async response => {
        if (!response.ok) throw new Error("目录读取失败，请检查插件素材并重试。");
        const data = await response.json();
        if (data.version !== 1 || !Array.isArray(data.items) || !data.items.length) throw new Error("姿势目录格式无效。");
        return data;
    }).catch(error => { catalogPromise = null; throw error; });
    return catalogPromise;
}

// Exported to let the isolated browser harness exercise the actual production dialog.
export function openPoseGallery(selectedId, onSelect, { itemFilter = () => true } = {}) {
    styles();
    const dialog = el("dialog", "dapao-pose");
    const header = el("header"), title = el("div");
    title.append(el("h2", "", "精选姿势参考"), el("small", "", "本地多角度姿势库 · 选择一个视角输出到节点"));
    const close = () => dialog.close();
    header.append(title, button("重新读取目录", () => load(true)), button("使用指南与来源", help), button("关闭", close));
    const filters = el("section", "dp-filters"), systems = el("div", "dp-row");
    const row = el("div", "dp-row"), search = el("input");
    search.placeholder = "搜索姿势名、编号、ID 或分类（/）"; search.setAttribute("aria-label", "搜索姿势");
    const category = el("select"), mode = el("select");
    category.setAttribute("aria-label", "动作分类"); mode.setAttribute("aria-label", "展示方式");
    mode.add(new Option("按姿势分组", "group")); mode.add(new Option("逐张浏览", "single"));
    row.append(search, category, mode);
    const angles = el("div", "dp-row"), count = el("span", "dp-count");
    const chosenAngles = new Set(); let system = "";
    angles.append(el("span", "", "视角（可多选）"));
    for (const angle of ANGLES) {
        const b = button(angle, () => {
            if (chosenAngles.has(angle)) chosenAngles.delete(angle); else chosenAngles.add(angle);
            b.setAttribute("aria-pressed", String(chosenAngles.has(angle))); renderList();
        }); b.setAttribute("aria-pressed", "false"); angles.append(b);
    }
    angles.append(button("重置筛选", reset), count);
    filters.append(systems, row, angles);
    const main = el("main", "dp-main"), list = el("section", "dp-list"), detail = el("aside", "dp-detail");
    list.setAttribute("aria-label", "姿势图库列表"); detail.setAttribute("aria-label", "姿势详情"); detail.tabIndex = 0;
    main.append(list, detail);
    const footer = el("footer"), status = el("span", "dp-status"), apply = button("请先选择姿势", applySelection);
    status.setAttribute("role", "status"); apply.className = "dp-primary"; apply.disabled = true;
    footer.append(status, apply);
    dialog.append(header, filters, main, footer);
    let items = [], allGroups = [], filteredGroups = [], cards = [], shown = 30, active = null, busy = false, epoch = 0;
    let zoomDialog = null, helpDialogRef = null;
    dialog.addEventListener("close", () => { epoch++; zoomDialog?.close(); helpDialogRef?.close(); dialog.remove(); });
    dialog.addEventListener("click", event => { if (event.target === dialog) close(); });
    dialog.addEventListener("keydown", event => {
        event.stopPropagation();
        if (event.target.closest("input,select,textarea")) return;
        if (event.key === "/") { event.preventDefault(); search.focus(); }
        if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
            event.preventDefault();
            if (event.key === "ArrowLeft" || event.key === "ArrowRight") stepAngle(event.key === "ArrowRight" ? 1 : -1);
            else stepPose(event.key === "ArrowDown" ? 1 : -1);
        }
    });
    search.oninput = category.onchange = mode.onchange = () => renderList();
    document.body.append(dialog); dialog.showModal(); load();

    async function load(reload = false) {
        list.replaceChildren(el("p", "", "正在加载姿势目录…"));
        try {
            const data = await catalog(reload);
            if (!dialog.isConnected) return;
            items = data.items.filter(itemFilter); allGroups = groups(items);
            renderSystems(); renderCategories(); renderList();
            const saved = items.find(i => i.id === selectedId);
            if (saved) choose(saved);
            else detail.replaceChildren(el("p", "", selectedId ? "上次选择不在目录中，请重新选择。" : "点击姿势卡片，查看所有可用角度。"));
        } catch (error) {
            list.replaceChildren(el("p", "dp-error", error.message), button("重试读取目录", () => load(true)));
        }
    }
    function renderSystems() {
        systems.replaceChildren();
        for (const value of ["", ...new Set(items.map(i => i.system))]) {
            const subset = value ? items.filter(i => i.system === value) : items;
            const b = button(`${value || "全部"} · ${groups(subset).length} 组 / ${subset.length} 张`, () => {
                system = value; category.value = ""; renderSystems(); renderCategories(); renderList();
            }); b.setAttribute("aria-pressed", String(system === value)); systems.append(b);
        }
    }
    function renderCategories() {
        category.replaceChildren(new Option("全部动作分类", ""));
        for (const value of new Set(items.filter(i => !system || i.system === system).map(i => i.category))) category.add(new Option(value, value));
    }
    function reset() {
        search.value = ""; system = ""; chosenAngles.clear();
        angles.querySelectorAll("[aria-pressed]").forEach(b => b.setAttribute("aria-pressed", "false"));
        renderSystems(); renderCategories(); renderList();
    }
    function renderList() {
        const tokens = search.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
        const matches = items.filter(i => (!system || i.system === system) && (!category.value || i.category === category.value)
            && (!chosenAngles.size || chosenAngles.has(i.angle))
            && tokens.every(t => `${i.id} ${i.name} ${i.number} ${i.system} ${i.category} ${i.angle}`.toLowerCase().includes(t)));
        filteredGroups = groups(matches); cards = mode.value === "group" ? filteredGroups : matches.map(i => [i]); shown = 30;
        count.textContent = `${filteredGroups.length} 组 · ${matches.length} 张`;
        list.scrollTop = 0; drawCards();
    }
    function drawCards() {
        list.replaceChildren(); const grid = el("div", "dp-grid");
        for (const group of cards.slice(0, shown)) {
            const card = el("article", "dp-card"); let preview = group[0];
            card.dataset.group = key(preview);
            const imageButton = button("", () => choose(preview));
            imageButton.className = "dp-card-image";
            const image = el("img"); image.loading = "lazy"; image.decoding = "async";
            const setPreview = item => { preview = item; image.src = url(item); image.alt = `${item.name} · ${item.angle}`; imageButton.setAttribute("aria-label", image.alt); };
            setPreview(preview); imageButton.append(image);
            image.onerror = () => { image.alt = "图片加载失败，点击查看详情重试"; };
            const tabs = el("div", "dp-card-angles");
            for (const item of group) {
                const b = button(item.angle, () => { setPreview(item); choose(item); });
                b.onpointerenter = () => setPreview(item); b.onfocus = () => setPreview(item); tabs.append(b);
            }
            card.append(imageButton, el("h3", "", preview.name), el("small", "", `${preview.number} · ${preview.category}`), tabs); grid.append(card);
        }
        list.append(grid);
        if (!cards.length) list.append(el("p", "", "没有找到匹配的姿势。"), button("清除全部筛选", reset));
        if (cards.length > shown) list.append(button(`加载更多（剩余 ${cards.length - shown}）`, () => {
            const scroll = list.scrollTop; shown += 30; drawCards(); list.scrollTop = scroll;
        }));
        markSelection();
    }
    function markSelection() {
        list.querySelectorAll(".dp-card").forEach(c => c.classList.toggle("dp-selected", !!active && c.dataset.group === key(active)));
    }
    function choose(item) {
        if (busy) return;
        active = item; status.textContent = "图片：ギア · 中文整理：Anhao"; markSelection(); renderDetail();
        detail.focus({preventScroll: true});
        if (window.innerWidth <= 760) detail.scrollIntoView({block: "start"});
    }
    function currentGroup() { return active ? allGroups.find(g => key(g[0]) === key(active)) || [] : []; }
    function stepAngle(direction) {
        const g = currentGroup(); if (g.length) choose(g[(g.findIndex(i => i.id === active.id) + direction + g.length) % g.length]);
    }
    function stepPose(direction) {
        if (!filteredGroups.length) return;
        const index = filteredGroups.findIndex(g => active && key(g[0]) === key(active));
        const g = filteredGroups[index < 0 ? (direction > 0 ? 0 : filteredGroups.length - 1) : (index + direction + filteredGroups.length) % filteredGroups.length];
        choose(g.find(i => i.angle === active?.angle) || g[0]);
    }
    function renderDetail() {
        const item = active, group = currentGroup(); detail.replaceChildren();
        detail.append(el("h3", "", item.name));
        const photo = el("img", "dp-detail-image"); photo.src = url(item); photo.alt = `${item.name} · ${item.angle}`;
        const photoButton = button("", zoom); photoButton.className = "dp-photo-button"; photoButton.setAttribute("aria-label", "放大查看当前姿势"); photoButton.append(photo);
        const loading = el("small", "", "正在加载图片…");
        photo.onload = () => { loading.textContent = "点击图片放大 · 右键可保存图片"; };
        photo.onerror = () => { loading.replaceChildren(el("span", "dp-error", "图片加载失败。"), button("重新加载", () => { photo.src = `${url(item)}?retry=${Date.now()}`; })); };
        let touch;
        photoButton.ontouchstart = event => { touch = event.touches[0]; };
        photoButton.ontouchend = event => {
            const end = event.changedTouches[0];
            if (touch && Math.abs(end.clientX - touch.clientX) > 45 && Math.abs(end.clientY - touch.clientY) < 60) {
                event.preventDefault(); stepAngle(end.clientX < touch.clientX ? 1 : -1);
            } touch = null;
        };
        const tabs = el("div", "dp-row");
        for (const angle of ANGLES) {
            const candidate = group.find(i => i.angle === angle);
            const b = button(angle + (candidate ? "" : "（缺）"), () => choose(candidate)); b.disabled = !candidate || busy;
            b.title = candidate ? angle : "原图库未提供此视角"; b.setAttribute("aria-pressed", String(item.angle === angle)); tabs.append(b);
        }
        const actions = el("div", "dp-row");
        actions.append(button("复制 ID", async () => {
            try { await navigator.clipboard.writeText(item.id); status.textContent = "已复制姿势 ID"; }
            catch { status.textContent = "复制失败，可手动选中下方 ID 复制。"; }
        }), button("下载参考图", () => download(item)), button("新窗口打开", () => window.open(url(item), "_blank", "noopener,noreferrer")));
        const navigation = el("div", "dp-row"); navigation.append(button("上一姿势", () => stepPose(-1)), button("下一姿势", () => stepPose(1)));
        detail.append(photoButton, loading, tabs, el("p", "", `${item.name}，属于「${item.system}」的「${item.category}」动作。当前为${item.angle}，提供 ${group.length} 个视角。`),
            el("small", "", "600 × 800 · WebP · 本地压缩素材"), el("code", "", item.id), actions, navigation,
            el("small", "", "← → 切换角度 · ↑ ↓ 切换姿势 · / 搜索 · Esc 关闭"));
        apply.disabled = busy; apply.textContent = `使用此视角 · ${item.angle}`;
    }
    async function download(item) {
        try {
            const response = await api.fetchApi(`${ROOT}/image/${encodeURIComponent(item.id)}`);
            if (!response.ok) throw new Error("图片读取失败");
            const blobUrl = URL.createObjectURL(await response.blob());
            const a = el("a"); a.href = blobUrl; a.download = `${item.id}.webp`; a.click();
            setTimeout(() => URL.revokeObjectURL(blobUrl), 1000); status.textContent = "已发起参考图下载";
        } catch (error) { status.textContent = `下载失败：${error.message}`; }
    }
    async function applySelection() {
        if (!active || busy) return;
        const selected = active, token = ++epoch; busy = true; apply.disabled = true; apply.textContent = "正在检查参考图…";
        try {
            const image = new Image(); image.src = url(selected); await image.decode();
            if (token !== epoch || !dialog.isConnected) return;
            await onSelect(selected); close();
        } catch (error) { if (dialog.isConnected) status.textContent = `选择失败，原输出已保留：${error.message}`; }
        finally { busy = false; if (dialog.isConnected) { renderDetail(); } }
    }
    function zoom() {
        zoomDialog = el("dialog", "dapao-pose dp-zoom"); const viewer = zoomDialog;
        const toolbar = el("div", "dp-row"), viewport = el("div", "dp-zoom-viewport"), image = el("img");
        let scale = 1; image.src = url(active); image.alt = `${active.name} · ${active.angle}`;
        const resize = value => { scale = Math.max(0.5, Math.min(4, value)); image.style.width = `${scale * 100}%`; };
        const move = direction => { stepAngle(direction); image.src = url(active); image.alt = `${active.name} · ${active.angle}`; };
        toolbar.append(button("上一视角", () => move(-1)), button("下一视角", () => move(1)), button("−", () => resize(scale - .25)), button("+", () => resize(scale + .25)), button("适应窗口", () => resize(1)), button("关闭大图", () => viewer.close()));
        viewport.append(image); viewer.append(toolbar, viewport); document.body.append(viewer);
        viewport.onwheel = event => { if (event.ctrlKey) { event.preventDefault(); resize(scale + (event.deltaY < 0 ? .1 : -.1)); } };
        viewer.onkeydown = event => { event.stopPropagation(); if (event.key === "ArrowLeft" || event.key === "ArrowRight") { event.preventDefault(); move(event.key === "ArrowRight" ? 1 : -1); } };
        viewer.onclose = () => { viewer.remove(); zoomDialog = null; if (dialog.isConnected) detail.focus({preventScroll:true}); }; viewer.showModal();
    }
    function help() {
        const helpDialog = el("dialog", "dapao-pose dp-help");
        helpDialogRef = helpDialog;
        helpDialog.append(el("h2", "", "使用指南与来源"), el("p", "", "按体系、动作分类、名称、编号或 ID 搜索；视角可多选。分组模式比较同一姿势的角度，逐张模式浏览具体画面。卡片视角支持悬停预览，点击查看详情。原库缺失的角度不会补造。"),
            el("p", "", "点击“使用此视角”保存选择，工作流运行时从本地素材输出 IMAGE。取消或失败保留原选择，不会自动生成图片。← → 切换角度，↑ ↓ 切换姿势，/ 搜索；触屏可左右滑动详情图。大图支持缩放、滚动和浏览器原生右键保存。"),
            el("p", "", "图片与原始数据：ギア；中文整理：Anhao。图库包含成人向分类。素材不属于代码 MIT 许可范围，不得用于商业用途；具体声明见 resources/pose-gallery/SOURCE-LICENSE.txt。"));
        const link = el("a", "", "查看原项目网页"); link.href = "https://ai-pose-gallery.anhao.app/"; link.target = "_blank"; link.rel = "noopener noreferrer";
        helpDialog.append(link, button("关闭指南", () => helpDialog.close()));
        helpDialog.onclose = () => { helpDialog.remove(); helpDialogRef = null; }; helpDialog.onkeydown = event => event.stopPropagation();
        document.body.append(helpDialog); helpDialog.showModal();
    }
    return dialog;
}

app.registerExtension({
    name: "Dapao.PoseReference",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DapaoPoseReference") return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = original?.apply(this, arguments); styles();
            const node = this, widget = node.widgets.find(w => w.name === "🧍姿势ID");
            if (!widget) return result;
            // Keep serialized ID at its original widget index; only hide its editor.
            widget.computeSize = () => [0, -4];
            widget.hidden = true;
            widget.options ??= {}; widget.options.hidden = true;
            const input = widget.inputEl || widget.element || widget.domElement;
            if (input?.style) input.style.display = "none";
            const root = el("div", "dapao-pose-node");
            const heading = el("div", "dpn-heading");
            heading.append(el("strong", "", "🧍 常用姿势"), el("small", "", "点击缩略图快速选择"));
            const grid = el("div", "dpn-grid"); grid.setAttribute("aria-label", "常用姿势");
            const selected = el("div", "dpn-selected"), label = el("span"), idLabel = el("small");
            selected.append(label, idLabel);
            const angleRow = el("div", "dpn-angles"); angleRow.setAttribute("aria-label", "快捷视角");
            const message = el("div", "dpn-message"); message.setAttribute("role", "status");
            const footer = el("small", "dpn-footer", "288 组姿势 · 1439 张参考图");
            let alive = true, revision = 0, operation = 0, busy = false, data, openDialog;
            const presetIds = ["full_single_P018_front", "full_single_P015_front", "full_single_P001_front",
                "full_single_P019_front", "full_single_P033_front", "full_single_P017_front"];
            const commit = item => {
                if (!alive) throw new Error("原节点已删除");
                node.graph?.beforeChange?.();
                try {
                    widget.value = item.id;
                    if (node.widgets_values) node.widgets_values[node.widgets.indexOf(widget)] = item.id;
                    widget.callback?.(item.id);
                } finally { node.graph?.afterChange?.(); }
                node.setDirtyCanvas(true, true);
            };
            const select = async item => {
                if (busy || !alive) return;
                const token = ++operation, previous = widget.value;
                busy = true; render(); message.textContent = "正在读取参考图…";
                try {
                    const image = new Image(); image.src = url(item); await image.decode();
                    if (!alive || token !== operation || widget.value !== previous) return;
                    commit(item);
                } catch (error) {
                    if (alive && token === operation) message.textContent = "图片读取失败，原选择已保留；可重试或打开完整图库。";
                } finally {
                    if (alive && token === operation) { busy = false; render(); }
                }
            };
            const open = () => {
                if (busy || openDialog?.isConnected) return;
                openDialog = openPoseGallery(widget.value, commit);
            };
            const openButton = button("打开完整姿势库  →", open); openButton.className = "dpn-open";
            root.append(heading, grid, selected, angleRow, message, openButton, footer);
            const controls = referenceControls(node, root, { items: () => data?.items, current: () => widget.value, select, imageUrl: url, blocked: () => busy || !!openDialog?.isConnected, textOutputs: item => [null, `${item.name}；${item.system}；${item.category}；${item.angle}；${item.number}`, item.id] });
            // One DOM flow owns all controls: canvas buttons cannot overlap the preview.
            for (const event of ["pointerdown", "keydown", "wheel"]) root.addEventListener(event, e => e.stopPropagation());
            const dom = node.addDOMWidget("pose_preview", "pose_preview", root, {
                serialize: false, getMinHeight: () => 480, getHeight: () => 480,
                hideOnZoom: false, hideInPanel: true,
            });
            dom.options ??= {}; dom.options.minNodeSize = [360, 0];
            dom.computeSize = width => [Math.max(360, width || 0), 480];
            function render() {
                if (!alive) return;
                const active = data?.items.find(i => i.id === widget.value);
                controls.update(active);
                label.textContent = active ? `已选：${active.name}` : widget.value ? "姿势 ID 无效，请重新选择" : "请选择一个姿势";
                idLabel.textContent = active ? active.id : ""; idLabel.title = active?.id || "";
                grid.replaceChildren(); angleRow.replaceChildren();
                if (data) {
                    let presets = presetIds.map(id => data.items.find(i => i.id === id)).filter(Boolean);
                    // Gallery selections also stay visible in the quick window.
                    if (active && !presets.some(i => key(i) === key(active))) presets = [active, ...presets.slice(0, 5)];
                    for (const preset of presets) {
                        const chosen = active && key(active) === key(preset);
                        const item = chosen ? active : preset;
                        const card = button("", () => select(item)); card.className = "dpn-card";
                        card.disabled = busy; card.setAttribute("aria-pressed", String(!!chosen));
                        card.setAttribute("aria-label", `${item.name} · ${item.angle}`);
                        card.title = `${item.name} · ${item.angle}`;
                        const image = el("img"); image.src = url(item); image.alt = item.name; image.draggable = false;
                        image.onerror = () => { image.alt = "素材读取失败，点击重试"; };
                        card.append(image, el("span", "dpn-caption", item.name));
                        if (chosen) card.append(el("span", "dpn-check", "✓"));
                        grid.append(card);
                    }
                }
                for (const angle of ANGLES) {
                    const item = active && data.items.find(i => key(i) === key(active) && i.angle === angle);
                    const b = button(angle, () => select(item)); b.disabled = !item || busy;
                    b.setAttribute("aria-pressed", String(active?.angle === angle));
                    b.title = !active ? "请先选择姿势" : !item ? "原图库未提供此视角" : `切换到${angle}`;
                    angleRow.append(b);
                }
                openButton.disabled = busy;
            }
            const refresh = async (reload = false) => {
                const rev = ++revision;
                message.replaceChildren();
                if (!data) { message.textContent = "正在加载常用姿势…"; render(); }
                try {
                    data = await catalog(reload);
                    if (!alive || rev !== revision) return;
                    message.textContent = ""; render();
                } catch (error) {
                    if (alive && rev === revision) {
                        message.replaceChildren(el("span", "", "目录读取失败。"), button("重试", () => refresh(true)));
                    }
                }
            };
            const callback = widget.callback; widget.callback = function () { const r = callback?.apply(this, arguments); refresh(); return r; };
            // Use the actual slot + visible-widget height; do not retain old 650px frames.
            const contentHeight = () => Math.ceil(node.computeSize?.()[1] || 558);
            const fit = () => node.setSize([Math.max(360, node.size?.[0] || 430), contentHeight()]);
            const configured = node.onConfigure; node.onConfigure = function () {
                const r = configured?.apply(this, arguments); operation++; busy = false; fit(); refresh(); return r;
            };
            const resized = node.onResize; node.onResize = function (size) {
                size[0] = Math.max(360, size[0]); size[1] = Math.max(contentHeight(), size[1]);
                return resized?.apply(this, arguments);
            };
            const removed = node.onRemoved; node.onRemoved = function () {
                alive = false; operation++; revision++; openDialog?.close(); root.remove(); return removed?.apply(this, arguments);
            };
            node.setSize([430, contentHeight()]); refresh(); return result;
        };
    },
});

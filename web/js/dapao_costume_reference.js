import { referenceControls } from "./dapao_reference_controls.js";
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const BASE = "/dapao/costume-gallery";
const imageUrl = (item, thumb = false) => api.apiURL(`${BASE}/image/${encodeURIComponent(item.id)}${thumb ? "?kind=thumb" : ""}`);
const el = (tag, cls = "", text = "") => {
    const e = document.createElement(tag); e.className = cls; e.textContent = text; return e;
};
const button = (text, action, cls = "") => {
    const b = el("button", cls, text); b.type = "button"; b.onclick = action; return b;
};
function styles() {
    for (const file of ["dapao_pose_reference", "dapao_costume_reference"]) {
        const styleId = file === "dapao_pose_reference" ? "dapao-pose-style" : `${file}-css`;
        if (document.getElementById(styleId)) continue;
        const link = el("link"); link.id = styleId; link.rel = "stylesheet";
        link.href = new URL(`./${file}.css`, import.meta.url).href; document.head.append(link);
    }
}
let pendingCatalog;
async function catalog(reload = false) {
    if (reload) pendingCatalog = null;
    if (!pendingCatalog) pendingCatalog = api.fetchApi(`${BASE}/catalog`).then(async response => {
        const data = await response.json();
        if (!response.ok) throw Error(data.error || "服装目录读取失败");
        if (data.version !== 1 || !data.items?.length) throw Error("服装目录格式无效");
        return data;
    }).catch(error => { pendingCatalog = null; throw error; });
    return pendingCatalog;
}
async function verifyImage(item) { const image = new Image(); image.src = imageUrl(item); await image.decode(); }
function languageButtons(language, change) {
    const row = el("div", "dc-language"); row.setAttribute("aria-label", "提示词语言");
    for (const value of ["中文", "English"]) {
        const b = button(value, () => change(value)); b.setAttribute("aria-pressed", String(value === language)); row.append(b);
    }
    return row;
}

export function openCostumeGallery(selectedId, initialLanguage, onSelect) {
    styles();
    const dialog = el("dialog", "dapao-pose dapao-costume"), header = el("header"), title = el("div");
    title.append(el("h2", "", "精选服装参考"), el("small", "", "1407 套服装 · 本地浏览 · 原尺寸无损输出"));
    header.append(title, button("重新读取目录", () => load(true)), button("使用说明", help), button("关闭", () => dialog.close()));
    const filters = el("section", "dp-filters"), categories = el("div", "dp-row"), row = el("div", "dp-row");
    const search = el("input"); search.placeholder = "搜索编号、中文标签或英文关键词（/）"; search.setAttribute("aria-label", "搜索服装");
    const count = el("span", "dp-count"); count.setAttribute("role", "status");
    row.append(search, button("重置筛选", () => { category = ""; search.value = ""; renderCategories(); filter(); }), count);
    filters.append(categories, row);
    const main = el("main", "dp-main"), list = el("section", "dp-list"), detail = el("aside", "dp-detail");
    list.setAttribute("aria-label", "服装列表"); detail.setAttribute("aria-label", "服装详情"); detail.tabIndex = 0;
    main.append(list, detail);
    const footer = el("footer"), status = el("span", "dp-status"), apply = button("请先选择服装", applySelection, "dp-primary");
    status.setAttribute("role", "status"); apply.disabled = true; footer.append(status, apply);
    dialog.append(header, filters, main, footer);
    let data, matches = [], category = "", shown = 30, active, language = initialLanguage || "中文", busy = false, epoch = 0, loadEpoch = 0;
    const children = new Set();
    dialog.addEventListener("close", () => { epoch++; loadEpoch++; for (const child of children) child.close(); dialog.remove(); });
    dialog.onkeydown = event => {
        event.stopPropagation(); if (event.target.closest("input,textarea,select")) return;
        if (event.key === "/") { event.preventDefault(); search.focus(); }
        if (["ArrowLeft", "ArrowRight"].includes(event.key)) { event.preventDefault(); step(event.key === "ArrowRight" ? 1 : -1); }
    };
    search.oninput = filter;
    document.body.append(dialog); dialog.showModal(); load();
    function childDialog(cls) {
        const child = el("dialog", `dapao-pose ${cls}`); children.add(child);
        child.onkeydown = e => e.stopPropagation();
        child.onclose = () => { children.delete(child); child.remove(); if (dialog.isConnected) detail.focus({preventScroll:true}); };
        document.body.append(child); return child;
    }
    async function load(reload = false) {
        const token = ++loadEpoch; list.replaceChildren(el("p", "", "正在加载服装目录…"));
        try {
            const value = await catalog(reload); if (!dialog.isConnected || token !== loadEpoch) return;
            data = value; renderCategories(); filter();
            const item = data.items.find(i => i.id === (active?.id || selectedId));
            if (item) choose(item); else detail.replaceChildren(el("p", "", "点击服装，查看无损大图、标签和提示词。"));
        } catch (error) {
            if (dialog.isConnected && token === loadEpoch) list.replaceChildren(el("p", "dp-error", error.message), button("重试读取目录", () => load(true)));
        }
    }
    function renderCategories() {
        categories.replaceChildren();
        for (const c of [{id:"", name_zh:"全部", count:data.items.length}, ...data.categories]) {
            const b = button(`${c.name_zh} · ${c.count}`, () => { category = c.id; renderCategories(); filter(); });
            b.setAttribute("aria-pressed", String(category === c.id)); categories.append(b);
        }
    }
    function filter() {
        if (!data) return;
        const terms = search.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
        matches = data.items.filter(i => (!category || i.category_id === category) && terms.every(t =>
            `${i.id} ${i.category_zh} ${i.category_en} ${i.tags_zh.join(" ")} ${i.tags_en.join(" ")}`.toLowerCase().includes(t)));
        shown = 30; count.textContent = `${matches.length} 套`; list.scrollTop = 0; drawCards();
    }
    function drawCards() {
        list.replaceChildren(); const grid = el("div", "dp-grid");
        for (const item of matches.slice(0, shown)) {
            const card = button("", () => choose(item), "dp-card dc-card"); card.dataset.id = item.id;
            card.setAttribute("aria-label", `${item.id} ${item.category_zh}`);
            const image = el("img"); image.src = imageUrl(item, true); image.loading = "lazy"; image.decoding = "async"; image.alt = `${item.id} ${item.tags_zh.slice(0,3).join("、")}`;
            image.onerror = () => { image.alt = "缩略图读取失败，点击查看原图"; };
            card.append(image, el("strong", "", `${item.id} · ${item.category_zh}`), el("small", "", item.tags_zh.slice(0,4).join(" · "))); grid.append(card);
        }
        list.append(grid);
        if (!matches.length) list.append(el("p", "", "没有找到匹配的服装，请调整分类或关键词。"));
        if (matches.length > shown) list.append(button(`加载更多（剩余 ${matches.length-shown}）`, () => { const scroll = list.scrollTop; shown += 30; drawCards(); list.scrollTop = scroll; }));
        mark();
    }
    function mark() { list.querySelectorAll(".dc-card").forEach(c => { c.classList.toggle("dp-selected", c.dataset.id === active?.id); c.setAttribute("aria-pressed", String(c.dataset.id === active?.id)); }); }
    function choose(item) {
        if (busy) return; active = item; status.textContent = ""; mark(); renderDetail(); detail.scrollTop = 0; detail.focus({preventScroll:true});
        if (innerWidth <= 760) detail.scrollIntoView({block:"start"});
    }
    function step(direction) {
        if (!matches.length || busy) return;
        const index = matches.findIndex(i => i.id === active?.id);
        choose(matches[index < 0 ? 0 : (index + direction + matches.length) % matches.length]);
    }
    function renderDetail() {
        const item = active, suffix = language === "English" ? "en" : "zh"; detail.replaceChildren();
        const image = el("img", "dp-detail-image"), imageButton = button("", () => zoom(item), "dp-photo-button"), hint = el("small", "", "正在加载原尺寸无损图…");
        imageButton.setAttribute("aria-label", "放大查看服装细节"); image.src = imageUrl(item); image.alt = `${item.id} 原尺寸无损图`; imageButton.append(image);
        image.onload = () => { hint.textContent = `${item.width} × ${item.height} · 无损 WebP · 点击放大`; };
        image.onerror = () => hint.replaceChildren(el("span", "dp-error", "图片读取失败。"), button("重试图片", () => { image.src = `${imageUrl(item)}?retry=${Date.now()}`; }));
        let touch;
        imageButton.ontouchstart = e => { touch = e.touches[0]; };
        imageButton.ontouchend = e => { const end=e.changedTouches[0]; if(touch && Math.abs(end.clientX-touch.clientX)>45 && Math.abs(end.clientY-touch.clientY)<60){e.preventDefault();step(end.clientX<touch.clientX?1:-1);}touch=null; };
        const tags = el("div", "dc-tags"); for (const tag of item.tags_zh) tags.append(el("span", "", tag));
        const languages = languageButtons(language, value => { if (busy) return; language = value; renderDetail(); detail.focus({preventScroll:true}); });
        const prompt = el("pre", "dc-prompt", item[`prompt_${suffix}`]); prompt.setAttribute("aria-label", "服装提示词");
        const negative = el("pre", "dc-prompt", item[`negative_prompt_${suffix}`]); negative.setAttribute("aria-label", "反向提示词");
        const actions = el("div", "dp-row");
        actions.append(button("复制提示词", () => copy(item[`prompt_${suffix}`])), button("复制反向提示词", () => copy(item[`negative_prompt_${suffix}`])), button("复制 ID", () => copy(item.id)), button("下载无损图", () => download(item)));
        const nav = el("div", "dp-row"); nav.append(button("上一套", () => step(-1)), button("下一套", () => step(1)));
        detail.append(el("h3", "", `${item.id} · ${item.category_zh}服装`), imageButton, hint, tags, languages, el("strong", "", "服装提示词"), prompt, el("strong", "", "反向提示词"), negative, el("small", "", item.prompt_note_zh), actions, nav);
        apply.disabled = busy; apply.textContent = `使用这套服装 · ${language}`;
    }
    async function copy(text) { try { await navigator.clipboard.writeText(text); status.textContent = "已复制"; } catch { status.textContent = "复制失败，请选中文本手动复制。"; } }
    async function download(item) {
        try {
            const response = await api.fetchApi(`${BASE}/image/${encodeURIComponent(item.id)}`); if (!response.ok) throw Error("图片读取失败");
            const objectUrl = URL.createObjectURL(await response.blob()), a = el("a"); a.href = objectUrl; a.download = `${item.id}-lossless.webp`; a.click();
            setTimeout(() => URL.revokeObjectURL(objectUrl), 1500); status.textContent = "已发起无损图下载";
        } catch (e) { status.textContent = `下载失败：${e.message}`; }
    }
    async function applySelection() {
        if (!active || busy) return;
        const item = active, selectedLanguage = language, token = ++epoch; busy = true; apply.disabled = true; apply.textContent = "正在检查原尺寸图片…";
        try { await verifyImage(item); if (!dialog.isConnected || token !== epoch) return; await onSelect(item, selectedLanguage); dialog.close(); }
        catch (e) { if (dialog.isConnected) status.textContent = `选择失败，原选择已保留：${e.message}`; }
        finally { busy = false; if (dialog.isConnected) renderDetail(); }
    }
    function zoom(item) {
        const viewer = childDialog("dp-zoom"), toolbar = el("div", "dp-row"), viewport = el("div", "dp-zoom-viewport"), image = el("img");
        image.src = imageUrl(item); image.alt = `${item.id} 服装细节`; let scale = 1;
        const resize = value => { scale = Math.max(.25,Math.min(5,value)); image.style.width = `${100*scale}%`; };
        toolbar.append(el("strong", "", item.id), button("−",()=>resize(scale-.25)), button("+",()=>resize(scale+.25)),button("适应窗口",()=>resize(1)),button("原始尺寸 1:1",()=>{scale=item.width/viewport.clientWidth;image.style.width=`${item.width}px`;}),button("关闭大图",()=>viewer.close()));
        viewport.append(image);viewer.append(toolbar,viewport);viewer.showModal();
        viewport.onwheel = e => { if(e.ctrlKey){e.preventDefault();resize(scale+(e.deltaY<0?.1:-.1));} };
    }
    function help() {
        const help = childDialog("dp-help");
        help.append(el("h2", "", "使用说明"),el("p", "", "列表使用缩略图；详情、放大和节点输出均读取原尺寸无损 WebP。支持分类、编号和中英文关键词组合搜索。左右方向键或滑动详情图切换服装，/ 搜索，Esc 关闭。"),el("p", "", "确认选择时保存服装 ID 和提示词语言；运行工作流输出图片、说明、正向和反向提示词。不调用模型。关闭或读取失败保留原选择。"),el("p", "", "服装为 AI 生成的虚构设计；中文标签和提示词根据公开关键词整理，不是原作者的原始提示词。素材自带说明允许创作参考，但禁止图片原件及加工版本再次分发。个人素材与节点代码分开保存。"));
        const a=el("a","","查看来源网站");a.href="https://ai-costume-gallery.pages.dev/";a.target="_blank";a.rel="noopener noreferrer";
        help.append(a,button("关闭说明",()=>help.close()));help.showModal();
    }
    return dialog;
}

app.registerExtension({
    name: "Dapao.CostumeReference",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DapaoCostumeReference") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments); styles();
            const node = this, idWidget = node.widgets.find(w=>w.name==="👗服装ID"), languageWidget = node.widgets.find(w=>w.name==="🌐提示词语言");
            if (!idWidget || !languageWidget) return result;
            for (const w of [idWidget,languageWidget]) { w.hidden=true;w.options??={};w.options.hidden=true;w.computeSize=()=>[0,-4];const input=w.inputEl||w.element||w.domElement;if(input?.style)input.style.display="none"; }
            const root=el("div","dapao-pose-node dapao-costume-node"),heading=el("div","dpn-heading"),grid=el("div","dpn-grid"),selected=el("div","dpn-selected"),languages=el("div"),status=el("div","dpn-message"),footer=el("small","dpn-footer","1407 套服装 · 原尺寸无损输出");
            heading.append(el("strong","","👗 常用服装"),el("small","","点击缩略图快速选择"));grid.setAttribute("aria-label","常用服装");status.setAttribute("role","status");
            let data,alive=true,busy=false,revision=0,operation=0,opened;
            function commit(item,lang) {
                if(!alive)throw Error("原节点已删除");node.graph?.beforeChange?.();
                try { idWidget.value=item.id;languageWidget.value=lang;for(const w of [idWidget,languageWidget]) {if(node.widgets_values)node.widgets_values[node.widgets.indexOf(w)]=w.value;w.callback?.(w.value);} }
                finally {node.graph?.afterChange?.();}node.setDirtyCanvas(true,true);
            }
            async function choose(item) {
                if(busy||!alive)return;const token=++operation,previous=idWidget.value;busy=true;render();status.textContent="正在检查无损图…";
                try {await verifyImage(item);if(!alive||operation!==token||idWidget.value!==previous)return;commit(item,languageWidget.value);status.textContent="";}
                catch(e){if(alive&&token===operation)status.textContent="图片读取失败，原选择已保留；点击重试。";}
                finally{if(alive&&token===operation){busy=false;render();}}
            }
            const open=()=>{
                if(busy||opened?.isConnected)return;
                const previous=idWidget.value,previousLanguage=languageWidget.value;
                opened=openCostumeGallery(previous,previousLanguage,(item,lang)=>{if(idWidget.value!==previous||languageWidget.value!==previousLanguage)throw Error("节点参数已变化，请重新打开图库");commit(item,lang);});
            };
            const openButton=button("打开完整服装库  →",open,"dpn-open");root.append(heading,grid,selected,languages,status,openButton,footer);
            const controls = referenceControls(node, root, { items: () => data?.items, current: () => idWidget.value, select: choose, imageUrl, blocked: () => busy || !!opened?.isConnected, textOutputs: item => {
                const suffix = languageWidget.value === "English" ? "en" : "zh";
                return [null, `${item.id} · ${item.category_zh}；${item.tags_zh.join("、")}`, item[`prompt_${suffix}`], item[`negative_prompt_${suffix}`], item.id];
            } });
            for(const event of ["pointerdown","keydown","wheel"])root.addEventListener(event,e=>e.stopPropagation());
            const dom=node.addDOMWidget("costume_preview","costume_preview",root,{serialize:false,getMinHeight:()=>480,getHeight:()=>480,hideOnZoom:false,hideInPanel:true});
            dom.options??={};dom.options.minNodeSize=[360,0];dom.computeSize=w=>[Math.max(360,w||0),480];
            function render() {
                if(!alive)return;const active=data?.items.find(i=>i.id===idWidget.value);grid.replaceChildren();selected.replaceChildren();controls.update(active);
                selected.append(el("span","",active?`已选：${active.id} · ${active.category_zh}`:idWidget.value?"素材 ID 无效，请重新选择":"请选择一套服装"));
                if(active)selected.append(el("small","",active.tags_zh.slice(0,5).join(" · ")));
                if(data){
                    let presets=["spring","summer","autumn","winter","party","swimwear"].map(category=>data.items.find(i=>i.category_id===category)).filter(Boolean);
                    if(active&&!presets.some(i=>i.id===active.id))presets=[active,...presets.slice(0,5)];
                    for(const item of presets){const card=button("",()=>choose(item),"dpn-card");card.disabled=busy;card.title=item.tags_zh.join(" · ");card.setAttribute("aria-label",`${item.id} ${item.category_zh}`);card.setAttribute("aria-pressed",String(item.id===active?.id));const image=el("img");image.src=imageUrl(item,true);image.alt=`${item.id} ${item.category_zh}`;image.draggable=false;image.onerror=()=>{image.alt="缩略图读取失败，点击重试";};card.append(image,el("span","dpn-caption",`${item.category_zh} · ${item.id}`));if(item.id===active?.id)card.append(el("span","dpn-check","✓"));grid.append(card);}
                }
                const row=languageButtons(languageWidget.value,value=>{if(busy)return;node.graph?.beforeChange?.();languageWidget.value=value;if(node.widgets_values)node.widgets_values[node.widgets.indexOf(languageWidget)]=value;languageWidget.callback?.(value);node.graph?.afterChange?.();node.setDirtyCanvas(true,true);});
                row.querySelectorAll("button").forEach(b=>b.disabled=busy);languages.replaceChildren(row);openButton.disabled=busy;
            }
            async function refresh(reload=false){const token=++revision;status.replaceChildren();if(!data){status.textContent="正在加载本地服装…";render();}try{const value=await catalog(reload);if(!alive||token!==revision)return;data=value;status.textContent="";render();}catch(e){if(alive&&token===revision)status.replaceChildren(el("span","",e.message),button("重试",()=>refresh(true)));}}
            for(const w of [idWidget,languageWidget]){const callback=w.callback;w.callback=function(){const r=callback?.apply(this,arguments);refresh();return r;};}
            const height=()=>Math.ceil(node.computeSize?.()[1]||598);
            const configured=node.onConfigure;node.onConfigure=function(){const r=configured?.apply(this,arguments);operation++;busy=false;node.setSize([Math.max(360,node.size?.[0]||430),height()]);refresh();return r;};
            const resized=node.onResize;node.onResize=function(size){size[0]=Math.max(360,size[0]);size[1]=Math.max(height(),size[1]);return resized?.apply(this,arguments);};
            const removed=node.onRemoved;node.onRemoved=function(){alive=false;revision++;operation++;opened?.close();root.remove();return removed?.apply(this,arguments);};
            node.setSize([430,height()]);refresh();return result;
        };
    },
});

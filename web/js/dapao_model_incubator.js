import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { openFaceGallery } from "./dapao_face_reference.js";
import { openCostumeGallery } from "./dapao_costume_reference.js";
import { openPoseGallery } from "./dapao_pose_reference.js";
import { installCanvasTitleButton, showConnectedPreview } from "./dapao_reference_controls.js";

const SINGLE = new Set(["全身单人・标准系", "全身单人・性感系", "上半身单人"]);
const kinds = [
    {key:"face", label:"脸模", emoji:"👩", open:openFaceGallery},
    {key:"costume", label:"服装", emoji:"👗", open:openCostumeGallery},
    {key:"pose", label:"姿势", emoji:"🧍", open:openPoseGallery},
];
const el = (tag, cls = "", text = "") => Object.assign(document.createElement(tag), {className:cls, textContent:text});
const button = (text, action) => { const b = el("button", "", text); b.type="button"; b.onclick=action; return b; };
function styles() {
    for (const name of ["dapao_pose_reference", "dapao_model_incubator"]) {
        const id = name === "dapao_pose_reference" ? "dapao-pose-style" : `${name}-css`;
        if (document.getElementById(id)) continue;
        const link=el("link"); link.id=id; link.rel="stylesheet"; link.href=new URL(`./${name}.css`, import.meta.url).href;
        document.head.append(link);
    }
}
app.registerExtension({
    name:"Dapao.ModelIncubator",
    async beforeRegisterNodeDef(Node, data) {
        if (data.name !== "DapaoModelIncubator") return;
        const created=Node.prototype.onNodeCreated;
        Node.prototype.onNodeCreated=function() {
            created?.apply(this, arguments); styles();
            const node=this, root=el("div", "dapao-pose-node dmi-node");
            let alive=true, busy=false, epoch=0, dialog=null, pendingRefresh=false, rollingKey=null;
            const catalogs=new Map(), rows=[];
            const getWidget=name=>node.widgets.find(w=>w.name===name);
            const get=(kind, field)=>getWidget(`${kind.emoji}${kind.label}${field}`).value;
            const put=(kind, field, value)=>{getWidget(`${kind.emoji}${kind.label}${field}`).value=value;};
            const seedWidget=getWidget("🎲提示词种子");
            // Serialized native widgets remain the source of truth; the compact UI edits them.
            for (const w of [...node.widgets]) {
                if (w === seedWidget || kinds.some(k=>w.name.startsWith(`${k.emoji}${k.label}`))) {
                    w.type="dmi-hidden"; w.computeSize=()=>[0,-4];
                    if(w.options)w.options.hidden=true;
                }
            }
            const title=installCanvasTitleButton(node);
            const foreground=node.onDrawForeground;
            node.onDrawForeground=function(ctx) {
                foreground?.apply(this,arguments);
                if(title.titleButton || this.flags?.collapsed)return;
                ctx.save();ctx.fillStyle="#244b70";ctx.strokeStyle="#71a9dc";ctx.beginPath();ctx.roundRect(...title.linkBox(),5);ctx.fill();ctx.stroke();
                ctx.fillStyle="#f0f7ff";ctx.font='12px "Microsoft YaHei"';ctx.textAlign="left";ctx.textBaseline="middle";ctx.fillText("无限画布 ↗",title.linkBox()[0]+6,-16);ctx.restore();
            };
            const down=node.onMouseDown;
            node.onMouseDown=function(e,p) {
                const b=title.linkBox();
                if(!title.titleButton&&!this.flags?.collapsed&&(e.button===undefined||e.button===0)&&p&&p[0]>=b[0]&&p[0]<=b[0]+b[2]&&p[1]>=b[1]&&p[1]<=b[1]+b[3]){title.openCanvas();return true;}
                return down?.apply(this,arguments);
            };
            const toolbar=el("div","dmi-toolbar"), random=button("",()=>act(randomize,"all"));
            const dice=el("span","dmi-dice","🎲"); random.append(dice,document.createTextNode(" 随机组合"));random.title="随机图库位置，保留上传图片；同时变换提示词";
            const preview=button("更新下游预览",()=>act(()=>refresh(true)));
            toolbar.append(random,preview);
            root.append(el("strong","","三图组合 · 单人模特"),el("small","","1 脸模 → 2 服装 → 3 姿势 · 上传图片不参与随机"),toolbar);
            const grid=el("div","dmi-grid");root.append(grid);
            for(const [index,kind] of kinds.entries()) {
                const card=el("section","dmi-card"), image=el("img"), label=el("small","dmi-selection");image.alt=`图片${index+1} ${kind.label}参考`;
                const source=el("select");source.setAttribute("aria-label",`${kind.label}来源`);
                for(const mode of ["素材库","上传图片"])source.add(new Option(mode,mode));
                source.onchange=()=>{const value=source.value;void act(async()=>{change(()=>put(kind,"来源",value));render();await refresh(true);});};
                const choose=button("打开素材库",()=>openGallery(kind));
                const upload=button("上传图片",()=>file.click());
                const file=el("input");file.type="file";file.accept="image/png,image/jpeg,image/webp";file.hidden=true;
                file.onchange=()=>{const selected=file.files?.[0];file.value="";if(selected)void act(()=>uploadFile(kind,selected));};
                image.onerror=()=>{image.hidden=true;label.textContent="图片无法读取，请重新选择或上传";};
                const heading=el("div","dmi-card-heading"), roll=button("",()=>act(()=>randomize(kind),kind.key));
                roll.className="dmi-single-roll";roll.setAttribute("aria-label",`随机${kind.label}`);
                roll.append(el("span","dmi-dice","🎲"));
                heading.append(el("b","",`${kind.emoji} ${index+1} · ${kind.label}`),roll);
                card.append(heading,source,image,label,choose,upload,file);grid.append(card);
                rows.push({kind,source,image,label,choose,upload,roll});
            }
            const prompt=el("textarea","dmi-prompt");prompt.readOnly=true;prompt.setAttribute("aria-label","整合提示词");prompt.placeholder="选择三张素材后，可连接整合提示词输出使用";
            const copy=button("复制提示词",async()=>{try{await navigator.clipboard.writeText(prompt.value);status.textContent="提示词已复制";}catch{prompt.focus();prompt.select();status.textContent="请按 Ctrl+C 复制选中的提示词";}});
            const status=el("div","dpn-message");status.setAttribute("role","status");
            root.append(el("b","","✨ 整合提示词"),prompt,copy,status,el("small","","输出三张独立参考图；成片需连接生图节点后手动运行。自传姿势请使用单人图。"));
            for(const event of ["pointerdown","keydown","wheel"])root.addEventListener(event,e=>e.stopPropagation());
            const dom=node.addDOMWidget("model_incubator","model_incubator",root,{serialize:false,getMinHeight:()=>670,getHeight:()=>670,hideOnZoom:false,hideInPanel:true});
            dom.options??={};dom.options.minNodeSize=[560,0];dom.computeSize=w=>[Math.max(560,w||0),670];node.setSize([Math.max(node.size[0],560),Math.max(node.size[1],800)]);
            const configure=node.onConfigure;
            node.onConfigure=function(){epoch++;const result=configure?.apply(this,arguments);queueMicrotask(()=>{if(alive){render();if(busy)pendingRefresh=true;else void act(()=>refresh(false));}});return result;};
            const removed=node.onRemoved;
            node.onRemoved=function(){alive=false;epoch++;dialog?.close();return removed?.apply(this,arguments);};

            function url(kind) {
                if(get(kind,"来源")==="上传图片") {
                    const name=get(kind,"上传");if(!name)return "";
                    const normalized=name.replaceAll("\\","/"), pos=normalized.lastIndexOf("/");
                    return api.apiURL(`/view?${new URLSearchParams({filename:normalized.slice(pos+1),subfolder:pos<0?"":normalized.slice(0,pos),type:"input"})}`);
                }
                const id=get(kind,"ID");return id?api.apiURL(`/dapao/${kind.key}-gallery/image/${encodeURIComponent(id)}`):"";
            }
            function render() {
                if(!alive)return;
                random.disabled=preview.disabled=busy;random.classList.toggle("dmi-rolling",busy&&rollingKey==="all");
                for(const row of rows) {
                    const {kind,source,image,label,choose,upload,roll}=row, uploaded=get(kind,"来源")==="上传图片";
                    roll.disabled=busy||uploaded;
                    roll.title=uploaded?"上传图片保持不变；切换为素材库后可随机":`只随机${kind.label}，其他素材和提示词保持不变`;
                    roll.classList.toggle("dmi-rolling",busy&&rollingKey===kind.key);
                    source.value=get(kind,"来源");source.disabled=choose.disabled=upload.disabled=busy;
                    choose.hidden=uploaded;upload.hidden=!uploaded;
                    const src=url(kind);image.hidden=!src;if(src&&image.getAttribute("src")!==src)image.src=src;if(!src)image.removeAttribute("src");
                    label.textContent=(uploaded?get(kind,"上传"):get(kind,"ID"))||"尚未选择";label.title=label.textContent;
                }
                node.setDirtyCanvas?.(true,true);
            }
            async function act(action, rollKey=null) {
                if(!alive||busy||dialog?.isConnected)return;
                busy=true;rollingKey=rollKey;render();status.textContent="正在更新…";
                try{await action();}catch(error){if(alive)status.textContent=`操作失败：${error.message}`;}
                finally{busy=false;rollingKey=null;render();if(pendingRefresh&&alive){pendingRefresh=false;void act(()=>refresh(false));}}
            }
            function change(action) {
                node.graph?.beforeChange?.();
                try{action();}finally{node.graph?.afterChange?.();}
            }
            async function catalog(kind) {
                if(catalogs.has(kind.key))return catalogs.get(kind.key);
                const response=await api.fetchApi(`/dapao/${kind.key}-gallery/catalog`), data=await response.json();
                if(!response.ok)throw Error(data.error||`${kind.label}库读取失败`);
                const list=kind.key==="pose"?data.items.filter(i=>SINGLE.has(i.system)):data.items;
                if(!list?.length)throw Error(`${kind.label}库没有可用素材`);catalogs.set(kind.key,list);return list;
            }
            async function randomize(onlyKind=null) {
                const token=epoch;
                const changes=await Promise.all((onlyKind?[onlyKind]:kinds).map(async kind=>{
                    if(get(kind,"来源")==="上传图片")return null;
                    const list=await catalog(kind), other=list.filter(i=>i.id!==get(kind,"ID")), pool=other.length?other:list;
                    return [kind,pool[Math.floor(Math.random()*pool.length)].id];
                }));
                await new Promise(resolve=>setTimeout(resolve,480));
                if(!alive||epoch!==token)return;
                node.graph?.beforeChange?.();
                try{for(const change of changes)if(change)put(change[0],"ID",change[1]);if(!onlyKind)seedWidget.value=(Number(seedWidget.value)+1+Math.floor(Math.random()*8))%2147483647;}
                finally{node.graph?.afterChange?.();}
                render();await refresh(true,onlyKind?kinds.indexOf(onlyKind):null);
            }
            function openGallery(kind) {
                if(busy||!alive||dialog?.isConnected)return;
                const token=epoch;
                const apply=item=>{
                    if(!alive||token!==epoch)throw Error("节点状态已变化，请重新打开图库");
                    if(kind.key==="pose"&&!SINGLE.has(item.system))throw Error("请选择单人姿势");
                    change(()=>{put(kind,"ID",item.id);put(kind,"来源","素材库");});render();
                    setTimeout(()=>{if(alive)void act(()=>refresh(true));},0);
                };
                dialog=kind.key==="pose"?kind.open(get(kind,"ID"),apply,{itemFilter:i=>SINGLE.has(i.system)}):kind.open(get(kind,"ID"),"中文",apply);
            }
            async function uploadFile(kind,file) {
                if(file.size>30*1024*1024)throw Error("请选择不超过 30MB 的图片");
                const token=epoch, form=new FormData();form.append("image",file);form.append("type","input");form.append("subfolder","dapao-model-incubator");form.append("overwrite","false");
                const response=await api.fetchApi("/upload/image",{method:"POST",body:form});
                if(!response.ok)throw Error(`上传失败 (${response.status})`);
                const result=await response.json();if(!alive||token!==epoch)return;
                change(()=>{put(kind,"上传",[result.subfolder,result.name].filter(Boolean).join("/"));put(kind,"来源","上传图片");});render();await refresh(true);
            }
            function targets(slot) {
                return (node.outputs?.[slot]?.links||[]).map(id=>{
                    const edge=node.graph?.links?.get?.(id)||node.graph?.links?.[id];
                    if(!edge||edge.origin_id!==node.id||edge.origin_slot!==slot)return null;
                    return {edge,target:node.graph.getNodeById(edge.target_id)};
                }).filter(v=>v?.target);
            }
            async function refresh(autoCreate, onlySlot=null) {
                const token=epoch, seed=seedWidget.value;
                const response=await api.fetchApi(`/dapao/model-incubator/prompt?seed=${encodeURIComponent(seed)}`);
                const result=await response.json();if(!response.ok)throw Error(result.error||"提示词读取失败");
                if(!alive||token!==epoch)return;prompt.value=result.prompt;
                const skipped=new Set();
                for(const {target,edge} of targets(3)) {
                    if(edge.target_slot===0&&["PreviewAny","ShowText|pysssss","easy showAnything"].includes(target.comfyClass||target.type)&&target.onExecuted)target.onExecuted({text:[result.prompt]});
                    else skipped.add(target.title||target.type);
                }
                const complete=kinds.every(kind=>!!url(kind));
                const shouldCreate=autoCreate&&complete&&!node.outputs?.some(o=>o.links?.length);
                const images=await Promise.all(kinds.map(async (kind,slot)=>{if(onlySlot!==null&&slot!==onlySlot&&!shouldCreate)return null;const src=url(kind);if(!src)return null;const image=new Image();image.src=src;await image.decode();return image;}));
                if(!alive||token!==epoch)return;
                if(shouldCreate&&!node.outputs?.some(o=>o.links?.length))createPreviews();
                for(let slot=0;slot<3;slot++)if(images[slot])for(const {target,edge} of targets(slot)) {
                    if(edge.target_slot===0&&(target.comfyClass||target.type)==="PreviewImage")showConnectedPreview(target,images[slot]);
                    else skipped.add(target.title||target.type);
                }
                status.textContent=!complete?"请补齐三张素材；已选素材可单独预览。":skipped.size?`组合已更新；${[...skipped].join("、")}需手动运行工作流。`:"组合已就绪，参考图与提示词已更新。";
            }
            function createPreviews() {
                const graph=node.graph;if(!graph)throw Error("节点已离开画布");const added=[];
                graph.beforeChange?.();
                try {
                    for(let slot=0;slot<3;slot++) {
                        const target=globalThis.LiteGraph?.createNode("PreviewImage");if(!target)throw Error("无法创建预览图像，请刷新页面后重试");
                        target.title=`图片${slot+1} · ${kinds[slot].label}`;target.size=[320,360];
                        const x=node.pos[0]+node.size[0]+80;let y=node.pos[1]+slot*420;
                        const others=graph.nodes||graph._nodes||[];
                        for(let attempt=0;attempt<=others.length;attempt++) {
                            const overlap=others.find(n=>n!==node&&n.pos&&n.size&&x<n.pos[0]+n.size[0]+24&&x+344>n.pos[0]&&y<n.pos[1]+n.size[1]+40&&y+400>n.pos[1]);
                            if(!overlap)break;y=overlap.pos[1]+overlap.size[1]+60;
                        }
                        target.pos=[x,y];graph.add(target);added.push(target);if(!node.connect(slot,target,0))throw Error("自动连线失败");
                    }
                }catch(error){for(const target of added)if(target.graph===graph)graph.remove(target);throw error;}
                finally{graph.afterChange?.();}
            }
            render();void act(()=>refresh(false));
        };
    },
});

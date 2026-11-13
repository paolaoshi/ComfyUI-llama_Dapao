import { app } from "../../../scripts/app.js";
import { createPosterChoice, closePosterPicker } from "./dapao_poster_picker.js";
import { installCanvasTitleButton } from "./dapao_reference_controls.js";

const groups = [["用途与视觉", ["🛠️创作模式", "🧲参考模式", "📌海报用途", "🎨视觉风格", "🌈配色"]], ["文字与排版", ["🔤标题字形", "📐版式结构", "💬文案方式", "🔒锁定原文"]], ["参考图用途", ["📎参考1用途", "📎参考2用途", "📎参考3用途", "📎参考4用途", "📎参考5用途", "📎参考6用途"]]];
const element = (tag, text) => Object.assign(document.createElement(tag), {textContent:text || ""});
app.registerExtension({
    name:"Dapao.PosterOptions",
    async beforeRegisterNodeDef(Node, data) {
        if (data.name !== "DapaoPosterOptions") return;
        const created = Node.prototype.onNodeCreated;
        Node.prototype.onNodeCreated = function () {
            created?.apply(this, arguments);
            const node=this;
            const root=element("div"); root.className="dapao-poster-options";
            root.style.cssText="box-sizing:border-box;padding:16px;background:#151e2c;color:#e6edf7;border:1px solid #39516e;border-radius:12px;font:14px 'Microsoft YaHei',sans-serif;height:100%;overflow:auto";
            const native=new Map(node.widgets.map(w=>[w.name,w]));
            for(const [,keys] of groups) for(const key of keys) {
                const w=native.get(key); if(!w)continue;
                w.type="dapao-poster-hidden";w.computeSize=()=>[0,-4];
                if(w.options) w.options.hidden=true;
            }
            posterBadge(node);
            const tabs=element("div");tabs.setAttribute("role","tablist");tabs.style.cssText="display:flex;gap:6px;margin-bottom:16px";
            const body=element("div");body.setAttribute("role","tabpanel");
            const summary=element("div");summary.style.cssText="padding:10px;margin-top:14px;background:#233249;border-radius:8px;line-height:1.7;font-size:12px;color:#aacaf1";
            const status=element("div");status.setAttribute("role","status");status.style.cssText="margin-top:10px;color:#9baabe;font-size:12px;line-height:1.7";
            root.append(tabs,body,summary,status);
            let active=0;
            const controls=[];
            const buttons=groups.map(([label],index)=>{
                const b=element("button",label);b.type="button";b.setAttribute("role","tab");
                b.style.cssText="flex:1;padding:10px 6px;border:1px solid #426487;border-radius:8px;color:#e7f1ff;cursor:pointer";
                b.onclick=()=>{active=index;node.properties??={};node.properties.poster_tab=index;render();};tabs.append(b);return b;
            });
            function updateSummary(){summary.textContent=["📌海报用途","🎨视觉风格","📐版式结构","💬文案方式"].map(k=>`${k}：${native.get(k)?.value}`).join(" · ");}
            function referenceNode(slot){const index=node.inputs?.findIndex(input=>input.name===`🖼️参考图${slot}`);return index>=0?node.getInputNode(index):null;}
            function referenceState(slot){const n=referenceNode(slot);return !n?'未连接':n.mode===4?'已旁路':n.mode===2?'已停用':'已启用';}
            function referenceSwitch(slot){
                const n=referenceNode(slot), row=element('div');row.style.cssText='display:flex;align-items:center;gap:8px;margin:6px 0 12px;font-size:12px';
                row.append(element('span',`参考${slot} · ${referenceState(slot)}`));
                if(n){const b=element('button',n.mode===0?'停用':'启用');b.type='button';b.style.cssText='margin-left:auto;border:1px solid #476b90;border-radius:5px;background:#203b56;color:#daeaff;padding:4px 12px;cursor:pointer';b.onclick=()=>{node.graph?.beforeChange?.();n.mode=n.mode===0?4:0;node.graph?.afterChange?.();node.setDirtyCanvas?.(true,true);render();};row.append(b);}
                return row;
            }
            function render(){
                closePosterPicker(root);
                body.replaceChildren();controls.length=0;
                buttons.forEach((b,i)=>{b.setAttribute("aria-selected",String(i===active));b.style.background=i===active?"#23558b":"#1c2a3c";});
                const manual=native.get('🛠️创作模式')?.value==='手动直通';
                const strong=native.get('🧲参考模式')?.value==='强参考';
                const notice=element('div',manual?'手动直通：原文直接出图。':strong?'强参考：按每张图的用途还原。':'选择用途与画风，再设置文案和布局；六张参考图均可选。');
                notice.style.cssText='padding:9px;margin-bottom:12px;border-radius:6px;background:#20374e;color:#b9dcff;font-size:12px;line-height:1.6';body.append(notice);

                for(const key of groups[active][1]){
                    const w=native.get(key);if(!w)continue;
                    const label=element("label",key);label.style.cssText="display:block;margin-bottom:14px;color:#c9dbf2";
                    const change=(value)=>{
                        node.graph?.beforeChange?.();w.value=value;w.callback?.(value);
                        node.graph?.afterChange?.();node.setDirtyCanvas?.(true,true);
                        status.textContent="选项已保存到当前画布；点击运行后才生成。";render();
                    };
                    if(active===2)body.append(referenceSwitch(Number(key.match(/\d+/)[0])));
                    let control;
                    if(key==="🔒锁定原文"){
                        if(!/^(全部锁字|部分锁字)/.test(native.get("💬文案方式").value))continue;
                        control=element("textarea");control.value=w.value||"";control.rows=6;control.placeholder="只写要印的文字，一行一项。可以仅填原稿，另有层级与排版要求时写在需求框。";
                        control.setAttribute("aria-label",key);control.style.cssText="width:100%;box-sizing:border-box;margin-top:6px;background:#101a29;color:#eef5ff;padding:10px;border:1px solid #456284;border-radius:6px";
                        control.oninput=()=>{w.value=control.value;node.graph?.change?.();};
                    }else control=createPosterChoice(key,w,change);
                    label.append(control);body.append(label);controls.push(control);
                }
                const hint=element("div",active===2?"参考图通过视觉模型提取特征。启用后生效，纯文字模式忽略参考。":"缩略图是设计方向示意。菜单价格、日期与联系方式请填写原文；需要精确印刷文字时可选留白后期加字。");
                if(active===1 && native.get("💬文案方式")?.value==="AI 儿童科普短文"){
                    hint.textContent="科普扩写适合制作草稿。用于作业时，可在需求里提供课本要点；已核实的正文可用「全部锁字」排版，并在成图中核对。";
                }
                hint.style.cssText="font-size:12px;color:#a8b7ca;line-height:1.7";body.append(hint);updateSummary();
            }
            for(const event of ["pointerdown","keydown","wheel"])root.addEventListener(event,e=>e.stopPropagation());
            const dom=node.addDOMWidget("poster_options","poster_options",root,{serialize:false,getMinHeight:()=>480,getHeight:()=>480,hideOnZoom:false,hideInPanel:true});
            dom.computeSize=()=>[560,480];node.setSize([600,560]);
            // Name-based state survives frontend widget-to-input conversion, which can
            // shift positional widget arrays when a saved graph is opened again.
            const serialize=node.onSerialize;
            node.onSerialize=function(data){
                serialize?.apply(this,arguments);
                data.poster_values=Object.fromEntries(groups.flatMap(([,keys])=>keys).map(key=>[key,native.get(key)?.value]));
            };
            const configure=node.onConfigure;
            node.onConfigure=function(data){
                const result=configure?.apply(this,arguments);

                active=Math.min(groups.length-1,Math.max(0,Number(node.properties?.poster_tab)||0));
                const saved=data?.poster_values || data?.widgets_values_named;
                queueMicrotask(()=>{
                    if(saved)for(const [,keys]of groups)for(const key of keys){
                        if(Object.prototype.hasOwnProperty.call(saved,key)&&native.has(key))native.get(key).value=saved[key];
                    }
                    render();
                });return result;
            };
            const removed=node.onRemoved;
            node.onRemoved=function(){closePosterPicker(root);return removed?.apply(this,arguments);};
            render();
        };
    },
});

function posterBadge(node){
            const title=installCanvasTitleButton(node);
            const draw=node.onDrawForeground;
            node.onDrawForeground=function(ctx){
                draw?.apply(this,arguments); if(title.titleButton||this.flags?.collapsed)return;
                ctx.save();ctx.fillStyle="#244b70";ctx.strokeStyle="#71a9dc";ctx.beginPath();ctx.roundRect(...title.linkBox(),5);ctx.fill();ctx.stroke();
                ctx.fillStyle="#fff";ctx.font='12px "Microsoft YaHei"';ctx.textAlign="left";ctx.fillText("无限画布 ↗",title.linkBox()[0]+6,-12);ctx.restore();
            };
            const down=node.onMouseDown;
            node.onMouseDown=function(e,p){const b=title.linkBox();if(!title.titleButton&&!this.flags?.collapsed&&e.button===0&&p&&p[0]>=b[0]&&p[0]<=b[0]+b[2]&&p[1]>=b[1]&&p[1]<=b[1]+b[3]){title.openCanvas();return true;}return down?.apply(this,arguments);};
}
app.registerExtension({name:"Dapao.PosterDirectorBrand",beforeRegisterNodeDef(Node,data){if(data.name!=="DapaoPosterDirector")return;const orig=Node.prototype.onNodeCreated;Node.prototype.onNodeCreated=function(){orig?.apply(this,arguments);posterBadge(this);};}});

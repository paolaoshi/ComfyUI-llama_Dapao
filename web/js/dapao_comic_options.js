import { app } from "../../../scripts/app.js";
import { createComicChoice, closeComicPicker } from "./dapao_comic_picker.js";
import { installCanvasTitleButton } from "./dapao_reference_controls.js";

const groups = [
    ["风格与题材", ["🛠️创作模式", "🧲参考模式", "🎨漫画风格", "📚故事题材", "🌈色彩"]],
    ["分镜与节奏", ["🧩分格方式", "🔢格数", "📖阅读方向", "🎬镜头节奏", "💥视觉表现"]],
    ["对白与字效", ["💬文字方式", "🔤对白字形", "✨强调字效", "↕️文字排布", "📚叙事走向", "🗨️气泡表现"]],
    ["参考与角色", Array.from({length:5},(_,i)=>[`📎参考${i+2}用途`,`🏷️参考${i+2}称呼`]).flat()],
];
const element = (tag, text) => Object.assign(document.createElement(tag), {textContent:text || ""});
app.registerExtension({
    name:"Dapao.ComicOptions",
    async beforeRegisterNodeDef(Node, data) {
        if (data.name !== "DapaoComicOptions") return;
        const created = Node.prototype.onNodeCreated;
        Node.prototype.onNodeCreated = function () {
            created?.apply(this, arguments);
            const node=this;
            const updateLegacyTitle=()=>{if(!node.title || /漫画创作选项/.test(node.title))node.title='漫画创造大师@炮老师的小课堂';};
            updateLegacyTitle();
            const root=element("div"); root.className="dapao-comic-options";
            root.style.cssText="box-sizing:border-box;padding:16px;background:#151e2c;color:#e6edf7;border:1px solid #39516e;border-radius:12px;font:14px 'Microsoft YaHei',sans-serif;height:100%;overflow:auto";
            const native=new Map(node.widgets.map(w=>[w.name,w]));
            for(const [,keys] of groups) for(const key of keys) {
                const w=native.get(key); if(!w)continue;
                w.type="dapao-comic-hidden";w.computeSize=()=>[0,-4];
                if(w.options) w.options.hidden=true;
            }
            const title=installCanvasTitleButton(node);
            const draw=node.onDrawForeground;
            node.onDrawForeground=function(ctx){
                draw?.apply(this,arguments); if(title.titleButton||this.flags?.collapsed)return;
                ctx.save();ctx.fillStyle="#244b70";ctx.strokeStyle="#71a9dc";ctx.beginPath();ctx.roundRect(...title.linkBox(),5);ctx.fill();ctx.stroke();
                ctx.fillStyle="#fff";ctx.font='12px "Microsoft YaHei"';ctx.textAlign="left";ctx.fillText("无限画布 ↗",title.linkBox()[0]+6,-12);ctx.restore();
            };
            const down=node.onMouseDown;
            node.onMouseDown=function(e,p){const b=title.linkBox();if(!title.titleButton&&!this.flags?.collapsed&&e.button===0&&p&&p[0]>=b[0]&&p[0]<=b[0]+b[2]&&p[1]>=b[1]&&p[1]<=b[1]+b[3]){title.openCanvas();return true;}return down?.apply(this,arguments);};
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
                b.onclick=()=>{active=index;node.properties??={};node.properties.comic_tab=index;render();};tabs.append(b);return b;
            });
            function updateSummary(){summary.textContent=["🧲参考模式","🎨漫画风格","🧩分格方式","🔢格数"].map(k=>`${k}：${native.get(k)?.value}`).join(" · ");}
            function referenceNode(slot){const index=node.inputs?.findIndex(input=>input.name===`🖼️参考图${slot}`);return index>=0?node.getInputNode(index):null;}
            function referenceState(slot){const n=referenceNode(slot);return !n?'未连接':n.mode===4?'已旁路':n.mode===2?'已停用':'已启用';}
            function referenceSwitch(slot){
                const n=referenceNode(slot), row=element('div');row.style.cssText='display:flex;align-items:center;gap:8px;margin:6px 0 12px;font-size:12px';
                row.append(element('span',`参考${slot}${slot===1?' · 画风（固定）':''} · ${referenceState(slot)}`));
                if(n){const b=element('button',n.mode===0?'停用':'启用');b.type='button';b.style.cssText='margin-left:auto;border:1px solid #476b90;border-radius:5px;background:#203b56;color:#daeaff;padding:4px 12px;cursor:pointer';b.onclick=()=>{node.graph?.beforeChange?.();n.mode=n.mode===0?4:0;node.graph?.afterChange?.();node.setDirtyCanvas?.(true,true);render();};row.append(b);}
                return row;
            }
            function render(){
                closeComicPicker(root);
                body.replaceChildren();controls.length=0;
                buttons.forEach((b,i)=>{b.setAttribute("aria-selected",String(i===active));b.style.background=i===active?"#23558b":"#1c2a3c";});
                const manual=native.get('🛠️创作模式')?.value==='手动直通';
                const strong=native.get('🧲参考模式')?.value==='强参考';
                const notice=element('div',manual?'手动直通：参考图及导演选项不参与生成。':strong?'强参考：参考1决定画风；其他图按用途保留角色、场景或布局。具体剧情与修改要求以你的文字为准。':'自动参考可结合选项创作；需要以图片为主时选择“强参考”，再到“参考与角色”设置用途。');
                notice.style.cssText='padding:9px;margin-bottom:12px;border-radius:6px;background:#20374e;color:#b9dcff;font-size:12px;line-height:1.6';body.append(notice);
                if(active===3)body.append(referenceSwitch(1));
                for(const key of groups[active][1]){
                    const w=native.get(key);if(!w)continue;
                    const label=element("label",key);label.style.cssText="display:block;margin-bottom:14px;color:#c9dbf2";
                    const change=(value)=>{
                        node.graph?.beforeChange?.();w.value=value;w.callback?.(value);
                        if(key==="🧩分格方式"&&["四格短篇","单幅跨格大场面"].includes(value))native.get("🔢格数").value=value==="四格短篇"?4:1;
                        node.graph?.afterChange?.();node.setDirtyCanvas?.(true,true);
                        status.textContent="选项已保存到当前画布；点击运行后才生成。";render();
                    };
                    let control;
                    if(active===3&&key.startsWith('📎'))body.append(referenceSwitch(Number(key.match(/\d+/)[0])));
                    if(key==="🔢格数"){
                        control=element("input");control.type="number";control.min="0";control.step="1";control.value=w.value;control.setAttribute("aria-label",key);
                        control.style.cssText="display:block;box-sizing:border-box;width:100%;margin-top:7px;padding:12px;background:#0e1724;color:#f1f6ff;border:1px solid #456284;border-radius:7px;font:14px 'Microsoft YaHei'";
                        control.onchange=()=>{const value=Number(control.value);if(!Number.isSafeInteger(value)||value<0){control.value=w.value;status.textContent="格数请输入非负整数；0表示自动。";return;}change(value);};
                    }else if(key.startsWith('🏷️')){
                        control=element('input');control.type='text';control.value=w.value||'';control.placeholder='例如：男主林川、女主阿月、古城、主角的剑';control.setAttribute('aria-label',key);control.style.cssText='width:100%;box-sizing:border-box;padding:9px;margin-top:6px;background:#0e1724;color:#edf4ff;border:1px solid #405d7f;border-radius:6px';control.onchange=()=>change(control.value);
                    }else control=createComicChoice(key,w,change);
                    label.append(control);body.append(label);controls.push(control);
                }
                const hint=element("div",active===0?"智能扩写补剧情；忠实整理保留故事。强参考有画风图时，漫画风格和色彩下拉值不覆盖它；要改为彩色可直接写在需求里。":active===1?"0＝按剧情自动决定格数；可输入12、16、24等。强参考有布局图时优先取其结构。选项不会自动更改尺寸。":active===3?"参考1固定画风，可不用。参考2～6用途自由，可分别放男女主、多人角色、场景、布局、道具或动作。图号固定对应入口，即使中间旁路也不会错位。上传图片后启用；强参考通过文字提取特征，不是精确锁脸。":"字形由MOSS绘制，不是加载电脑字体，不能保证字体精确复现。无字/留白模式不因字效添字；手动直通保持原文。参考用途在“参考与角色”设置。");
                hint.style.cssText="font-size:12px;color:#a8b7ca;line-height:1.7";body.append(hint);updateSummary();
            }
            for(const event of ["pointerdown","keydown","wheel"])root.addEventListener(event,e=>e.stopPropagation());
            const dom=node.addDOMWidget("comic_options","comic_options",root,{serialize:false,getMinHeight:()=>480,getHeight:()=>480,hideOnZoom:false,hideInPanel:true});
            dom.computeSize=()=>[560,480];node.setSize([600,560]);
            // Name-based state survives frontend widget-to-input conversion, which can
            // shift positional widget arrays when a saved graph is opened again.
            const serialize=node.onSerialize;
            node.onSerialize=function(data){
                serialize?.apply(this,arguments);
                data.comic_values=Object.fromEntries(groups.flatMap(([,keys])=>keys).map(key=>[key,native.get(key)?.value]));
            };
            const configure=node.onConfigure;
            node.onConfigure=function(data){
                const result=configure?.apply(this,arguments);
                updateLegacyTitle();
                active=Math.min(groups.length-1,Math.max(0,Number(node.properties?.comic_tab)||0));
                const saved=data?.comic_values || data?.widgets_values_named;
                queueMicrotask(()=>{
                    if(saved)for(const [,keys]of groups)for(const key of keys){
                        if(Object.prototype.hasOwnProperty.call(saved,key)&&native.has(key))native.get(key).value=saved[key];
                    }
                    render();
                });return result;
            };
            const removed=node.onRemoved;
            node.onRemoved=function(){closeComicPicker(root);return removed?.apply(this,arguments);};
            render();
        };
    },
});

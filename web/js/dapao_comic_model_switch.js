import { app } from '../../../scripts/app.js';
import { installCanvasTitleButton } from './dapao_reference_controls.js';

app.registerExtension({
    name: 'dapao.comic.pro-model-switch',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== 'DapaoComicModelSwitch') return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            created?.apply(this, arguments);
            const node = this, widget = node.widgets.find(w => w.name === '模式');
            if (!widget) return;
            widget.type = 'dapao-comic-hidden'; widget.computeSize = () => [0, -4];
            widget.options.hidden = true;
            const root = document.createElement('div');
            root.style.cssText = 'box-sizing:border-box;padding:14px;background:#151e2c;color:#e6edf7;border:1px solid #39516e;border-radius:10px;font:14px Microsoft YaHei,sans-serif;height:100%';
            const row = document.createElement('div'); row.setAttribute('role','group'); row.setAttribute('aria-label','PRO模型模式');
            row.style.cssText='display:flex;gap:8px';
            const modes=['1 普通版本','2 真人优化','3 社区狂野优化'];
            const buttons = modes.map(value => {
                const b=document.createElement('button'); b.type='button'; b.textContent=value;
                b.style.cssText='flex:1;min-height:48px;border:1px solid #547da4;border-radius:8px;color:#edf5ff;cursor:pointer;font:14px Microsoft YaHei,sans-serif';
                b.onclick=()=>{node.graph?.beforeChange?.();widget.value=value;widget.callback?.(value);node.graph?.afterChange?.();render();node.setDirtyCanvas?.(true,true);};
                row.append(b);return b;
            });
            const status=document.createElement('div');status.setAttribute('role','status');status.style.cssText='margin-top:12px;line-height:1.7;color:#b4c7dc;font-size:12px';
            function render(){buttons.forEach((b,i)=>{const selected=widget.value===modes[i];b.setAttribute('aria-pressed',String(selected));b.style.background=selected?'#235b91':'#1c2a3c';});status.textContent=`当前：${widget.value}。点击运行后生效，仅执行所选模型分支；画风与色彩仍由漫画创造大师设置。`;}
            root.append(row,status);render();
            node.addDOMWidget('pro_model_buttons','div',root,{serialize:false,getValue:()=>widget.value,setValue:()=>render()});
            const configured=node.onConfigure;node.onConfigure=function(){configured?.apply(this,arguments);render();};
            const title=installCanvasTitleButton(node), draw=node.onDrawForeground, down=node.onMouseDown;
            node.onDrawForeground=function(ctx){draw?.apply(this,arguments);if(title.titleButton||this.flags?.collapsed)return;ctx.save();ctx.fillStyle='#244b70';ctx.strokeStyle='#71a9dc';ctx.beginPath();ctx.roundRect(...title.linkBox(),5);ctx.fill();ctx.stroke();ctx.fillStyle='#fff';ctx.font='12px Microsoft YaHei';ctx.fillText('无限画布 ↗',title.linkBox()[0]+6,-12);ctx.restore();};
            node.onMouseDown=function(e,p){const b=title.linkBox();if(!title.titleButton&&!this.flags?.collapsed&&e.button===0&&p&&p[0]>=b[0]&&p[0]<=b[0]+b[2]&&p[1]>=b[1]&&p[1]<=b[1]+b[3]){title.openCanvas();return true;}return down?.apply(this,arguments);};
            node.setSize([590,180]);
        };
    }
});

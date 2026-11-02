import { comicPreviews } from './dapao_comic_preview_data.js';

const css = document.createElement('link');
css.rel = 'stylesheet';
css.href = new URL('./dapao_comic_picker.css', import.meta.url).href;
document.head.append(css);
let active = null;
const make = (tag, cls, text) => {
    const el = document.createElement(tag); el.className = cls || ''; if (text) el.textContent = text; return el;
};
function info(field, value) {
    const data = comicPreviews[`${field}|${value}`];
    return {description: data?.description || '按所选方向创作。', url: data ? new URL(`./comic_previews/${data.file}`, import.meta.url).href : ''};
}
function thumbnail(field, value, cls) {
    const img = make('img', cls); img.src = info(field, value).url; img.alt = `${value}漫画样页`; img.draggable = false;
    img.addEventListener('error', () => { img.style.visibility = 'hidden'; img.parentElement?.classList.add('comic-image-unavailable'); }, {once:true});
    return img;
}
export function closeComicPicker(owner) {
    if (active && (!owner || owner.contains(active.anchor))) active.close();
}
export function createComicChoice(field, widget, commit) {
    const trigger = make('button', 'comic-choice'); trigger.type = 'button'; trigger.setAttribute('aria-label', field);
    trigger.setAttribute('aria-haspopup','listbox'); trigger.setAttribute('aria-expanded','false');
    const label = make('span','comic-choice-copy'); label.append(make('strong','',String(widget.value)),make('small','',info(field,widget.value).description));
    trigger.append(thumbnail(field,widget.value,'comic-choice-thumb'),label,make('span','comic-chevron','⌄'));
    function open(event) {
        if(active?.anchor===trigger){active.close();return;}
        closeComicPicker();
        const values=typeof widget.options?.values==='function'?widget.options.values():widget.options?.values;
        if(!Array.isArray(values)||!values.length)return;
        const popup=make('div','comic-picker');popup.setAttribute('aria-label',`${field}可视化选择`);
        const top=make('div','comic-picker-top');const close=make('button','comic-picker-close','×');close.type='button';close.setAttribute('aria-label','关闭预览');
        top.append(make('strong','',field),make('span','','悬浮预览 · 点击选中'),close);
        const body=make('div','comic-picker-body');const list=make('div','comic-picker-list');list.setAttribute('role','listbox');list.setAttribute('aria-label',field);
        const detail=make('div','comic-picker-detail comic-hover-preview');detail.hidden=true;detail.setAttribute('role','tooltip');const artwork=make('img','comic-picker-art');artwork.draggable=false;
        const heading=make('strong','comic-picker-heading');const description=make('p','comic-picker-description');const badge=make('span','comic-picker-badge');
        const fallback=make('div','comic-picker-fallback','漫画样页暂不可用，请参考下方说明');fallback.hidden=true;
        artwork.onerror=()=>{artwork.hidden=true;fallback.hidden=false;};
        detail.append(artwork,fallback,heading,description,badge,make('small','comic-picker-note','漫画样页用于说明选项，不是本次MOSS生成结果。'));
        body.append(list);popup.append(top,body);document.body.append(popup,detail);trigger.setAttribute('aria-expanded','true');
        const abort=new AbortController();let timer=0,frame=0,closed=false,previewRow=null;const rows=[];
        function hidePreview(){clearTimeout(timer);detail.hidden=true;previewRow=null;rows.forEach(row=>row.classList.remove('is-preview'));}
        function show(value){
            previewRow=rows.find(row=>row.dataset.value===value);detail.hidden=false;
            const data=info(field,value);artwork.hidden=false;fallback.hidden=true;artwork.src=data.url;artwork.alt=`${value}漫画样页`;
            heading.textContent=value;description.textContent=data.description;badge.textContent=value===widget.value?'✓ 当前已选':'预览中 · 点击左侧选用';
            rows.forEach(row=>row.classList.toggle('is-preview',row.dataset.value===value));
        }
        const cleanup=()=>{
            if(closed)return;closed=true;clearTimeout(timer);cancelAnimationFrame(frame);abort.abort();popup.remove();detail.remove();trigger.setAttribute('aria-expanded','false');if(active?.anchor===trigger)active=null;
        };
        active={anchor:trigger,close:cleanup};
        values.forEach(value=>{
            const row=make('button','comic-picker-option');row.type='button';row.dataset.value=value;row.setAttribute('role','option');row.setAttribute('aria-label',value);row.setAttribute('aria-selected',String(value===widget.value));
            const copy=make('span','comic-option-copy');copy.append(make('strong','',value),make('small','',info(field,value).description));
            row.append(thumbnail(field,value,'comic-option-thumb'),copy,make('span','comic-option-check',value===widget.value?'✓':''));
            row.onpointerenter=()=>{clearTimeout(timer);timer=setTimeout(()=>show(value),180);};
            row.onpointerleave=hidePreview;
            row.onfocus=()=>{clearTimeout(timer);show(value);};
            row.onblur=hidePreview;
            row.onclick=()=>{const owner=trigger.closest('.dapao-comic-options');cleanup();commit(value);owner?.querySelector(`[aria-label="${CSS.escape(field)}"]`)?.focus();};list.append(row);rows.push(row);
        });
        const initial=Math.max(0,values.indexOf(widget.value));
        function place(){
            if(closed)return;
            const r=trigger.getBoundingClientRect();
            if(!trigger.isConnected||r.bottom<0||r.top>innerHeight||r.right<0||r.left>innerWidth){cleanup();return;}
            const width=Math.min(320,innerWidth-24),height=Math.min(360,48+values.length*52,innerHeight-24);
            popup.style.width=`${width}px`;popup.style.height=`${height}px`;
            popup.style.left=`${Math.max(12,Math.min(r.left,innerWidth-width-12))}px`;
            const below=r.bottom+8,above=r.top-height-8;
            popup.style.top=`${Math.max(12,Math.min(below+height<=innerHeight-12?below:above>=12?above:r.top,innerHeight-height-12))}px`;
            if(previewRow&&!detail.hidden){
                const box=popup.getBoundingClientRect(),row=previewRow.getBoundingClientRect();
                if(row.bottom<box.top+42||row.top>box.bottom){hidePreview();}
                else {
                    const pw=Math.min(210,innerWidth-24);detail.style.width=`${pw}px`;
                    const x=box.right+8+pw<=innerWidth-12?box.right+8:box.left-pw-8>=12?box.left-pw-8:Math.max(12,innerWidth-pw-12);
                    detail.style.left=`${x}px`;detail.style.top=`${Math.max(12,Math.min(row.top,innerHeight-detail.offsetHeight-12))}px`;
                }
            }
            frame=requestAnimationFrame(place);
        }
        place();rows[initial].scrollIntoView({block:'nearest'});
        close.onclick=()=>{cleanup();trigger.focus();};
        document.addEventListener('pointerdown',e=>{if(!popup.contains(e.target)&&!trigger.contains(e.target))cleanup();},{capture:true,signal:abort.signal});
        document.addEventListener('keydown',e=>{if(e.key==='Escape'){e.preventDefault();e.stopPropagation();cleanup();trigger.focus();}},{capture:true,signal:abort.signal});
        popup.addEventListener('keydown',e=>{
            e.stopPropagation();const current=rows.indexOf(document.activeElement);let next;
            if(e.key==='ArrowDown')next=Math.min(rows.length-1,current+1);
            if(e.key==='ArrowUp')next=Math.max(0,current-1);
            if(e.key==='Home')next=0;if(e.key==='End')next=rows.length-1;
            if(next!==undefined){e.preventDefault();rows[next].focus();rows[next].scrollIntoView({block:'nearest'});}
            if(e.key==='Tab')cleanup();
        });
        for(const event of ['pointerdown','wheel','pointerup'])popup.addEventListener(event,e=>e.stopPropagation());
        rows[initial].focus({preventScroll:true});
        if(event?.detail!==0&&event?.type!=='keydown')hidePreview();
    }
    trigger.onclick=open;
    trigger.onkeydown=e=>{if(['ArrowDown','ArrowUp'].includes(e.key)){e.preventDefault();if(active?.anchor!==trigger)open(e);}};
    return trigger;
}

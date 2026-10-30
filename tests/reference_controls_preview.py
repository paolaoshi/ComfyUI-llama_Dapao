"""Isolated real-asset UI harness; does not start ComfyUI or run models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiohttp import web
from test_pose_reference import routes as pose_routes, ROOT
from test_face_reference import routes as face_routes
from test_costume_reference import routes as costume_routes

HTML = '''<!doctype html><meta charset="utf-8"><title>Reference controls verification</title>
<style>body{background:#14161b;color:white;font:14px system-ui;display:flex;gap:20px;padding:20px;flex-wrap:wrap}section{width:430px;flex:none;background:#191b20;padding:10px;border:1px solid #555;border-radius:10px}canvas{display:block}h3{margin:4px;font-size:14px;display:flex;justify-content:space-between} .downstream img{width:100%;height:240px;object-fit:contain}</style>
<script type="module">
window.extensions=[];window.nodes=[];
window.createdPreviews=0;
window.LiteGraph={createNode(type){if(type!=='PreviewImage')throw Error('Unexpected node type');window.createdPreviews++;return {id:'auto-'+window.createdPreviews,type,inputs:[{type:'IMAGE'}],size:[320,360],setSize(size){this.size=size;},setDirtyCanvas(){this.panel.replaceChildren(Object.assign(document.createElement('h3'),{textContent:'自动创建的预览图像'}),...this.imgs.map(i=>i.cloneNode()));}};}};
for (const kind of ['face','costume','pose']) await import(`/extensions/test/js/dapao_${kind}_reference.js`);
for (const [kind,title,idName] of [['Face','👩 美女脸模素材','👩脸模ID'],['Costume','👗 精选服装参考','👗服装ID'],['Pose','🧍 精选姿势参考','🧍姿势ID']]) {
 class Node {
  widgets=[{name:idName,value:''},...(kind==='Pose'?[]:[{name:'🌐提示词语言',value:'中文'}])];size=[430,900];graph={beforeChange(){},afterChange(){}};
  constructor(){this.host=document.createElement('section');this.host.dataset.kind=kind;this.host.innerHTML=`<h3>${title}@炮老师的小课堂</h3>`;this.canvas=document.createElement('canvas');this.canvas.width=430;this.canvas.height=kind==='Pose'?80:120;this.host.append(this.canvas);document.body.append(this.host);this.canvas.onpointerdown=e=>{const r=this.canvas.getBoundingClientRect();this.onMouseDown?.(e,[e.clientX-r.left,e.clientY-r.top]);};}
  addDOMWidget(name,type,root,options){root.style.height=options.getHeight()+'px';this.host.append(root);const w={name,type,root,options,computeSize:()=>[430,options.getHeight()]};this.widgets.push(w);return w;}
  addTitleButton(options){const b=document.createElement('button');b.textContent=options.text;b.onclick=()=>this.onTitleButtonClick(options);this.host.querySelector('h3').append(b);return options;}
  computeSize(){return [430,640];}setSize(size){this.size=size;}
  setDirtyCanvas(){const ctx=this.canvas.getContext('2d');ctx.clearRect(0,0,430,120);ctx.fillStyle='#ddd';ctx.font='13px sans-serif';ctx.textAlign='right';ctx.fillText('🖼️ 参考图 ○',420,20);this.onDrawForeground?.(ctx);}
 }
 await extensions.find(e=>e.name===`Dapao.${kind}Reference`).beforeRegisterNodeDef(Node,{name:`Dapao${kind}Reference`});
 const node=new Node();node.id=kind;node.pos=[0,0];
 const panel=document.createElement('div');panel.className='downstream';panel.innerHTML='<h3>连接的预览图像</h3>';node.host.append(panel);
 const preview={id:kind+'Preview',type:'PreviewImage',size:[430,300],setSize(s){this.size=s;},setDirtyCanvas(){panel.replaceChildren(Object.assign(document.createElement('h3'),{textContent:'连接的预览图像'}),...this.imgs.map(i=>i.cloneNode()));}};
 node.outputs=[{links:[1]}];node.graph.links={1:{origin_id:kind,origin_slot:0,target_id:preview.id,target_slot:0}};node.graph.getNodeById=id=>id===preview.id?preview:null;node.target=preview;
 node.graph.nodes=[node,preview];node.graph.created=[];
 node.graph.getNodeById=id=>node.graph.nodes.find(n=>n.id===id);
 node.graph.add=target=>{target.graph=node.graph;target.panel=panel;node.graph.nodes.push(target);node.graph.created.push(target);};
 node.graph.remove=target=>{node.graph.nodes=node.graph.nodes.filter(n=>n!==target);};
 node.connect=(slot,target,input)=>{const id=Object.keys(node.graph.links).length+1;const edge={origin_id:node.id,origin_slot:slot,target_id:target.id,target_slot:input};node.graph.links[id]=edge;(node.outputs[slot].links??=[]).push(id);return edge;};
 node.onNodeCreated();node.setDirtyCanvas();nodes.push(node);node.host.append(panel);
}
</script>'''
app=web.Application()
for routes in (pose_routes,face_routes,costume_routes): app.add_routes(routes)
app.router.add_get('/',lambda r:web.Response(text=HTML,content_type='text/html'))
app.router.add_get('/scripts/app.js',lambda r:web.Response(text='export const app={registerExtension(e){window.extensions.push(e)}}',content_type='text/javascript'))
app.router.add_get('/scripts/api.js',lambda r:web.Response(text='export const api={apiURL:p=>p,fetchApi:(p,o)=>fetch(p,o)}',content_type='text/javascript'))
app.router.add_static('/extensions/test/js/',ROOT/'web/js')
if __name__=='__main__': web.run_app(app,host='127.0.0.1',port=18746,print=None)

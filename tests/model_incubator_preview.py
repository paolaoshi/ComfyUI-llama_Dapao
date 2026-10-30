"""Isolated browser harness using real galleries; graph/title/preview APIs are mocks."""
from tempfile import TemporaryDirectory
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiohttp import web
from test_model_incubator import routes, module, ROOT

HTML = '''<!doctype html><meta charset="utf-8"><title>孵化器独立模拟验证</title>
<style>body{background:#14161b;color:#edf0f5;font:14px system-ui;padding:24px}#host{width:620px}h2{display:flex;justify-content:space-between;font-size:18px}h2 button{background:#244b70;color:#f0f7ff;border:1px solid #71a9dc;border-radius:5px}#outputs{display:flex;gap:10px;margin-top:24px}#outputs img{width:200px;height:220px;object-fit:contain}</style>
<div id="host"><h2>✨ 无限模特孵化器@炮老师小课堂</h2></div><div id="outputs"></div>
<script type="module">
window.extensions=[];
await import('/extensions/test/js/dapao_model_incubator.js');
const host=document.querySelector('#host');let next=2;
window.LiteGraph={createNode(type){return {id:next++,type,size:[320,360],pos:[0,0],setSize(s){this.size=s},setDirtyCanvas(){let box=document.getElementById('preview-'+this.id);if(!box){box=document.createElement('div');box.id='preview-'+this.id;document.querySelector('#outputs').append(box)}box.replaceChildren(...(this.imgs||[]).map(i=>i.cloneNode()))}}}};
class Node{
 constructor(){this.id=1;this.pos=[0,0];this.size=[620,720];this.outputs=Array.from({length:4},()=>({links:[]}));this.widgets=[];
 for(const [label,emoji] of [['脸模','👩'],['服装','👗'],['姿势','🧍']])for(const [field,value] of [['来源','素材库'],['ID',''],['上传','']])this.widgets.push({name:emoji+label+field,value,options:{}});
 this.widgets.push({name:'🎲提示词种子',value:0,options:{}});}
 addDOMWidget(name,type,root,options){host.append(root);root.style.height=options.getHeight()+'px';const w={name,type,options};this.widgets.push(w);return w}
 addTitleButton(options){const b=document.createElement('button');b.textContent=options.text;b.onclick=()=>this.onTitleButtonClick(options);host.querySelector('h2').append(b);return options}
 setSize(s){this.size=s}setDirtyCanvas(){}
 connect(slot,target,input){const id=Object.keys(this.graph.links).length+1;this.graph.links[id]={origin_id:this.id,origin_slot:slot,target_id:target.id,target_slot:input};this.outputs[slot].links.push(id);return this.graph.links[id]}
}
await extensions.find(e=>e.name==='Dapao.ModelIncubator').beforeRegisterNodeDef(Node,{name:'DapaoModelIncubator'});
window.node=new Node();window.graph={nodes:[node],links:{},beforeChange(){},afterChange(){},getNodeById(id){return this.nodes.find(n=>n.id===id)},add(n){n.graph=this;this.nodes.push(n)},remove(n){this.nodes=this.nodes.filter(x=>x!==n);for(const [id,e]of Object.entries(this.links))if(e.target_id===n.id){delete this.links[id];node.outputs[e.origin_slot].links=node.outputs[e.origin_slot].links.filter(x=>x!=id)}}};node.graph=graph;node.onNodeCreated();
</script>'''

app = web.Application(client_max_size=32 * 1024 * 1024)
app.add_routes(routes)
app.router.add_get('/', lambda r:web.Response(text=HTML, content_type='text/html'))
app.router.add_get('/scripts/app.js',lambda r:web.Response(text='export const app={registerExtension(e){window.extensions.push(e)}}',content_type='text/javascript'))
app.router.add_get('/scripts/api.js',lambda r:web.Response(text='export const api={apiURL:p=>p,fetchApi:(p,o)=>fetch(p,o)}',content_type='text/javascript'))
app.router.add_static('/extensions/test/js/',ROOT/'web/js')

async def upload(request):
    data=await request.post(); file=data['image']; name=Path(file.filename).name
    path=Path(module.folder_paths.get_input_directory())/name
    if path.exists():name='copy-'+name;path=path.with_name(name)
    path.write_bytes(file.file.read())
    return web.json_response({'name':name,'subfolder':'','type':'input'})

async def view(request):
    return web.FileResponse(Path(module.folder_paths.get_input_directory())/Path(request.query['filename']).name)

app.router.add_post('/upload/image',upload);app.router.add_get('/view',view)
if __name__=='__main__':
    with TemporaryDirectory(prefix='dapao-incubator-test-') as directory:
        module.folder_paths.get_input_directory=lambda:directory
        web.run_app(app,host='127.0.0.1',port=18747,print=None)

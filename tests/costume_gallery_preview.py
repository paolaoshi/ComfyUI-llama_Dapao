"""Isolated browser harness using production costume routes, CSS and JS."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_costume_reference import ROOT,routes
from aiohttp import web
HTML='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Costume node verification</title>
<style>body{background:#15181e;color:#eee;font:14px system-ui;padding:12px}.node-host{width:430px;border:2px solid #559bdf;border-radius:12px;padding:10px;box-sizing:border-box}.node-host h3{font-size:15px}</style>
<script type="module">
import '/extensions/costume/js/dapao_costume_reference.js';
class TestNode {
 widgets=[{name:'👗服装ID',value:''},{name:'🌐提示词语言',value:'中文'}];size=[430,700];changes=0;
 graph={beforeChange:()=>this.changes++,afterChange(){}};
 addDOMWidget(name,type,root,options){const host=document.createElement('section');host.className='node-host';const title=document.createElement('h3');title.textContent='精选服装参考@泡老师的小课堂';host.append(title,root);root.style.height='480px';document.body.append(host);const dom={name,type,options,root,computeSize:()=>[360,480]};this.widgets.push(dom);this.host=host;this.dom=dom;return dom;}
 computeSize(){return [360,5*20+this.widgets.filter(w=>!w.hidden).reduce((sum,w)=>sum+(w.computeSize?.()[1]||20)+4,0)+14];}
 setSize(size){this.size=size;if(this.host)this.host.style.width=size[0]+'px';}setDirtyCanvas(){}
}
await window.costumeExtension.beforeRegisterNodeDef(TestNode,{name:'DapaoCostumeReference'});
window.testNode=new TestNode();window.testNode.onNodeCreated();
</script>'''
app=web.Application();app.add_routes(routes)
app.router.add_get('/',lambda r:web.Response(text=HTML,content_type='text/html'))
app.router.add_get('/favicon.ico',lambda r:web.Response(status=204))
app.router.add_get('/scripts/app.js',lambda r:web.Response(text='export const app={registerExtension(value){window.costumeExtension=value}};',content_type='text/javascript'))
app.router.add_get('/scripts/api.js',lambda r:web.Response(text='export const api={apiURL:path=>path,fetchApi:(path,options)=>fetch(path,options)};',content_type='text/javascript'))
app.router.add_static('/extensions/costume/js/',ROOT/'web/js')
if __name__=='__main__':web.run_app(app,host='127.0.0.1',port=18744,print=None)

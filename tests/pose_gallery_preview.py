"""Isolated UI harness using real plugin routes/assets, not a ComfyUI instance."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pose_reference import ROOT, routes
from aiohttp import web

HTML = '''<!doctype html><meta charset="utf-8"><title>Pose gallery isolated verification</title>
<style>body{background:#18191e;color:white;font:16px system-ui;padding:20px}</style>
<button id="open">打开测试图库</button><p id="selection">尚未选择</p>
<script type="module">
import {openPoseGallery} from '/extensions/pose/js/dapao_pose_reference.js';
let id=localStorage.getItem('pose')||'';
document.querySelector('#selection').textContent=id||'尚未选择';
document.querySelector('#open').onclick=()=>openPoseGallery(id,item=>{id=item.id;localStorage.setItem('pose',id);document.querySelector('#selection').textContent=id;});
</script>'''

app = web.Application()
app.add_routes(routes)
app.router.add_get("/", lambda request: web.Response(text=HTML, content_type="text/html"))
app.router.add_get("/scripts/app.js", lambda request: web.Response(text="export const app = {registerExtension(value){window.poseExtension=value}};", content_type="text/javascript"))
app.router.add_get("/scripts/api.js", lambda request: web.Response(text="export const api = {apiURL:path=>path,fetchApi:(path,options)=>fetch(path,options)};", content_type="text/javascript"))
app.router.add_static("/extensions/pose/js/", ROOT / "web" / "js")
if __name__ == "__main__":
    web.run_app(app, host="127.0.0.1", port=18743, print=None)

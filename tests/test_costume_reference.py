"""Offline costume checks; no ComfyUI startup or model execution."""
import hashlib
import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('costume_test_plugin');package.__path__=[str(ROOT)];sys.modules[package.__name__]=package
routes = web.RouteTableDef();server = types.ModuleType('server');server.PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes));sys.modules['server']=server
module=importlib.import_module('costume_test_plugin.costume_reference_node')

class CostumeTests(unittest.TestCase):
    def test_all_assets(self):
        data,index=module.catalog();self.assertEqual(len(index),1407);self.assertEqual(len(data['categories']),8)
        for item in data['items']:
            raw=module.image_path(item).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),item['sha256']);self.assertEqual(len(raw),item['bytes'])
            with Image.open(module.image_path(item)) as image:image.load();self.assertEqual(image.size,(item['width'],item['height']))
            with Image.open(module.image_path(item,'thumb')) as image:image.load();self.assertEqual(image.size,(360,540))
        print('Verified 1407 lossless files and 1407 thumbnails')
    def test_native_output_and_languages(self):
        import torch
        for category in module.catalog()[0]['categories']:
            item=next(i for i in module.catalog()[0]['items'] if i['category_id']==category['id'])
            for lang,suffix in [('中文','zh'),('English','en')]:
                image,description,prompt,negative,identifier=module.DapaoCostumeReference().load_costume(**{'👗服装ID':item['id'],'🌐提示词语言':lang})
                self.assertEqual(tuple(image.shape),(1,item['height'],item['width'],3));self.assertEqual(image.dtype,torch.float32)
                self.assertTrue(0<=image.min()<=image.max()<=1);self.assertEqual(prompt,item['prompt_'+suffix]);self.assertEqual(negative,item['negative_prompt_'+suffix]);self.assertEqual(identifier,item['id']);self.assertIn(item['category_zh'],description)
    def test_invalid_and_missing(self):
        for identifier in ('','../manifest.json','SPR-0000'):
            with self.assertRaises(ValueError):module.selected_item(identifier)
        with self.assertRaises(ValueError):module.image_path(module.selected_item('SPR-0001'),'../../file')
        with TemporaryDirectory() as directory,patch.object(module,'RESOURCE_ROOT',Path(directory)):
            with self.assertRaisesRegex(FileNotFoundError,'本地服装素材'):module.catalog()
    def test_registration(self):
        cls=module.NODE_CLASS_MAPPINGS['DapaoCostumeReference'];self.assertEqual(module.NODE_DISPLAY_NAME_MAPPINGS['DapaoCostumeReference'],'精选服装参考@泡老师的小课堂')
        for attr in ('INPUT_TYPES','RETURN_TYPES','RETURN_NAMES','CATEGORY','FUNCTION'):self.assertTrue(hasattr(cls,attr))
        self.assertEqual(len(cls.RETURN_TYPES),5)

class Routes(AioHTTPTestCase):
    async def get_application(self):
        app=web.Application();app.add_routes(routes);return app
    async def test_routes(self):
        response=await self.client.get('/dapao/costume-gallery/catalog');self.assertEqual(response.status,200)
        self.assertEqual(len((await response.json())['items']),1407)
        for kind in ('image','thumb'):
            response=await self.client.get('/dapao/costume-gallery/image/SPR-0001?kind='+kind);self.assertEqual(response.status,200)
            self.assertEqual(await response.read(),module.image_path(module.selected_item('SPR-0001'),kind).read_bytes())
        for path in ('/dapao/costume-gallery/image/NOPE','/dapao/costume-gallery/image/SPR-0001?kind=invalid'):
            response=await self.client.get(path);self.assertEqual(response.status,404)

if __name__=='__main__':unittest.main()

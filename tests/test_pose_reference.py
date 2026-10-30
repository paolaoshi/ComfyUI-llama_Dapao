"""Run with ComfyUI's Python. Does not start ComfyUI or load any model."""
import hashlib
import importlib
import sys
import types
import unittest
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("pose_test_plugin")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
routes = web.RouteTableDef()
server = types.ModuleType("server")
server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes))
sys.modules["server"] = server
module = importlib.import_module("pose_test_plugin.pose_reference_node")


class PoseTests(unittest.TestCase):
    def test_all_assets(self):
        data = module.catalog()
        self.assertEqual(len(data["items"]), 1439)
        self.assertEqual(len({(i["system"], i["number"]) for i in data["items"]}), 288)
        size = 0
        for item in data["items"]:
            path = module.image_path(item)
            raw = path.read_bytes()
            self.assertEqual(len(raw), item["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), item["sha256"])
            with Image.open(path) as image:
                image.load()
                self.assertEqual(image.size, (600, 800))
                self.assertEqual(image.format, "WEBP")
            size += len(raw)
        print(f"Verified 1439 images, 288 groups, {size / 1024**2:.2f} MiB")

    def test_output(self):
        import torch
        for system in {i["system"] for i in module.catalog()["items"]}:
            item = next(i for i in module.catalog()["items"] if i["system"] == system)
            image, description, identifier = module.DapaoPoseReference().load_pose(**{"🧍姿势ID": item["id"]})
            self.assertEqual(tuple(image.shape), (1, 800, 600, 3))
            self.assertEqual(image.dtype, torch.float32)
            self.assertTrue(0 <= image.min() <= image.max() <= 1)
            self.assertIn(item["angle"], description)
            self.assertEqual(identifier, item["id"])

    def test_invalid_selection(self):
        for identifier in ("", "../manifest.json", "missing", "images/test.webp"):
            with self.assertRaises(ValueError):
                module.DapaoPoseReference().load_pose(**{"🧍姿势ID": identifier})

    def test_registration(self):
        cls = module.NODE_CLASS_MAPPINGS["DapaoPoseReference"]
        self.assertEqual(module.NODE_DISPLAY_NAME_MAPPINGS["DapaoPoseReference"], "🧍 精选姿势参考@炮老师的小课堂")
        for name in ("INPUT_TYPES", "RETURN_TYPES", "RETURN_NAMES", "FUNCTION", "CATEGORY"):
            self.assertTrue(hasattr(cls, name))


class RouteTests(AioHTTPTestCase):
    async def get_application(self):
        app = web.Application()
        app.add_routes(routes)
        return app

    async def test_routes(self):
        response = await self.client.get("/dapao/pose-gallery/catalog")
        self.assertEqual(response.status, 200)
        item = (await response.json())["items"][0]
        response = await self.client.get("/dapao/pose-gallery/image/" + item["id"])
        self.assertEqual(response.status, 200)
        self.assertEqual(hashlib.sha256(await response.read()).hexdigest(), item["sha256"])
        response = await self.client.get("/dapao/pose-gallery/image/missing")
        self.assertEqual(response.status, 404)


if __name__ == "__main__":
    unittest.main()

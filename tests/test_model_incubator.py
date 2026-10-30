"""Offline integration of the real catalog and image loading; no ComfyUI startup."""
import importlib
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from aiohttp import web
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("incubator_test_plugin")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
routes = web.RouteTableDef()
sys.modules["server"] = types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes)))
sys.modules["folder_paths"] = types.SimpleNamespace(get_input_directory=lambda: str(ROOT / "unused-test-input"))
module = importlib.import_module("incubator_test_plugin.model_incubator_node")


def defaults():
    result = {"🎲提示词种子": 0}
    for label, emoji, gallery in module.KINDS:
        data = gallery.catalog()
        if isinstance(data, tuple):
            data = data[0]
        item = next(i for i in data["items"] if label != "姿势" or i["system"] in module.SINGLE_SYSTEMS)
        result.update({f"{emoji}{label}来源": "素材库", f"{emoji}{label}ID": item["id"], f"{emoji}{label}上传": ""})
    return result


class IncubatorTests(unittest.TestCase):
    def test_three_outputs_native_shapes_and_prompt(self):
        import torch
        result = module.DapaoModelIncubator().compose(**defaults())
        self.assertEqual(len(result), 4)
        for tensor, (label, emoji, gallery) in zip(result[:3], module.KINDS):
            with Image.open(module.resolve_source(label, emoji, gallery, defaults())) as image:
                self.assertEqual(tuple(tensor.shape), (1, image.height, image.width, 3))
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertTrue(0 <= tensor.min() <= tensor.max() <= 1)
        self.assertIn("图片1", result[3]); self.assertIn("图片2", result[3]); self.assertIn("图片3", result[3])

    def test_single_pose_and_reject_pair(self):
        items = module.pose.catalog()["items"]
        self.assertEqual(sum(i["system"] in module.SINGLE_SYSTEMS for i in items), 1164)
        values = defaults(); values["🧍姿势ID"] = next(i["id"] for i in items if i["system"] not in module.SINGLE_SYSTEMS)
        with self.assertRaisesRegex(ValueError, "单人姿势"):
            module.resolve_source("姿势", "🧍", module.pose, values)

    def test_upload_mixed_and_traversal_and_missing(self):
        values = defaults(); values.update({"👩脸模来源": "上传图片", "👩脸模上传": "custom.png"})
        with TemporaryDirectory() as directory, patch.object(module.folder_paths, "get_input_directory", return_value=directory):
            Image.new("RGB", (41, 29), "red").save(Path(directory) / "custom.png")
            result = module.DapaoModelIncubator().compose(**values)
            self.assertEqual(tuple(result[0].shape), (1, 29, 41, 3))
            for invalid in ("../outside.png", "missing.png", ""):
                values["👩脸模上传"] = invalid
                with self.assertRaises(ValueError):module.resolve_source("脸模", "👩", module.face, values)

    def test_prompt_reproducible_varied_and_registration(self):
        self.assertEqual(module.integrated_prompt(12), module.integrated_prompt(12))
        self.assertEqual(len({module.integrated_prompt(i) for i in range(9)}), 9)
        cls = module.NODE_CLASS_MAPPINGS["DapaoModelIncubator"]
        for attr in ("INPUT_TYPES", "RETURN_TYPES", "RETURN_NAMES", "FUNCTION", "CATEGORY"):self.assertTrue(hasattr(cls, attr))
        self.assertTrue(module.NODE_DISPLAY_NAME_MAPPINGS["DapaoModelIncubator"].endswith("@炮老师小课堂"))


if __name__ == "__main__":unittest.main()

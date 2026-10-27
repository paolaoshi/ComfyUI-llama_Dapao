"""Local H3 contracts; real ComfyUI schema/media, mocked model inference."""
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))
sys.argv = [sys.argv[0]]
sys.stdout.reconfigure(encoding="utf-8")
package = types.ModuleType("dapao_h3_tests")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
port = importlib.import_module("dapao_h3_tests.h3_local_prompt_node")
chat = importlib.import_module("dapao_h3_tests.multi_turn_chat_node")
import torch


def response(text):
    return ({"choices": [{"message": {"content": text}}]}, 256)


class H3Tests(unittest.TestCase):
    def setUp(self):
        self.model = chat.DapaoLocalModel(object(), {
            "family": "Qwen3.5-VL", "model_file": "test.gguf",
            "mmproj_file": "mmproj.gguf", "think": False,
        })
        self.params = dict(model=self.model, prompt="测试", skill="h3-prompt-writing",
                           duration=5.0, seed=2**48, max_tokens=256,
                           video_sample_frames_per_sec=2, force_unload_model=False)
        self.sync = self.enterContext(patch.object(port, "_sync_model", return_value=self.model))
        self.release = self.enterContext(patch.object(port, "_release_model"))
        self.generate = self.enterContext(patch.object(port, "_create_completion", return_value=response("有效正文")))
        self.route = self.enterContext(patch.object(port, "_route", return_value="h3-prompt-writing"))

    def run_node(self, **kwargs):
        return port.DapaoLocalH3Prompt.execute(**(self.params | kwargs)).result

    def test_schema(self):
        node = port.DapaoLocalH3Prompt
        self.assertEqual(node.INPUT_TYPES()["required"]["model"][0], "DAPAO_LOCAL_MODEL")
        self.assertEqual(tuple(node.RETURN_TYPES), ("STRING",) * 3)
        original_skills = {
            "3d-animation-short-generator", "brand-promo-video-generator",
            "co-op-game-intro-generator", "h3-prompt-writing",
            "handdrawn-live-video-generator", "minimalist-product-ad-generator",
            "music-video-subtitle-generator", "paper-collage-explainer-generator",
            "papercraft-stop-motion-explainer",
        }
        self.assertTrue(original_skills.issubset(port.SKILL_NAMES))
        self.assertEqual(len(port.SKILL_NAMES), len(set(port.SKILL_NAMES)))
        self.assertIs(port.NODE_CLASS_MAPPINGS["DapaoLocalH3Prompt"], node)

    def test_moss_custom_skill_execution(self):
        skill = "moss-image-prompt-local"
        self.assertIn(skill, port.SKILL_NAMES)
        self.assertEqual(self.run_node(skill=skill), ("有效正文", skill, "t2va"))
        self.route.assert_not_called()
        messages = self.generate.call_args.args[1]
        self.assertIn("Return exactly one cohesive MOSS-Image prompt", messages[0]["content"])

    def test_text_and_auto_skill(self):
        self.assertEqual(self.run_node(skill="auto"), ("有效正文", "h3-prompt-writing", "t2va"))
        self.route.assert_called_once()
        messages, params = self.generate.call_args.args[1:]
        self.assertIsInstance(messages[1]["content"], str)
        self.assertLess(params["seed"], 2**32 - 1)
        self.release.assert_not_called()

    def test_first_last_and_reference_images(self):
        image = torch.zeros((1, 8, 8, 3))
        for count, selected in ((1, "i2va"), (1, "l2va"), (2, "fl2va"), (2, "ref2va"), (3, "ref2va")):
            with self.subTest(mode=selected, count=count):
                self.route.return_value = selected
                result = self.run_node(reference_images={f"reference_image_{n}": image for n in range(count)})
                self.assertEqual(result[2], selected)
                content = self.generate.call_args.args[1][1]["content"]
                self.assertEqual(sum(item["type"] == "image_url" for item in content), count)

    def test_video_contact_sheets(self):
        result = self.run_node(reference_videos={"reference_video_0": torch.zeros((48, 8, 8, 3))})
        self.assertEqual(result[2], "ref2va")
        content = self.generate.call_args.args[1][1]["content"]
        self.assertEqual(sum(item["type"] == "image_url" for item in content), 2)
        self.assertIn("2.00 seconds", content[1]["text"])
        self.route.assert_not_called()

    def test_missing_projector_and_invalid_skill(self):
        self.model.settings["mmproj_file"] = "None"
        with self.assertRaisesRegex(ValueError, "mmproj"):
            self.run_node(reference_images={"reference_image_0": torch.zeros((1, 8, 8, 3))})
        with self.assertRaisesRegex(ValueError, "Skill"):
            self.run_node(skill="../../outside")
        self.sync.assert_not_called()
        self.assertEqual(self.release.call_count, 2)

    def test_empty_output_repair_and_failure(self):
        self.generate.side_effect = [response(None), response("修复完成")]
        self.assertEqual(self.run_node()[0], "修复完成")
        self.generate.side_effect = [response("<think>未完成"), response("")]
        with self.assertRaisesRegex(RuntimeError, "正文"):
            self.run_node()
        self.release.assert_called_once()

    def test_unload_and_inference_error(self):
        self.run_node(force_unload_model=True)
        self.release.assert_called_once()
        self.release.reset_mock()
        self.model.settings["unload_after_run"] = True
        self.run_node()
        self.release.assert_called_once()
        self.generate.side_effect = ValueError("context overflow")
        with self.assertRaisesRegex(ValueError, "context overflow"):
            self.run_node()
        self.assertEqual(self.release.call_count, 2)

    def test_thinking_settings(self):
        self.model.settings["think"] = True
        self.run_node()
        params = self.generate.call_args.args[2]
        self.assertEqual(params["temperature"], 1.0)
        self.assertEqual(params["top_p"], 0.95)
        self.assertEqual(port._reply(response("思考</think>正文")[0]), "正文")


if __name__ == "__main__":
    unittest.main()

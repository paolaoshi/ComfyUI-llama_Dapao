"""Bundled, offline pose references; no model, download or generation involved."""
import json
import re
from functools import lru_cache
from pathlib import Path

from aiohttp import web
from PIL import Image
from server import PromptServer

from .image_input_utils import pil2tensor

RESOURCE_ROOT = Path(__file__).parent / "resources" / "pose-gallery"


@lru_cache(maxsize=1)
def catalog():
    data = json.loads((RESOURCE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    if data.get("version") != 1 or not data.get("items"):
        raise ValueError("姿势目录无效，请检查插件素材是否完整。")
    seen = set()
    for item in data["items"]:
        identifier = item.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in seen:
            raise ValueError("姿势目录包含无效或重复 ID。")
        if item.get("file") != f"images/{identifier}.webp":
            raise ValueError("姿势图片路径无效。")
        seen.add(identifier)
    return data


@lru_cache(maxsize=1)
def item_index():
    return {item["id"]: item for item in catalog()["items"]}


def selected_item(identifier):
    item = item_index().get(identifier)
    if item is None:
        raise ValueError("请打开姿势库并选择一个有效视角。")
    return item


def image_path(item):
    path = RESOURCE_ROOT / item["file"]
    if not path.is_file():
        raise FileNotFoundError("参考图缺失，请恢复插件的 pose-gallery 素材目录。")
    return path


@PromptServer.instance.routes.get("/dapao/pose-gallery/catalog")
async def get_catalog(_request):
    try:
        return web.json_response(catalog())
    except (ValueError, OSError) as error:
        return web.json_response({"error": str(error)}, status=500)


@PromptServer.instance.routes.get("/dapao/pose-gallery/image/{identifier}")
async def get_image(request):
    try:
        return web.FileResponse(image_path(selected_item(request.match_info["identifier"])))
    except (ValueError, OSError):
        raise web.HTTPNotFound(text="姿势不存在或图片缺失")


class DapaoPoseReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"🧍姿势ID": ("STRING", {"default": "", "multiline": False})}}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("🖼️姿势参考图", "📝姿势说明", "🆔姿势ID")
    FUNCTION = "load_pose"
    CATEGORY = "🍭大炮-llama-cpp"
    DESCRIPTION = "打开本地姿势库，选择姿势与视角。输出标准 IMAGE；不会调用模型。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        try:
            stat = image_path(selected_item(kwargs.get("🧍姿势ID", ""))).stat()
            return f"{stat.st_mtime_ns}:{stat.st_size}"
        except (ValueError, OSError):
            return float("nan")

    def load_pose(self, **kwargs):
        item = selected_item(kwargs.get("🧍姿势ID", ""))
        with Image.open(image_path(item)) as image:
            tensor = pil2tensor(image)
        description = f"{item['name']}；{item['system']}；{item['category']}；{item['angle']}；{item['number']}"
        return tensor, description, item["id"]


NODE_CLASS_MAPPINGS = {"DapaoPoseReference": DapaoPoseReference}
NODE_DISPLAY_NAME_MAPPINGS = {"DapaoPoseReference": "精选姿势参考@泡老师的小课堂"}

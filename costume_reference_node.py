"""Local-only garment references; native-resolution lossless IMAGE output."""
import json
import re
from functools import lru_cache
from pathlib import Path

from aiohttp import web
from PIL import Image
from server import PromptServer

from .image_input_utils import pil2tensor

RESOURCE_ROOT = Path(__file__).parent / "resources" / "costume-gallery"


@lru_cache(maxsize=1)
def _read_catalog(stamp):
    data = json.loads((RESOURCE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    if data.get("version") != 1 or not data.get("items"):
        raise ValueError("服装目录格式无效。")
    index = {}
    for item in data["items"]:
        identifier = item.get("id", "")
        if not re.fullmatch(r"[A-Z]+-[0-9]+", identifier) or identifier in index:
            raise ValueError("服装目录含无效或重复 ID。")
        for kind in ("image", "thumb"):
            if item.get("files", {}).get(kind) != f"{kind}/{identifier}.webp":
                raise ValueError("服装素材路径无效。")
        for field in ("category_zh", "prompt_zh", "prompt_en", "negative_prompt_zh", "negative_prompt_en"):
            if not isinstance(item.get(field), str):
                raise ValueError("服装目录缺少必要字段。")
        index[identifier] = item
    return data, index


def catalog():
    try:
        stat = (RESOURCE_ROOT / "manifest.json").stat()
    except FileNotFoundError as error:
        raise FileNotFoundError("未找到本地服装素材，请按 docs/精选服装参考.md 配置素材目录。") from error
    return _read_catalog((stat.st_mtime_ns, stat.st_size))


def selected_item(identifier):
    item = catalog()[1].get(identifier)
    if item is None:
        raise ValueError("请打开服装库，选择一个有效的服装素材。")
    return item


def image_path(item, kind="image"):
    if kind not in ("image", "thumb"):
        raise ValueError("不支持的图片规格。")
    path = RESOURCE_ROOT / item["files"][kind]
    if not path.is_file():
        raise FileNotFoundError("服装图片缺失，请恢复本地服装素材。")
    return path


@PromptServer.instance.routes.get("/dapao/costume-gallery/catalog")
async def get_catalog(_request):
    try:
        return web.json_response(catalog()[0])
    except (ValueError, OSError) as error:
        return web.json_response({"error": str(error)}, status=500)


@PromptServer.instance.routes.get("/dapao/costume-gallery/image/{identifier}")
async def get_image(request):
    try:
        return web.FileResponse(image_path(selected_item(request.match_info["identifier"]), request.query.get("kind", "image")))
    except (ValueError, OSError):
        raise web.HTTPNotFound(text="服装图片不存在或规格无效")


class DapaoCostumeReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "👗服装ID": ("STRING", {"default": "", "multiline": False}),
            "🌐提示词语言": (["中文", "English"], {"default": "中文"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("🖼️服装参考图", "📝服装说明", "✨服装提示词", "🚫反向提示词", "🆔服装ID")
    FUNCTION = "load_costume"
    CATEGORY = "🍭大炮-llama-cpp"
    DESCRIPTION = "本地服装库：缩略图浏览，原尺寸无损图片输出；不调用模型。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        try:
            stat = image_path(selected_item(kwargs.get("👗服装ID", ""))).stat()
            manifest = (RESOURCE_ROOT / "manifest.json").stat()
            return f"{stat.st_mtime_ns}:{stat.st_size}:{manifest.st_mtime_ns}"
        except (ValueError, OSError):
            return float("nan")

    def load_costume(self, **kwargs):
        item = selected_item(kwargs.get("👗服装ID", ""))
        language = kwargs.get("🌐提示词语言", "中文")
        if language not in ("中文", "English"):
            raise ValueError("提示词语言必须为中文或 English。")
        with Image.open(image_path(item)) as image:
            tensor = pil2tensor(image)
        suffix = "zh" if language == "中文" else "en"
        description = f"{item['id']} · {item['category_zh']}；" + "、".join(item["tags_zh"])
        return tensor, description, item[f"prompt_{suffix}"], item[f"negative_prompt_{suffix}"], item["id"]


NODE_CLASS_MAPPINGS = {"DapaoCostumeReference": DapaoCostumeReference}
NODE_DISPLAY_NAME_MAPPINGS = {"DapaoCostumeReference": "👗 精选服装参考@炮老师的小课堂"}

import asyncio
import json
import tempfile
from pathlib import Path

import aiohttp.web
import server
from .model_incubator_node import NODE_CLASS_MAPPINGS as INCUBATOR_NODES, NODE_DISPLAY_NAME_MAPPINGS as INCUBATOR_NAMES
from .comic_options_node import NODE_CLASS_MAPPINGS as COMIC_NODES, NODE_DISPLAY_NAME_MAPPINGS as COMIC_NAMES
from .pose_reference_node import NODE_CLASS_MAPPINGS as POSE_NODES, NODE_DISPLAY_NAME_MAPPINGS as POSE_NAMES
from .costume_reference_node import NODE_CLASS_MAPPINGS as COSTUME_NODES, NODE_DISPLAY_NAME_MAPPINGS as COSTUME_NAMES

from .face_reference_node import NODE_CLASS_MAPPINGS as FACE_NODES, NODE_DISPLAY_NAME_MAPPINGS as FACE_NAMES

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .caption_node import NODE_CLASS_MAPPINGS_CAPTION, NODE_DISPLAY_NAME_MAPPINGS_CAPTION
from .caption_options import NODE_CLASS_MAPPINGS_OPTIONS, NODE_DISPLAY_NAME_MAPPINGS_OPTIONS
from .batch_prompt_node import NODE_CLASS_MAPPINGS_BATCH_PROMPT, NODE_DISPLAY_NAME_MAPPINGS_BATCH_PROMPT
from .local_api_common_nodes import (
    NODE_CLASS_MAPPINGS_LOCAL_API_COMMON,
    NODE_DISPLAY_NAME_MAPPINGS_LOCAL_API_COMMON,
)
from .multi_turn_chat_node import NODE_CLASS_MAPPINGS_MULTI, NODE_DISPLAY_NAME_MAPPINGS_MULTI
from .qwen_image_2_1_prompt_node import (
    NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_QWEN_IMAGE_21,
    NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_QWEN_IMAGE_21,
)
from .skill_loader_node import NODE_CLASS_MAPPINGS_SKILL, NODE_DISPLAY_NAME_MAPPINGS_SKILL
from .h3_local_prompt_node import NODE_CLASS_MAPPINGS as H3_LOCAL_NODES, NODE_DISPLAY_NAME_MAPPINGS as H3_LOCAL_NAMES
from .skill_runtime import (
    MAX_SKILL_UPLOAD_BYTES,
    install_uploaded_skills,
    set_skill_display_name,
    skill_catalog,
)

NODE_CLASS_MAPPINGS = {
    **COMIC_NODES,
    **INCUBATOR_NODES,
    **FACE_NODES,
    **COSTUME_NODES,
    **POSE_NODES,
    **NODE_CLASS_MAPPINGS,
    **NODE_CLASS_MAPPINGS_CAPTION,
    **NODE_CLASS_MAPPINGS_OPTIONS,
    **NODE_CLASS_MAPPINGS_BATCH_PROMPT,
    **NODE_CLASS_MAPPINGS_LOCAL_API_COMMON,
    **NODE_CLASS_MAPPINGS_MULTI,
    **NODE_CLASS_MAPPINGS_QWEN_IMAGE_21,
    **NODE_CLASS_MAPPINGS_SKILL,
    **H3_LOCAL_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **COMIC_NAMES,
    **INCUBATOR_NAMES,
    **FACE_NAMES,
    **COSTUME_NAMES,
    **POSE_NAMES,
    **NODE_DISPLAY_NAME_MAPPINGS,
    **NODE_DISPLAY_NAME_MAPPINGS_CAPTION,
    **NODE_DISPLAY_NAME_MAPPINGS_OPTIONS,
    **NODE_DISPLAY_NAME_MAPPINGS_BATCH_PROMPT,
    **NODE_DISPLAY_NAME_MAPPINGS_LOCAL_API_COMMON,
    **NODE_DISPLAY_NAME_MAPPINGS_MULTI,
    **NODE_DISPLAY_NAME_MAPPINGS_QWEN_IMAGE_21,
    **NODE_DISPLAY_NAME_MAPPINGS_SKILL,
    **H3_LOCAL_NAMES,
}

WEB_DIRECTORY = "./web"


@server.PromptServer.instance.routes.get("/dapao/local-skills/catalog")
async def get_dapao_local_skill_catalog(_request: aiohttp.web.Request):
    try:
        return aiohttp.web.json_response(await asyncio.to_thread(skill_catalog))
    except Exception as error:
        return aiohttp.web.json_response({"error": str(error)}, status=400)


@server.PromptServer.instance.routes.post("/dapao/local-skills/display-name")
async def update_dapao_local_skill_display_name(request: aiohttp.web.Request):
    try:
        body = await request.json()
        display_name = None if body.get("reset") else body.get("display_name")
        result = await asyncio.to_thread(
            set_skill_display_name,
            body.get("skill_id", ""),
            display_name,
            "manual",
        )
        return aiohttp.web.json_response(result)
    except Exception as error:
        return aiohttp.web.json_response({"error": str(error)}, status=400)


@server.PromptServer.instance.routes.post("/dapao/local-skills/install")
async def install_dapao_local_skills(request: aiohttp.web.Request):
    try:
        reader = await request.multipart()
        values = {}
        uploaded = []
        total = 0
        with tempfile.TemporaryDirectory(prefix="dapao-local-skill-route-") as temporary:
            temporary_root = Path(temporary)
            while True:
                field = await reader.next()
                if field is None:
                    break
                if field.name != "files":
                    values[field.name] = await field.text()
                    continue
                target = temporary_root / f"upload-{len(uploaded)}.bin"
                with target.open("wb") as stream:
                    while True:
                        chunk = await field.read_chunk(size=1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > MAX_SKILL_UPLOAD_BYTES:
                            raise ValueError("上传内容总大小超过512MB上限。")
                        stream.write(chunk)
                uploaded.append((Path(field.filename or f"file-{len(uploaded)}").name, target))

            try:
                relative_paths = json.loads(values.get("paths", "[]"))
            except json.JSONDecodeError as error:
                raise ValueError("上传文件路径清单无效。") from error
            if not isinstance(relative_paths, list) or len(relative_paths) != len(uploaded):
                raise ValueError("上传文件数量与路径清单不一致。")
            files = [(str(relative_paths[index]), path) for index, (_name, path) in enumerate(uploaded)]
            result = await asyncio.to_thread(
                install_uploaded_skills,
                files,
                values.get("mode", ""),
                values.get("package_hint", "uploaded-skill"),
            )
        return aiohttp.web.json_response(result)
    except Exception as error:
        status = 409 if isinstance(error, FileExistsError) else 400
        return aiohttp.web.json_response({"error": str(error)}, status=status)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

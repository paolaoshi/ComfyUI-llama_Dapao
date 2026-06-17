"""
💓 Llama 批量提示词节点

复用本项目 llama-cpp 本地模型能力，支持 A/B/C/D 四组图片智能对齐，
也支持无图文本批量生成提示词列表。
"""

import json
import random
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

import folder_paths
import comfy.model_management as mm

from .nodes import (
    CHAT_HANDLERS,
    DapaoLlamaStorage,
    Dapao_LlamaChat,
    image2base64,
    scale_image,
    tensor2pil,
)


NODE_NAME = "Dapao_LlamaBatchPrompt"
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
GROUP_KEYS = ("A", "B", "C", "D")
MISSING_STRATEGIES = ["严格报错", "单图复用", "末张补齐", "忽略缺失组"]
TASK_FAILURE_STRATEGIES = ["失败占位继续", "跳过失败继续", "任一失败中断"]
IMAGE_INFERENCE_MODES = ["逐条推理", "单次批量请求"]
DEFAULT_GROUP_ROLES = {
    "A": "目标图",
    "B": "参考图",
    "C": "补充参考",
    "D": "风格参考",
}

DEFAULT_SYSTEM_PROMPT = """你是一个专业的批量图像编辑提示词专家。你会严格根据每一组已对齐图片和用户元指令，为当前编号图片生成一个专属提示词。"""

DEFAULT_META_INSTRUCTION = """请根据当前组图片生成一个适合下游图像生成/图像编辑模型使用的最终提示词。
要求：
1. 只输出当前这一项的最终提示词文本。
2. 不输出编号、标题、Markdown、JSON、解释、寒暄或多余前后缀。
3. 必须保持与当前 A 图一一对应，不要描述其他编号图片。
4. 如果有 B/C/D 图，请按它们的角色说明理解并融合。"""

TEXT_ONLY_SYSTEM_HINT = """当用户没有提供图片时，你是一个批量提示词生成助手。你需要根据用户文字需求生成多条彼此不同、可直接用于下游文生图/图像生成模型的提示词。"""


def _log_info(message):
    print(f"[大炮-llama批量提示词] 信息：{message}")


def _log_error(message):
    print(f"[大炮-llama批量提示词] 错误：{message}")


def _natural_sort_key(path):
    text = path.name if isinstance(path, Path) else str(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _normalize_name(name):
    stem = Path(name).stem if name else ""
    return re.sub(r"[\s_\-\.()\[\]{}]+", "", stem.lower())


def _number_key(name):
    stem = Path(name).stem if name else ""
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        return ""
    return "-".join(str(int(number)) for number in numbers)


def _chinese_number_to_int(text):
    if not text:
        return None
    digits = {
        "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    }
    text = text.strip()
    if text == "十":
        return 10
    if "十" in text:
        left, _, right = text.partition("十")
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    return digits.get(text)


def _strip_list_prefix(text):
    return re.sub(r"^\s*(?:[-*•]+|\d+[\.\)、):：]|[一二三四五六七八九十]+[\.\)、):：])\s*", "", text).strip()


def _strip_think_blocks(text):
    text = (text or "").removeprefix(": ").lstrip()
    m = re.search(r"<think>.*?</think>(.*)", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    m = re.search(r"<\|channel>thought\n.*?<channel\|>(.*)", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return text.strip()


@dataclass
class BatchImageItem:
    group: str
    index: int
    name: str
    source: str
    path: str = ""
    tensor: object = None
    pil_image: object = None
    norm_key: str = field(init=False)
    num_key: str = field(init=False)
    encoded_size: tuple = field(default_factory=tuple, init=False)
    encoded_bytes: int = field(default=0, init=False)

    def __post_init__(self):
        self.norm_key = _normalize_name(self.name)
        self.num_key = _number_key(self.name)

    def to_public_dict(self):
        return {
            "group": self.group,
            "index": self.index + 1,
            "name": self.name,
            "source": self.source,
            "path": self.path,
            "normalized_key": self.norm_key,
            "number_key": self.num_key,
        }

    def to_pil(self, max_side):
        if self.pil_image is not None:
            image = self.pil_image
        elif self.tensor is not None:
            image = tensor2pil(self.tensor)
        elif self.path:
            with Image.open(self.path) as opened:
                image = opened.convert("RGB")
        else:
            raise ValueError(f"{self.group}组第 {self.index + 1} 张图片没有可用数据。")

        if image.mode != "RGB":
            image = image.convert("RGB")
        return scale_image(image, int(max_side or 1120))

    def to_image_part(self, max_side):
        pil_img = self.to_pil(max_side)
        b64 = image2base64(pil_img)
        self.encoded_size = pil_img.size
        self.encoded_bytes = len(b64) * 3 // 4
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        }


class Dapao_LlamaBatchPrompt:
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("📝 提示词列表", "📄 完整响应", "ℹ️ 处理信息")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "generate_batch_prompts"
    CATEGORY = "🍭大炮-llama-cpp"
    DESCRIPTION = "Llama 本地批量提示词：A/B/C/D 多组图片智能对齐，也支持无图批量提示词生成 @炮老师的小课堂"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        llm_files = folder_paths.get_filename_list("LLM")
        mmproj_files = ["None"] + [f for f in llm_files if "mmproj" in f.lower()]
        model_files = [f for f in llm_files if "mmproj" not in f.lower()]

        return {
            "required": {
                "🤖模型文件": (model_files,),
                "🔌对话处理器": (CHAT_HANDLERS, {"default": "None"}),
                "🖼️mmproj文件": (mmproj_files, {"default": "None"}),
                "📐上下文长度": ("INT", {"default": 8192, "min": 512, "max": 131072, "step": 512}),
                "💾显存限制(GB)": ("FLOAT", {"default": -1, "min": -1, "max": 999.0, "step": 0.5}),
                "🔢图像最小token": ("INT", {"default": 256, "min": 1, "max": 4096, "step": 1}),
                "🔢图像最大token": ("INT", {"default": 1344, "min": 1, "max": 8192, "step": 1}),
                "📝系统提示词": ("STRING", {"default": DEFAULT_SYSTEM_PROMPT, "multiline": True}),
                "🧾元指令": ("STRING", {"default": DEFAULT_META_INSTRUCTION, "multiline": True}),
                "🏷️A组角色": ("STRING", {"default": DEFAULT_GROUP_ROLES["A"], "multiline": False}),
                "🏷️B组角色": ("STRING", {"default": DEFAULT_GROUP_ROLES["B"], "multiline": False}),
                "🏷️C组角色": ("STRING", {"default": DEFAULT_GROUP_ROLES["C"], "multiline": False}),
                "🏷️D组角色": ("STRING", {"default": DEFAULT_GROUP_ROLES["D"], "multiline": False}),
                "📂A组文件夹": ("STRING", {"default": "", "multiline": False}),
                "📂B组文件夹": ("STRING", {"default": "", "multiline": False}),
                "📂C组文件夹": ("STRING", {"default": "", "multiline": False}),
                "📂D组文件夹": ("STRING", {"default": "", "multiline": False}),
                "🧩缺失处理策略": (MISSING_STRATEGIES, {"default": "严格报错"}),
                "🔢无图默认数量": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "🛡️多图模式最大提示词数量": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "🚦多图推理模式": (IMAGE_INFERENCE_MODES, {"default": "逐条推理"}),
                "🛟失败重试次数": ("INT", {"default": 0, "min": 0, "max": 5, "step": 1}),
                "🧪推理失败策略": (TASK_FAILURE_STRATEGIES, {"default": "失败占位继续"}),
                "📏图像最大边长": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32}),
                "🎲随机种子": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "🎰随机化": (["固定种子", "随机种子", "递增种子", "递减种子"], {"default": "固定种子"}),
                "📊最大输出token": ("INT", {"default": 1024, "min": 1, "max": 32768, "step": 1}),
                "🌡️温度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "🎯top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "🔝top_k": ("INT", {"default": 40, "min": 0, "max": 200, "step": 1}),
                "🔁重复惩罚": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.01}),
                "🧠思考模式": ("BOOLEAN", {"default": False}),
                "⚡推理后卸载模型": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "🖼️A组图像": ("IMAGE",),
                "🖼️B组图像": ("IMAGE",),
                "🖼️C组图像": ("IMAGE",),
                "🖼️D组图像": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    @staticmethod
    def _first_input_value(value, default=None):
        if isinstance(value, (list, tuple)):
            if not value:
                return default
            return value[0]
        return value if value is not None else default

    @classmethod
    def _text_input_value(cls, kwargs, name, default=""):
        value = cls._first_input_value(kwargs.get(name), default)
        return default if value is None else str(value)

    @classmethod
    def _int_input_value(cls, kwargs, name, default):
        value = cls._first_input_value(kwargs.get(name), default)
        try:
            return int(value)
        except Exception:
            return int(default)

    @classmethod
    def _float_input_value(cls, kwargs, name, default):
        value = cls._first_input_value(kwargs.get(name), default)
        try:
            return float(value)
        except Exception:
            return float(default)

    @classmethod
    def _bool_input_value(cls, kwargs, name, default=False):
        value = cls._first_input_value(kwargs.get(name), default)
        return bool(value)

    @staticmethod
    def _expand_image_input(value, group):
        if value is None:
            return []
        values = value if isinstance(value, (list, tuple)) else [value]
        items = []
        for raw in values:
            if raw is None:
                continue
            if isinstance(raw, torch.Tensor):
                if raw.dim() == 4:
                    for index in range(raw.shape[0]):
                        items.append(BatchImageItem(group, len(items), f"{group}_{len(items) + 1:03d}.png", "IMAGE", tensor=raw[index:index + 1]))
                elif raw.dim() == 3:
                    items.append(BatchImageItem(group, len(items), f"{group}_{len(items) + 1:03d}.png", "IMAGE", tensor=raw.unsqueeze(0)))
                continue
            if isinstance(raw, Image.Image):
                items.append(BatchImageItem(group, len(items), f"{group}_{len(items) + 1:03d}.png", "PIL", pil_image=raw))
        return items

    @staticmethod
    def _folder_images(folder_path, group):
        folder_text = (folder_path or "").strip().strip('"')
        if not folder_text:
            return []
        folder = Path(folder_text)
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"{group}组文件夹不存在或不是文件夹：{folder_text}")
        paths = sorted(
            [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS],
            key=_natural_sort_key,
        )
        return [BatchImageItem(group, index, path.name, "folder", path=str(path)) for index, path in enumerate(paths)]

    def _collect_group_items(self, kwargs, group):
        image_key = f"🖼️{group}组图像"
        folder_key = f"📂{group}组文件夹"
        image_items = self._expand_image_input(kwargs.get(image_key), group)
        if image_items:
            return image_items, {
                "group": group,
                "source": "IMAGE",
                "count": len(image_items),
                "folder_ignored": bool(self._text_input_value(kwargs, folder_key, "").strip()),
            }
        folder_items = self._folder_images(self._text_input_value(kwargs, folder_key, ""), group)
        return folder_items, {"group": group, "source": "folder" if folder_items else "empty", "count": len(folder_items)}

    @staticmethod
    def _unique_map(items, attr):
        result = {}
        duplicate_keys = set()
        for item in items:
            key = getattr(item, attr)
            if not key:
                continue
            if key in result:
                duplicate_keys.add(key)
                result[key] = None
            elif key not in duplicate_keys:
                result[key] = item
        return {key: item for key, item in result.items() if item is not None}

    @classmethod
    def _fallback_item(cls, items, strategy):
        if not items or strategy == "忽略缺失组":
            return None, "ignored_missing"
        if strategy == "单图复用" and len(items) == 1:
            return items[0], "single_reuse"
        if strategy == "末张补齐":
            return items[-1], "last_fill"
        return None, "missing"

    @classmethod
    def _match_item(cls, anchor, items, strategy):
        if not items:
            return None, "empty_group"
        if strategy == "单图复用" and len(items) == 1:
            return items[0], "single_reuse"

        norm_map = cls._unique_map(items, "norm_key")
        num_map = cls._unique_map(items, "num_key")
        target_has_numbers = any(item.num_key for item in items)

        if anchor.norm_key and anchor.norm_key in norm_map:
            return norm_map[anchor.norm_key], "filename_exact"
        if anchor.num_key and anchor.num_key in num_map:
            return num_map[anchor.num_key], "number_key"
        if anchor.index < len(items) and not (anchor.num_key and target_has_numbers):
            return items[anchor.index], "sequence"

        return cls._fallback_item(items, strategy)

    @classmethod
    def _build_alignment(cls, groups, roles, strategy, max_prompt_count=0):
        anchors = groups["A"]
        if not anchors:
            raise ValueError("检测到 B/C/D 组有图片，但 A组没有图片。图像对齐模式必须连接 A组图像或填写 A组文件夹；完全无图时会自动进入文本批量模式。")

        original_anchor_count = len(anchors)
        limit = max(0, int(max_prompt_count or 0))
        if limit > 0:
            anchors = anchors[:limit]

        rows = []
        errors = []
        for anchor in anchors:
            selected = {"A": anchor}
            row_groups = {
                "A": {
                    "role": roles["A"],
                    "match_method": "anchor",
                    "image": anchor.to_public_dict(),
                }
            }
            for group in ("B", "C", "D"):
                items = groups.get(group) or []
                match, method = cls._match_item(anchor, items, strategy)
                selected[group] = match
                row_groups[group] = {
                    "role": roles[group],
                    "match_method": method,
                    "image": match.to_public_dict() if match else None,
                }
                if items and method == "missing" and strategy == "严格报错":
                    errors.append(f"第 {anchor.index + 1} 项：{group}组找不到与 A={anchor.name} 对齐的图片。")

            rows.append({"index": anchor.index + 1, "selected": selected, "groups": row_groups})

        if errors:
            preview = "\n".join(errors[:10])
            extra = f"\n... 还有 {len(errors) - 10} 个错误" if len(errors) > 10 else ""
            raise ValueError(f"图片智能对齐失败，已按严格策略中止，避免提示词错位：\n{preview}{extra}")

        return rows, original_anchor_count, limit

    @staticmethod
    def _build_row_user_text(meta_instruction, row, total_count, roles):
        lines = [
            (meta_instruction or "").strip(),
            "",
            "【批量配对信息】",
            f"当前处理第 {row['index']}/{total_count} 项。",
        ]
        for group in GROUP_KEYS:
            image = row["groups"][group]["image"]
            role = roles[group]
            if image:
                lines.append(f"{group}组（{role}）：{image['name']}，匹配方式：{row['groups'][group]['match_method']}")
            else:
                lines.append(f"{group}组（{role}）：未提供或已忽略")
        lines.extend(["", "请只输出当前这一项的最终提示词文本，不要输出编号、标题、Markdown、JSON或解释。"])
        return "\n".join(line for line in lines if line is not None)

    @staticmethod
    def _build_batch_user_text(meta_instruction, rows, roles):
        lines = [
            (meta_instruction or "").strip(),
            "",
            "【批量配对信息】",
            f"本次共有 {len(rows)} 个任务。请为每个任务各生成 1 条专属提示词。",
        ]
        for row in rows:
            lines.append(f"任务 {row['index']}：")
            for group in GROUP_KEYS:
                image = row["groups"][group]["image"]
                role = roles[group]
                if image:
                    lines.append(f"- {group}组（{role}）：{image['name']}，匹配方式：{row['groups'][group]['match_method']}")
                else:
                    lines.append(f"- {group}组（{role}）：未提供或已忽略")
        lines.extend([
            "",
            "【强制输出格式】",
            "必须只输出一个 JSON 字符串数组。",
            f"数组长度必须严格等于 {len(rows)}。",
            "数组第 1 个元素对应任务 1，第 2 个元素对应任务 2，依此类推。",
            "不要输出 Markdown、编号、解释或 JSON 对象。",
        ])
        return "\n".join(line for line in lines if line is not None)

    @staticmethod
    def _clean_prompt(text):
        cleaned = _strip_think_blocks(text or "")
        cleaned = re.sub(r"^```(?:json|text)?\s*", "", cleaned.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())
        cleaned = re.sub(r"^(?:提示词|Prompt|prompt)\s*[:：]\s*", "", cleaned.strip())
        return cleaned.strip()

    @staticmethod
    def _detect_text_prompt_count(*texts, default_count=1):
        joined = "\n".join(str(text or "") for text in texts)
        patterns = [
            r"(\d{1,3})\s*(?:组|个|条|套|份|段)\s*(?:提示词|prompt|Prompt)?",
            r"(?:生成|输出|写|给我|帮我)\s*(\d{1,3})\s*(?:组|个|条|套|份|段)",
            r"(\d{1,3})\s*(?:prompts?|Prompts?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, joined)
            if match:
                return max(1, min(100, int(match.group(1))))

        chinese_match = re.search(r"([一二两三四五六七八九十]{1,3})\s*(?:组|个|条|套|份|段)", joined)
        if chinese_match:
            value = _chinese_number_to_int(chinese_match.group(1))
            if value:
                return max(1, min(100, value))

        return max(1, min(100, int(default_count or 1)))

    @classmethod
    def _extract_prompt_list(cls, response_text, expected_count):
        cleaned = _strip_think_blocks(response_text or "")
        candidates = [cleaned]
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if fence_match:
            candidates.insert(0, fence_match.group(1).strip())
        array_match = re.search(r"\[[\s\S]*\]", cleaned)
        if array_match:
            candidates.insert(0, array_match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                for key in ("prompts", "提示词", "list", "items", "data"):
                    value = parsed.get(key)
                    if isinstance(value, list):
                        parsed = value
                        break
            if isinstance(parsed, list):
                prompts = [cls._clean_prompt(str(item)) for item in parsed if str(item).strip()]
                if prompts:
                    return prompts[:expected_count]

        lines = []
        for line in re.split(r"\r?\n+", cleaned):
            line = _strip_list_prefix(line)
            if not line or line in ("[", "]", "{", "}"):
                continue
            line = line.rstrip(",")
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            lines.append(cls._clean_prompt(line))
        return [line for line in lines if line][:expected_count]

    @staticmethod
    def _extract_response_text(resp):
        return (resp["choices"][0]["message"]["content"] or "").removeprefix(": ").lstrip()

    def _ensure_model(self, model_file, handler_name, mmproj_file, n_ctx, vram_limit_gb, image_min_tokens, image_max_tokens, think_mode):
        load_config = {
            "model_file": model_file,
            "handler_name": handler_name,
            "mmproj_file": mmproj_file,
            "n_ctx": n_ctx,
            "vram_limit_gb": vram_limit_gb,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens,
            "think_mode": think_mode,
        }
        need_load = DapaoLlamaStorage.llm is None or DapaoLlamaStorage.current_config != load_config
        if need_load:
            DapaoLlamaStorage.clean()
            mm.soft_empty_cache()
            Dapao_LlamaChat()._load_model(
                model_file,
                handler_name,
                mmproj_file,
                n_ctx,
                vram_limit_gb,
                image_min_tokens,
                image_max_tokens,
                think_mode,
            )
        else:
            _log_info("复用已加载模型")
        return DapaoLlamaStorage.llm

    @staticmethod
    def _clear_local_cache_if_needed(llm, handler_name):
        if handler_name in ("Qwen3.5", "Qwen3.5-Thinking", "Qwen3-VL", "Qwen3-VL-Thinking"):
            try:
                llm.n_tokens = 0
                llm._ctx.memory_clear(True)
                if llm.is_hybrid and llm._hybrid_cache_mgr is not None:
                    llm._hybrid_cache_mgr.clear()
            except Exception:
                pass

    @staticmethod
    def _row_image_stats(row):
        images = []
        for group in GROUP_KEYS:
            item = row["selected"].get(group)
            if item is None:
                continue
            images.append({
                "group": group,
                "name": item.name,
                "encoded_size": item.encoded_size,
                "encoded_bytes": item.encoded_bytes,
            })
        return {
            "image_count": len(images),
            "total_bytes": sum(int(item.get("encoded_bytes", 0)) for item in images),
            "images": images,
        }

    @staticmethod
    def _build_messages(system_prompt, user_text, row=None, roles=None, max_side=1024):
        messages = []
        if (system_prompt or "").strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        if row is None:
            messages.append({"role": "user", "content": user_text})
            return messages

        content = [{"type": "text", "text": user_text}]
        for group in GROUP_KEYS:
            item = row["selected"].get(group)
            if item is None:
                continue
            content.append({"type": "text", "text": f"{group}组（{roles[group]}）：{item.name}"})
            content.append(item.to_image_part(max_side))
        messages.append({"role": "user", "content": content})
        return messages

    def _call_llm(self, llm, messages, params, think_mode):
        mm.throw_exception_if_processing_interrupted()
        resp = llm.create_chat_completion(messages=messages, **params)
        text = self._extract_response_text(resp)
        if not think_mode:
            text = _strip_think_blocks(text)
        return self._clean_prompt(text), resp

    def _run_row_task(self, llm, row, total_count, system_prompt, meta_instruction, roles, params, retry_count, think_mode, max_side):
        user_text = self._build_row_user_text(meta_instruction, row, total_count, roles)
        started_at = time.time()
        last_error = None
        last_trace = ""
        for attempt in range(retry_count + 1):
            try:
                encode_start = time.time()
                messages = self._build_messages(system_prompt, user_text, row, roles, max_side)
                encode_seconds = time.time() - encode_start
                infer_start = time.time()
                prompt, raw = self._call_llm(llm, messages, params, think_mode)
                infer_seconds = time.time() - infer_start
                if not prompt:
                    raise RuntimeError("本地 Llama 返回内容为空。")
                return {
                    "index": row["index"],
                    "ok": True,
                    "prompt": prompt,
                    "response": raw,
                    "attempts": attempt + 1,
                    "elapsed_seconds": round(time.time() - started_at, 3),
                    "timing": {
                        "encode_seconds": round(encode_seconds, 3),
                        "request_seconds": round(infer_seconds, 3),
                        "image_stats": self._row_image_stats(row),
                    },
                    "error": "",
                }
            except Exception as e:
                last_error = e
                last_trace = traceback.format_exc()
                if attempt < retry_count:
                    _log_info(f"第 {row['index']} 项第 {attempt + 1} 次失败，准备重试：{e}")
                    time.sleep(min(5, 1 + attempt * 2))

        return {
            "index": row["index"],
            "ok": False,
            "prompt": "",
            "response": None,
            "attempts": retry_count + 1,
            "elapsed_seconds": round(time.time() - started_at, 3),
            "timing": {},
            "error": str(last_error),
            "traceback": last_trace,
        }

    def _generate_text_only_prompts(self, llm, model_file, system_prompt, meta_instruction, params, default_count, seed, think_mode):
        prompt_count = self._detect_text_prompt_count(system_prompt, meta_instruction, default_count=default_count)
        user_text = "\n".join([
            (meta_instruction or "").strip(),
            "",
            "【输出要求】",
            f"请生成 {prompt_count} 条彼此不同的提示词。",
            "必须只输出一个 JSON 字符串数组，格式为：[\"提示词1\", \"提示词2\", ...]。",
            f"数组长度必须严格等于 {prompt_count}。",
            "每个数组元素是一条可以直接给下游图像生成模型使用的完整提示词。",
            "不要输出 Markdown、编号、解释或 JSON 对象。",
        ]).strip()
        final_system = "\n\n".join(part for part in [system_prompt.strip(), TEXT_ONLY_SYSTEM_HINT] if part)

        started = time.time()
        raw_text, raw = self._call_llm(llm, self._build_messages(final_system, user_text), params, think_mode)
        prompts = self._extract_prompt_list(raw_text, prompt_count)
        if len(prompts) != prompt_count:
            raise RuntimeError(f"无图文本模式要求输出 {prompt_count} 条提示词，但只解析到 {len(prompts)} 条。请在元指令中要求输出 JSON 字符串数组，或提高最大输出token。")

        elapsed = time.time() - started
        full_response = {
            "mode": "text_only",
            "model": model_file,
            "prompts": prompts,
            "raw_response": raw,
            "parsed_text": raw_text,
            "prompt_count": prompt_count,
            "elapsed_seconds": round(elapsed, 3),
        }
        info = "\n".join([
            "✅ Llama 批量提示词完成",
            "🧭 模式：无图文本批量生成",
            f"🤖 模型文件：{model_file}",
            f"📝 提示词数量：{prompt_count}",
            "🖼️ 图像输入数量：A=0，B=0，C=0，D=0",
            f"🎲 随机种子：{seed}",
            f"⏱️ 总耗时：{elapsed:.2f} 秒",
        ])
        return prompts, json.dumps(full_response, ensure_ascii=False, indent=2), info

    def generate_batch_prompts(self, unique_id=None, **kwargs):
        model_file = self._text_input_value(kwargs, "🤖模型文件", "")
        handler_name = self._text_input_value(kwargs, "🔌对话处理器", "None")
        mmproj_file = self._text_input_value(kwargs, "🖼️mmproj文件", "None")
        n_ctx = self._int_input_value(kwargs, "📐上下文长度", 8192)
        vram_limit_gb = self._float_input_value(kwargs, "💾显存限制(GB)", -1)
        image_min_tokens = self._int_input_value(kwargs, "🔢图像最小token", 256)
        image_max_tokens = self._int_input_value(kwargs, "🔢图像最大token", 1344)
        system_prompt = self._text_input_value(kwargs, "📝系统提示词", DEFAULT_SYSTEM_PROMPT)
        meta_instruction = self._text_input_value(kwargs, "🧾元指令", DEFAULT_META_INSTRUCTION)
        strategy = self._text_input_value(kwargs, "🧩缺失处理策略", "严格报错")
        default_count = self._int_input_value(kwargs, "🔢无图默认数量", 1)
        image_mode_max_prompts = self._int_input_value(kwargs, "🛡️多图模式最大提示词数量", 0)
        image_inference_mode = self._text_input_value(kwargs, "🚦多图推理模式", "逐条推理")
        retry_count = max(0, min(5, self._int_input_value(kwargs, "🛟失败重试次数", 0)))
        task_failure_strategy = self._text_input_value(kwargs, "🧪推理失败策略", "失败占位继续")
        max_side = max(64, min(4096, self._int_input_value(kwargs, "📏图像最大边长", 1024)))
        seed = self._int_input_value(kwargs, "🎲随机种子", 0)
        seed_mode = self._text_input_value(kwargs, "🎰随机化", "固定种子")
        max_tokens = self._int_input_value(kwargs, "📊最大输出token", 1024)
        temperature = self._float_input_value(kwargs, "🌡️温度", 0.7)
        top_p = self._float_input_value(kwargs, "🎯top_p", 0.9)
        top_k = self._int_input_value(kwargs, "🔝top_k", 40)
        repeat_penalty = self._float_input_value(kwargs, "🔁重复惩罚", 1.1)
        think_mode = self._bool_input_value(kwargs, "🧠思考模式", False)
        force_offload = self._bool_input_value(kwargs, "⚡推理后卸载模型", False)

        if not model_file:
            raise ValueError("模型文件为空，请先选择 LLM 模型。")
        if strategy not in MISSING_STRATEGIES:
            strategy = "严格报错"
        if task_failure_strategy not in TASK_FAILURE_STRATEGIES:
            task_failure_strategy = "失败占位继续"
        if image_inference_mode not in IMAGE_INFERENCE_MODES:
            image_inference_mode = "逐条推理"

        if seed_mode == "随机种子":
            seed = random.randint(0, 0xFFFFFFFF)
        elif seed_mode == "递增种子":
            seed = (seed + 1) % 0xFFFFFFFF
        elif seed_mode == "递减种子":
            seed = (seed - 1) % 0xFFFFFFFF

        roles = {
            "A": self._text_input_value(kwargs, "🏷️A组角色", DEFAULT_GROUP_ROLES["A"]),
            "B": self._text_input_value(kwargs, "🏷️B组角色", DEFAULT_GROUP_ROLES["B"]),
            "C": self._text_input_value(kwargs, "🏷️C组角色", DEFAULT_GROUP_ROLES["C"]),
            "D": self._text_input_value(kwargs, "🏷️D组角色", DEFAULT_GROUP_ROLES["D"]),
        }

        params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "seed": seed,
        }

        start_time = time.time()
        llm = self._ensure_model(model_file, handler_name, mmproj_file, n_ctx, vram_limit_gb, image_min_tokens, image_max_tokens, think_mode)

        groups = {}
        source_report = {}
        for group in GROUP_KEYS:
            groups[group], source_report[group] = self._collect_group_items(kwargs, group)

        has_any_images = any(groups[group] for group in GROUP_KEYS)
        if not has_any_images:
            result = self._generate_text_only_prompts(llm, model_file, system_prompt, meta_instruction, params, default_count, seed, think_mode)
            if force_offload:
                DapaoLlamaStorage.clean()
                mm.soft_empty_cache()
            else:
                self._clear_local_cache_if_needed(llm, handler_name)
            return result

        rows, original_anchor_count, image_mode_limit = self._build_alignment(groups, roles, strategy, image_mode_max_prompts)
        total_count = len(rows)
        prompts = [""] * total_count
        raw_responses = []
        task_results = [None] * total_count

        _log_info(f"开始本地批量提示词生成：模型 {model_file}，任务 {total_count} 条，模式 {image_inference_mode}，缺失策略 {strategy}")

        if image_inference_mode == "单次批量请求":
            batch_text = self._build_batch_user_text(meta_instruction, rows, roles)
            batch_row = {
                "index": 1,
                "selected": {},
            }
            content_rows = []
            for row in rows:
                content_rows.append({"type": "text", "text": f"任务 {row['index']} 开始"})
                for group in GROUP_KEYS:
                    item = row["selected"].get(group)
                    if item is None:
                        continue
                    content_rows.append({"type": "text", "text": f"任务 {row['index']} - {group}组（{roles[group]}）：{item.name}"})
                    content_rows.append(item.to_image_part(max_side))
                content_rows.append({"type": "text", "text": f"任务 {row['index']} 结束"})
            messages = []
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt.strip()})
            messages.append({"role": "user", "content": [{"type": "text", "text": batch_text}] + content_rows})
            started = time.time()
            response_text, raw = self._call_llm(llm, messages, params, think_mode)
            prompts = self._extract_prompt_list(response_text, total_count)
            if len(prompts) != total_count:
                raise RuntimeError(f"单次批量请求要求返回 {total_count} 条提示词，但只解析到 {len(prompts)} 条。可以改用“逐条推理”，或提高最大输出token。")
            request_elapsed = time.time() - started
            raw_responses.append({"index": "batch", "response": raw, "parsed_text": response_text})
            for idx, row in enumerate(rows):
                row["prompt"] = prompts[idx]
                row["error"] = ""
                row["attempts"] = 1
                row["elapsed_seconds"] = round(request_elapsed, 3)
                row["timing"] = {"request_seconds": round(request_elapsed, 3)}
                task_results[idx] = {
                    "index": row["index"],
                    "ok": True,
                    "prompt": prompts[idx],
                    "attempts": 1,
                    "elapsed_seconds": round(request_elapsed, 3),
                    "timing": row["timing"],
                    "error": "",
                }
            success_count = total_count
            failed_count = 0
        else:
            abort_error = None
            for idx, row in enumerate(rows):
                result = self._run_row_task(llm, row, total_count, system_prompt, meta_instruction, roles, params, retry_count, think_mode, max_side)
                task_results[idx] = result
                row["prompt"] = result.get("prompt", "")
                row["error"] = result.get("error", "")
                row["attempts"] = result.get("attempts", 1)
                row["elapsed_seconds"] = result.get("elapsed_seconds")
                row["timing"] = result.get("timing", {})

                if result.get("ok"):
                    prompts[idx] = result["prompt"]
                    raw_responses.append({"index": row["index"], "response": result.get("response")})
                    _log_info(f"第 {row['index']}/{total_count} 项完成，提示词长度 {len(result['prompt'])}，耗时 {result.get('elapsed_seconds')} 秒")
                else:
                    _log_error(f"第 {row['index']} 项失败：{result.get('error')}")
                    if result.get("traceback"):
                        _log_error(result["traceback"])
                    if task_failure_strategy == "任一失败中断":
                        abort_error = result
                        break
                    if task_failure_strategy == "失败占位继续":
                        prompts[idx] = f"ERROR: 第 {row['index']} 项提示词生成失败：{result.get('error')}"

            if abort_error:
                raise RuntimeError(f"第 {abort_error['index']}/{total_count} 项 Llama 提示词生成失败，已按策略中断：{abort_error.get('error')}")

            if task_failure_strategy == "跳过失败继续":
                prompts = [result.get("prompt", "") for result in task_results if result and result.get("ok") and result.get("prompt")]

            success_count = sum(1 for result in task_results if result and result.get("ok"))
            failed_count = sum(1 for result in task_results if result and not result.get("ok"))

        elapsed_time = time.time() - start_time
        alignment_report = {
            "status": "success",
            "node": NODE_NAME,
            "model": model_file,
            "prompt_count": len(prompts),
            "task_count": total_count,
            "success_count": success_count,
            "failed_count": failed_count,
            "image_inference_mode": image_inference_mode,
            "retry_count": retry_count,
            "task_failure_strategy": task_failure_strategy,
            "image_max_side": max_side,
            "original_anchor_count": original_anchor_count,
            "image_mode_max_prompt_count": image_mode_limit,
            "truncated_by_limit": bool(image_mode_limit > 0 and original_anchor_count > len(rows)),
            "anchor_group": "A",
            "missing_strategy": strategy,
            "roles": roles,
            "sources": source_report,
            "seed": seed,
            "elapsed_seconds": round(elapsed_time, 3),
            "rows": [
                {
                    "index": row["index"],
                    "groups": row["groups"],
                    "prompt": row.get("prompt", ""),
                    "prompt_length": len(row.get("prompt", "")),
                    "attempts": row.get("attempts", 1),
                    "elapsed_seconds": row.get("elapsed_seconds"),
                    "timing": row.get("timing", {}),
                    "error": row.get("error", ""),
                }
                for row in rows
            ],
        }
        full_response = {
            "prompts": prompts,
            "alignment_report": alignment_report,
            "raw_responses": raw_responses,
            "task_results": [{key: value for key, value in (result or {}).items() if key != "response"} for result in task_results],
        }

        if force_offload:
            DapaoLlamaStorage.clean()
            mm.soft_empty_cache()
        else:
            self._clear_local_cache_if_needed(llm, handler_name)

        info_lines = [
            "✅ Llama 批量提示词完成",
            f"🤖 模型文件：{model_file}",
            f"📝 提示词数量：{len(prompts)}",
            f"🔢 实际任务数量：{total_count}",
            f"🚦 多图推理模式：{image_inference_mode}",
            f"🛟 失败重试次数：{retry_count}",
            f"🧪 推理失败策略：{task_failure_strategy}",
            f"✅ 成功任务：{success_count}",
            f"❌ 失败任务：{failed_count}",
            f"📏 图像最大边长：{max_side}",
            f"🛡️ 多图模式最大提示词数量：{image_mode_limit if image_mode_limit > 0 else '不限制'}",
            f"⚓ A组原始数量：{original_anchor_count}",
            f"🧩 缺失处理策略：{strategy}",
            f"🖼️ 图像输入数量：A={source_report['A']['count']}，B={source_report['B']['count']}，C={source_report['C']['count']}，D={source_report['D']['count']}",
            f"🎲 随机种子：{seed}",
            f"⏱️ 总耗时：{elapsed_time:.2f} 秒",
            "📋 对齐明细：",
        ]
        for row in rows:
            parts = []
            for group in GROUP_KEYS:
                image = row["groups"][group]["image"]
                method = row["groups"][group]["match_method"]
                parts.append(f"{group}={image['name'] if image else '空'}({method})")
            timing = row.get("timing") or {}
            timing_text = ""
            if "image_stats" in timing:
                stats = timing.get("image_stats") or {}
                size_kb = int(stats.get("total_bytes", 0)) / 1024
                timing_text = f"，图片 {stats.get('image_count', 0)} 张/{size_kb:.1f}KB，推理 {timing.get('request_seconds', '未知')}秒"
            info_lines.append(f"#{row['index']} " + "，".join(parts) + timing_text)

        return prompts, json.dumps(full_response, ensure_ascii=False, indent=2), "\n".join(info_lines)


NODE_CLASS_MAPPINGS_BATCH_PROMPT = {
    NODE_NAME: Dapao_LlamaBatchPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS_BATCH_PROMPT = {
    NODE_NAME: "💓Llama批量提示词@炮老师的小课堂",
}

"""
😶‍🌫️llama图片反推节点
基于 Dapao_LlamaChat 的推理逻辑，内置图片反推元指令，
支持多种提示词风格和额外选项。
"""
import os
import io
import gc
import base64
import random
import re

import numpy as np
import torch
from PIL import Image

import folder_paths
import comfy.model_management as mm

from .gguf_layers import get_layer_count
from .cqdm import cqdm
from .caption_options import Dapao_LlamaCaptionOptions
# 复用主节点的存储和工具函数
from .nodes import (
    DapaoLlamaStorage, CHAT_HANDLERS, AnyType,
    tensor2pil, scale_image, image2base64,
)

any_type = AnyType("*")

# ── prompt 文件夹：动态扫描 .txt 作为反推提示词风格 ─────────────────────────
_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompt")


def _scan_prompt_styles():
    """扫描 prompt/ 文件夹下所有 .txt 文件，文件名(去掉.txt)作为风格名，内容作为提示词"""
    styles = {}
    if not os.path.isdir(_PROMPT_DIR):
        os.makedirs(_PROMPT_DIR, exist_ok=True)
        return styles
    for fname in sorted(os.listdir(_PROMPT_DIR)):
        if not fname.lower().endswith(".txt"):
            continue
        style_name = fname[:-4]  # 去掉 .txt
        fpath = os.path.join(_PROMPT_DIR, fname)
        try:
            for enc in ("utf-8", "gbk", "utf-16", "latin-1"):
                try:
                    with open(fpath, "r", encoding=enc) as f:
                        content = f.read().strip()
                    break
                except (UnicodeDecodeError, UnicodeError):
                    content = ""
            if content:
                styles[style_name] = content
        except Exception as e:
            print(f"[大炮-llama] 读取提示词文件失败: {fname} -> {e}")
    return styles


def _get_style_keys():
    """获取当前可用的风格列表，供 INPUT_TYPES 使用"""
    styles = _scan_prompt_styles()
    if not styles:
        return ["(prompt文件夹为空)"]
    return list(styles.keys())


class Dapao_LlamaCaption:
    CATEGORY = "🍭大炮-llama-cpp"
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("💬反推文本", "📋完整对话历史", "🔢使用的种子")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        llm_files = folder_paths.get_filename_list("LLM")
        mmproj_files = ["None"] + [f for f in llm_files if "mmproj" in f.lower()]
        model_files = [f for f in llm_files if "mmproj" not in f.lower()]

        return {
            "required": {
                # ── 模型加载 ──
                "🤖模型文件": (model_files,),
                "🔌对话处理器": (CHAT_HANDLERS, {"default": "None"}),
                "🖼️mmproj文件": (mmproj_files, {"default": "None"}),
                "📐上下文长度": ("INT", {"default": 8192, "min": 512, "max": 131072, "step": 512}),
                "💾显存限制(GB)": ("FLOAT", {"default": -1, "min": -1, "max": 999.0, "step": 0.5}),
                "🔢图像最小token": ("INT", {"default": 256, "min": 1, "max": 4096, "step": 1}),
                "🔢图像最大token": ("INT", {"default": 1344, "min": 1, "max": 8192, "step": 1}),
                # ── 反推设置 ──
                "🎨提示词风格": (_get_style_keys(),),
                "💬附加指令": ("STRING", {"default": "", "multiline": True, "tooltip": "在风格提示词基础上追加的额外指令（可留空）"}),
                # ── 推理参数 ──
                "📏图像最大边长": ("INT", {"default": 1120, "min": 64, "max": 4096, "step": 32}),
                # ── 生成参数 ──
                "🎲随机种子": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "🎰随机化": (["固定种子", "随机种子", "递增种子", "递减种子"], {"default": "固定种子"}),
                "📊最大输出token": ("INT", {"default": 1024, "min": 1, "max": 32768, "step": 1}),
                "🌡️温度": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01}),
                "🎯top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "🔝top_k": ("INT", {"default": 40, "min": 0, "max": 200, "step": 1}),
                "🔁重复惩罚": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.01}),
                "🧠思考模式": ("BOOLEAN", {"default": False}),
                "⚡推理后卸载模型": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "🖼️图像1": ("IMAGE",),
                "🖼️图像2": ("IMAGE",),
                "🖼️图像3": ("IMAGE",),
                "🖼️图像4": ("IMAGE",),
                "🖼️图像5": ("IMAGE",),
                "🖼️图像6": ("IMAGE",),
                "🖼️图像7": ("IMAGE",),
                "🖼️图像8": ("IMAGE",),
                "🍭反推额外选项": ("LLAMA_CAPTION_OPTIONS",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def run(self, unique_id, **kwargs):
        model_file       = kwargs["🤖模型文件"]
        handler_name     = kwargs["🔌对话处理器"]
        mmproj_file      = kwargs["🖼️mmproj文件"]
        n_ctx            = kwargs["📐上下文长度"]
        vram_limit_gb    = kwargs["💾显存限制(GB)"]
        image_min_tokens = kwargs["🔢图像最小token"]
        image_max_tokens = kwargs["🔢图像最大token"]
        caption_style    = kwargs["🎨提示词风格"]
        extra_instruction= kwargs["💬附加指令"].strip()
        max_size         = kwargs["📏图像最大边长"]
        seed             = kwargs["🎲随机种子"]
        seed_mode        = kwargs["🎰随机化"]
        max_tokens       = kwargs["📊最大输出token"]
        temperature      = kwargs["🌡️温度"]
        top_p            = kwargs["🎯top_p"]
        top_k            = kwargs["🔝top_k"]
        repeat_penalty   = kwargs["🔁重复惩罚"]
        think_mode       = kwargs["🧠思考模式"]
        force_offload    = kwargs["⚡推理后卸载模型"]
        extra_options    = kwargs.get("🍭反推额外选项", None)
        uid              = str(unique_id)

        # ── 种子处理 ──────────────────────────────────────────────────────────
        if seed_mode == "随机种子":
            seed = random.randint(0, 0xFFFFFFFF)
        elif seed_mode == "递增种子":
            seed = (seed + 1) % 0xFFFFFFFF
        elif seed_mode == "递减种子":
            seed = (seed - 1) % 0xFFFFFFFF

        # ── 构建提示词（附加指令优先，为空时使用内置风格）────────────────────
        if extra_instruction:
            base_prompt = extra_instruction
        else:
            styles = _scan_prompt_styles()
            base_prompt = styles.get(caption_style, "")
            if not base_prompt:
                raise ValueError(f"[大炮-llama] 找不到提示词风格文件: {caption_style}.txt，请检查 prompt/ 文件夹")
        if extra_options:
            base_prompt = Dapao_LlamaCaptionOptions.build_enhanced_prompt(base_prompt, extra_options)

        # ── 收集图像 ──────────────────────────────────────────────────────────
        image_slots = [
            kwargs.get("🖼️图像1"), kwargs.get("🖼️图像2"),
            kwargs.get("🖼️图像3"), kwargs.get("🖼️图像4"),
            kwargs.get("🖼️图像5"), kwargs.get("🖼️图像6"),
            kwargs.get("🖼️图像7"), kwargs.get("🖼️图像8"),
        ]
        all_tensors = [t for t in image_slots if t is not None]

        frame_list = []
        for tensor in all_tensors:
            for i in range(tensor.shape[0]):
                pil_img = tensor2pil(tensor[i:i+1])
                pil_img = scale_image(pil_img, max_size)
                frame_list.append(pil_img)

        # ── 加载或复用模型 ────────────────────────────────────────────────────
        load_config = {
            "model_file": model_file, "handler_name": handler_name,
            "mmproj_file": mmproj_file, "n_ctx": n_ctx,
            "vram_limit_gb": vram_limit_gb,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens,
            "think_mode": think_mode,
        }
        need_load = DapaoLlamaStorage.llm is None or DapaoLlamaStorage.current_config != load_config

        if need_load:
            DapaoLlamaStorage.clean()
            mm.soft_empty_cache()
            _load_llm(model_file, handler_name, mmproj_file, n_ctx,
                      vram_limit_gb, image_min_tokens, image_max_tokens, think_mode)
        else:
            print("[大炮-llama] 复用已加载模型")

        llm = DapaoLlamaStorage.llm

        # ── 推理（每张图独立，反推场景通常逐图处理）────────────────────────
        _params = dict(
            max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k,
            repeat_penalty=repeat_penalty, seed=seed,
        )

        def _infer_single(pil_img):
            b64 = image2base64(pil_img)
            msgs = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": base_prompt},
                ]},
            ]
            mm.throw_exception_if_processing_interrupted()
            resp = llm.create_chat_completion(messages=msgs, **_params)
            text = (resp["choices"][0]["message"]["content"] or "").removeprefix(": ").lstrip()
            if not think_mode:
                m = re.search(r"<think>.*?</think>(.*)", text, re.DOTALL)
                if m:
                    text = m.group(1).strip()
                m = re.search(r"<\|channel>thought\n.*?<channel\|>(.*)", text, re.DOTALL)
                if m:
                    text = m.group(1).strip()
            return text

        print(f"[大炮-llama] 反推开始  seed={seed}  图像数={len(frame_list)}")

        if not frame_list:
            reply = _infer_single_text(llm, base_prompt, _params, think_mode)
        elif len(frame_list) == 1:
            reply = _infer_single(frame_list[0])
        else:
            replies = []
            for pil_img in cqdm(frame_list, desc="逐图反推"):
                replies.append(_infer_single(pil_img))
            reply = "\n\n".join(replies)

        # ── 推理后处理 ────────────────────────────────────────────────────────
        if force_offload:
            DapaoLlamaStorage.clean()
            mm.soft_empty_cache()
        elif handler_name in ("Qwen3.5", "Qwen3.5-Thinking", "Qwen3-VL", "Qwen3-VL-Thinking"):
            try:
                llm.n_tokens = 0
                llm._ctx.memory_clear(True)
                if llm.is_hybrid and llm._hybrid_cache_mgr is not None:
                    llm._hybrid_cache_mgr.clear()
            except Exception:
                pass

        history_text = f"[system]: {base_prompt}\n[user]: <图像>\n[assistant]: {reply}"
        return (reply, history_text, seed)


def _infer_single_text(llm, prompt, params, think_mode):
    """无图像时纯文本推理"""
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    resp = llm.create_chat_completion(messages=msgs, **params)
    text = (resp["choices"][0]["message"]["content"] or "").removeprefix(": ").lstrip()
    if not think_mode:
        m = re.search(r"<think>.*?</think>(.*)", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    return text


def _load_llm(model_file, handler_name, mmproj_file, n_ctx, vram_limit_gb,
              image_min_tokens, image_max_tokens, think_mode):
    """复用 nodes.py 的加载逻辑"""
    from .nodes import Dapao_LlamaChat
    node = Dapao_LlamaChat()
    node._load_model(model_file, handler_name, mmproj_file, n_ctx,
                     vram_limit_gb, image_min_tokens, image_max_tokens, think_mode)


NODE_CLASS_MAPPINGS_CAPTION = {
    "Dapao_LlamaCaption": Dapao_LlamaCaption,
}

NODE_DISPLAY_NAME_MAPPINGS_CAPTION = {
    "Dapao_LlamaCaption": "😶‍🌫️llama图片反推@炮老师的小课堂",
}

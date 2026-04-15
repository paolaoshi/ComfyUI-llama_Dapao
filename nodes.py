import os
import io
import gc
import json
import base64
import random

import numpy as np
import torch
from PIL import Image

import folder_paths
import comfy.model_management as mm

from .gguf_layers import get_layer_count
from .cqdm import cqdm

# ── LLM 文件夹注册 ──────────────────────────────────────────────────────────
llm_extensions = {".gguf", ".bin"}
llm_dir = os.path.join(folder_paths.models_dir, "LLM")
os.makedirs(llm_dir, exist_ok=True)
folder_paths.folder_names_and_paths["LLM"] = ([llm_dir], llm_extensions)

# ── AnyType ─────────────────────────────────────────────────────────────────
class AnyType(str):
    def __ne__(self, other):
        return False

any_type = AnyType("*")

# ── 按需 import 各 ChatHandler（兼容不同版本 llama-cpp-python）──────────────
from llama_cpp import Llama
from llama_cpp.llama_chat_format import (
    Llava15ChatHandler, Llava16ChatHandler, MoondreamChatHandler,
    NanoLlavaChatHandler, Llama3VisionAlphaChatHandler, MiniCPMv26ChatHandler,
)

try:
    from llama_cpp.llama_chat_format import MTMDChatHandler
    _MTMD = True
except Exception:
    _MTMD = False

try:
    from llama_cpp.llama_chat_format import Gemma3ChatHandler
except Exception:
    Gemma3ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Gemma4ChatHandler
except Exception:
    if _MTMD:
        Gemma4ChatHandler = MTMDChatHandler
    else:
        Gemma4ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
except Exception:
    Qwen25VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen3VLChatHandler
except Exception:
    Qwen3VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen35ChatHandler
except Exception:
    Qwen35ChatHandler = None

try:
    from llama_cpp.llama_chat_format import GLM46VChatHandler, GLM41VChatHandler
except Exception:
    GLM46VChatHandler = None
    GLM41VChatHandler = None

try:
    from llama_cpp.llama_chat_format import LFM2VLChatHandler
except Exception:
    LFM2VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import GraniteDoclingChatHandler
except Exception:
    GraniteDoclingChatHandler = None

# ── 可用 handler 列表（固定顺序，始终包含全部，运行时若 import 失败会报错）──
CHAT_HANDLERS = [
    "None",
    "LLaVA-1.5", "LLaVA-1.6", "Moondream2", "nanoLLaVA", "llama3-Vision-Alpha",
    "MiniCPM-v2.6", "MiniCPM-v4.5", "MiniCPM-v4.5-Thinking",
    "Gemma3", "Gemma4",
    "Qwen2.5-VL",
    "Qwen3-VL", "Qwen3-VL-Thinking",
    "Qwen3.5", "Qwen3.5-Thinking",
    "GLM-4.6V", "GLM-4.6V-Thinking", "GLM-4.1V-Thinking",
    "LFM2-VL",
    "Granite-Docling",
]

# ── 模型状态管理 ─────────────────────────────────────────────────────────────
class DapaoLlamaStorage:
    llm = None
    chat_handler = None
    current_config = None
    messages = {}
    sys_prompts = {}

    @classmethod
    def clean(cls, all=False):
        if cls.llm is not None:
            del cls.llm
            cls.llm = None
        cls.chat_handler = None
        cls.current_config = None
        if all:
            cls.messages = {}
            cls.sys_prompts = {}
        gc.collect()

    @classmethod
    def clean_state(cls, uid=-1):
        if uid == -1:
            cls.messages = {}
            cls.sys_prompts = {}
        else:
            cls.messages.pop(str(uid), None)
            cls.sys_prompts.pop(str(uid), None)


# ── patch mm.unload_all_models（只 patch 一次）──────────────────────────────
if not hasattr(mm, "_dapao_llama_unload_backup"):
    mm._dapao_llama_unload_backup = mm.unload_all_models
    def _patched_unload(*args, **kwargs):
        DapaoLlamaStorage.clean(all=True)
        return mm._dapao_llama_unload_backup(*args, **kwargs)
    mm.unload_all_models = _patched_unload
    print("[大炮-llama] 模型卸载钩子已挂载")

# ── 图像工具 ─────────────────────────────────────────────────────────────────
def tensor2pil(tensor):
    """[1,H,W,C] float32 0-1 → PIL Image"""
    img = tensor.squeeze(0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def scale_image(pil_img, max_size):
    w, h = pil_img.size
    if max(w, h) <= max_size:
        return pil_img
    scale = max_size / max(w, h)
    return pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def image2base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def audio2base64(audio_dict):
    """ComfyUI AUDIO dict → base64 WAV string（纯 Python wave 模块，不依赖 torchaudio）"""
    import wave
    import struct
    waveform = audio_dict["waveform"]   # [B, C, T]
    sample_rate = audio_dict["sample_rate"]
    wav = waveform[0].cpu()             # [C, T]
    # 混合为单声道或保留立体声
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav[0]  # [T]
    # 转为 16-bit PCM
    pcm = (wav.clamp(-1.0, 1.0) * 32767).short().numpy()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── 主节点 ───────────────────────────────────────────────────────────────────
class Dapao_LlamaChat:
    CATEGORY = "🍭大炮-llama-cpp"
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("💬回复文本", "📋完整对话历史", "🔢使用的种子")
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
                "💾显存限制(GB)": ("FLOAT", {"default": -1, "min": -1, "max": 999.0, "step": 0.5, "tooltip": "-1 表示不限制（全部放 GPU）"}),
                "🔢图像最小token": ("INT", {"default": 256, "min": 1, "max": 4096, "step": 1}),
                "🔢图像最大token": ("INT", {"default": 1344, "min": 1, "max": 8192, "step": 1}),
                # ── 提示词 ──
                "📝系统提示词": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "💬用户提示词": ("STRING", {"default": "请描述这张图片。", "multiline": True}),
                # ── 推理参数 ──
                "🎯推理模式": (["one by one", "images", "video"], {"default": "one by one"}),
                "🎞️最大帧数": ("INT", {"default": 10, "min": 1, "max": 200, "step": 1}),
                "📏图像最大边长": ("INT", {"default": 1120, "min": 64, "max": 4096, "step": 32}),
                # ── 生成参数 ──
                "🎲随机种子": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "🎰随机化": (["固定种子", "随机种子", "递增种子", "递减种子"], {"default": "固定种子"}),
                "📊最大输出token": ("INT", {"default": 1024, "min": 1, "max": 32768, "step": 1}),
                "🌡️温度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "🎯top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "🔝top_k": ("INT", {"default": 40, "min": 0, "max": 200, "step": 1}),
                "🔁重复惩罚": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.01}),
                "🧠思考模式": ("BOOLEAN", {"default": False, "tooltip": "开启后模型会输出思考过程（仅 Thinking 系列模型有效）"}),
                "💾保存对话历史": ("BOOLEAN", {"default": False}),
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
                "🎬视频1": ("IMAGE",),
                "🎬视频2": ("IMAGE",),
                "🔊音频1": ("AUDIO",),
                "🔊音频2": ("AUDIO",),
                "🔗队列处理器": (any_type,),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def _load_model(self, model_file, handler_name, mmproj_file, n_ctx, vram_limit_gb,
                    image_min_tokens, image_max_tokens, think_mode=False):
        model_path = os.path.join(folder_paths.models_dir, "LLM", model_file)
        mmproj_path = None
        if mmproj_file and mmproj_file != "None":
            mmproj_path = os.path.join(folder_paths.models_dir, "LLM", mmproj_file)

        # ── n_gpu_layers 计算（与参考节点一致，含 1.55 系数）────────────────
        n_gpu_layers = -1
        if vram_limit_gb != -1:
            layer_count = get_layer_count(model_path) or 32
            model_size_gb = os.path.getsize(model_path) * 1.55 / (1024 ** 3)
            layer_size_gb = model_size_gb / layer_count

            if mmproj_path:
                mmproj_size_gb = os.path.getsize(mmproj_path) * 1.55 / (1024 ** 3)
                n_gpu_layers = max(1, int((vram_limit_gb - mmproj_size_gb) / layer_size_gb))
            else:
                n_gpu_layers = max(1, int(vram_limit_gb / layer_size_gb))

        print(f"[大炮-llama] 加载模型: {model_file}  n_gpu_layers={n_gpu_layers}")

        # ── 实例化 ChatHandler ────────────────────────────────────────────────
        chat_handler = None
        # think_mode 由外部参数控制，不再依赖 handler 名字里的 "Thinking" 后缀

        if mmproj_path and handler_name != "None":
            kwargs = {"clip_model_path": mmproj_path, "verbose": False}

            if handler_name in ("Qwen3-VL", "Qwen3-VL-Thinking"):
                if Qwen3VLChatHandler is None:
                    raise RuntimeError("Qwen3VLChatHandler 未找到，请升级 llama-cpp-python")
                kwargs["force_reasoning"] = think_mode
                kwargs["image_max_tokens"] = image_max_tokens
                kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = Qwen3VLChatHandler(**kwargs)

            elif handler_name == "Qwen2.5-VL":
                if Qwen25VLChatHandler is None:
                    raise RuntimeError("Qwen25VLChatHandler 未找到，请升级 llama-cpp-python")
                if _MTMD:
                    kwargs["image_max_tokens"] = image_max_tokens
                    kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = Qwen25VLChatHandler(**kwargs)

            elif handler_name in ("Qwen3.5", "Qwen3.5-Thinking"):
                if Qwen35ChatHandler is None:
                    raise RuntimeError("Qwen35ChatHandler 未找到，请升级 llama-cpp-python")
                kwargs["enable_thinking"] = think_mode
                if _MTMD:
                    kwargs["image_max_tokens"] = image_max_tokens
                    kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = Qwen35ChatHandler(**kwargs)

            elif handler_name in ("MiniCPM-v4.5", "MiniCPM-v4.5-Thinking"):
                kwargs["enable_thinking"] = think_mode
                if _MTMD:
                    kwargs["image_max_tokens"] = image_max_tokens
                    kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = MiniCPMv26ChatHandler(**kwargs)

            elif handler_name == "Gemma3":
                if Gemma3ChatHandler is None:
                    raise RuntimeError("Gemma3ChatHandler 未找到，请升级 llama-cpp-python")
                if _MTMD:
                    kwargs["image_max_tokens"] = image_max_tokens
                    kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = Gemma3ChatHandler(**kwargs)

            elif handler_name == "Gemma4":
                if Gemma4ChatHandler is None:
                    raise RuntimeError("Gemma4ChatHandler 未找到，请升级 llama-cpp-python")
                kwargs["enable_thinking"] = think_mode
                kwargs["image_max_tokens"] = image_max_tokens
                kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = Gemma4ChatHandler(**kwargs)

            elif handler_name in ("GLM-4.6V", "GLM-4.6V-Thinking"):
                if GLM46VChatHandler is None:
                    raise RuntimeError("GLM46VChatHandler 未找到，请升级 llama-cpp-python")
                kwargs["enable_thinking"] = think_mode
                if _MTMD:
                    kwargs["image_max_tokens"] = image_max_tokens
                    kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = GLM46VChatHandler(**kwargs)

            elif handler_name == "GLM-4.1V-Thinking":
                if GLM41VChatHandler is None:
                    raise RuntimeError("GLM41VChatHandler 未找到，请升级 llama-cpp-python")
                kwargs["enable_thinking"] = think_mode
                if _MTMD:
                    kwargs["image_max_tokens"] = image_max_tokens
                    kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = GLM41VChatHandler(**kwargs)

            elif handler_name == "LFM2-VL":
                if LFM2VLChatHandler is None:
                    raise RuntimeError("LFM2VLChatHandler 未找到，请升级 llama-cpp-python")
                if _MTMD:
                    kwargs["image_max_tokens"] = image_max_tokens
                    kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = LFM2VLChatHandler(**kwargs)

            elif handler_name == "Granite-Docling":
                if GraniteDoclingChatHandler is None:
                    raise RuntimeError("GraniteDoclingChatHandler 未找到，请升级 llama-cpp-python")
                if _MTMD:
                    kwargs["image_max_tokens"] = image_max_tokens
                    kwargs["image_min_tokens"] = image_min_tokens
                chat_handler = GraniteDoclingChatHandler(**kwargs)

            elif handler_name == "LLaVA-1.5":
                chat_handler = Llava15ChatHandler(**kwargs)
            elif handler_name == "LLaVA-1.6":
                chat_handler = Llava16ChatHandler(**kwargs)
            elif handler_name == "Moondream2":
                chat_handler = MoondreamChatHandler(**kwargs)
            elif handler_name == "nanoLLaVA":
                chat_handler = NanoLlavaChatHandler(**kwargs)
            elif handler_name == "llama3-Vision-Alpha":
                chat_handler = Llama3VisionAlphaChatHandler(**kwargs)
            elif handler_name == "MiniCPM-v2.6":
                chat_handler = MiniCPMv26ChatHandler(**kwargs)

        elif handler_name not in ("None", "Qwen3.5", "Qwen3.5-Thinking"):
            # 无 mmproj 但有 handler（纯文本模式下某些 handler 可无 mmproj）
            pass

        # ── 加载 Llama ────────────────────────────────────────────────────────
        llm = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )
        DapaoLlamaStorage.llm = llm
        DapaoLlamaStorage.chat_handler = chat_handler
        DapaoLlamaStorage.current_config = {
            "model_file": model_file,
            "handler_name": handler_name,
            "mmproj_file": mmproj_file,
            "n_ctx": n_ctx,
            "vram_limit_gb": vram_limit_gb,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens,
            "think_mode": think_mode,
        }
        return llm

    def run(self, unique_id, **kwargs):
        model_file       = kwargs["🤖模型文件"]
        handler_name     = kwargs["🔌对话处理器"]
        mmproj_file      = kwargs["🖼️mmproj文件"]
        n_ctx            = kwargs["📐上下文长度"]
        vram_limit_gb    = kwargs["💾显存限制(GB)"]
        image_min_tokens = kwargs["🔢图像最小token"]
        image_max_tokens = kwargs["🔢图像最大token"]
        system_prompt    = kwargs["📝系统提示词"]
        user_prompt      = kwargs["💬用户提示词"]
        inference_mode   = kwargs["🎯推理模式"]
        max_frames       = kwargs["🎞️最大帧数"]
        max_size         = kwargs["📏图像最大边长"]
        seed             = kwargs["🎲随机种子"]
        seed_mode        = kwargs["🎰随机化"]
        max_tokens       = kwargs["📊最大输出token"]
        temperature      = kwargs["🌡️温度"]
        top_p            = kwargs["🎯top_p"]
        top_k            = kwargs["🔝top_k"]
        repeat_penalty   = kwargs["🔁重复惩罚"]
        think_mode       = kwargs["🧠思考模式"]
        save_states      = kwargs["💾保存对话历史"]
        force_offload    = kwargs["⚡推理后卸载模型"]
        uid              = str(unique_id)

        # ── 收集所有图像输入（8个图像口 + 2个视频口）────────────────────────
        image_slots = [
            kwargs.get("🖼️图像1"), kwargs.get("🖼️图像2"),
            kwargs.get("🖼️图像3"), kwargs.get("🖼️图像4"),
            kwargs.get("🖼️图像5"), kwargs.get("🖼️图像6"),
            kwargs.get("🖼️图像7"), kwargs.get("🖼️图像8"),
        ]
        video_slots = [kwargs.get("🎬视频1"), kwargs.get("🎬视频2")]
        audio_slots = [kwargs.get("🔊音频1"), kwargs.get("🔊音频2")]

        # 各 tensor 单独保留，不拼接（不同尺寸图片无法 cat）
        all_image_tensors = [t for t in image_slots + video_slots if t is not None]

        # ── 种子处理 ──────────────────────────────────────────────────────────
        if seed_mode == "随机种子":
            seed = random.randint(0, 0xFFFFFFFF)
        elif seed_mode == "递增种子":
            seed = (seed + 1) % 0xFFFFFFFF
        elif seed_mode == "递减种子":
            seed = (seed - 1) % 0xFFFFFFFF

        # ── 加载或复用模型 ────────────────────────────────────────────────────
        need_load = DapaoLlamaStorage.llm is None
        if not need_load:
            cfg = DapaoLlamaStorage.current_config or {}
            need_load = (
                cfg.get("model_file") != model_file
                or cfg.get("handler_name") != handler_name
                or cfg.get("mmproj_file") != mmproj_file
                or cfg.get("n_ctx") != n_ctx
                or cfg.get("vram_limit_gb") != vram_limit_gb
                or cfg.get("image_min_tokens") != image_min_tokens
                or cfg.get("image_max_tokens") != image_max_tokens
                or cfg.get("think_mode") != think_mode
            )

        if need_load:
            DapaoLlamaStorage.clean()
            mm.soft_empty_cache()
            self._load_model(model_file, handler_name, mmproj_file, n_ctx,
                             vram_limit_gb, image_min_tokens, image_max_tokens, think_mode)
            DapaoLlamaStorage.clean_state(uid)
        else:
            print("[大炮-llama] 复用已加载模型")

        llm = DapaoLlamaStorage.llm

        # ── 对话历史管理 ──────────────────────────────────────────────────────
        prev_sys = DapaoLlamaStorage.sys_prompts.get(uid, "")
        if not save_states or prev_sys != system_prompt:
            DapaoLlamaStorage.clean_state(uid)
            DapaoLlamaStorage.sys_prompts[uid] = system_prompt

        messages = DapaoLlamaStorage.messages.get(uid, [])
        if not messages:
            messages = [{"role": "system", "content": system_prompt}]

        # ── 构建图像帧列表 ────────────────────────────────────────────────────
        frame_list = []
        for tensor in all_image_tensors:
            for i in range(tensor.shape[0]):
                if len(frame_list) >= max_frames:
                    break
                pil_img = tensor2pil(tensor[i:i+1])
                pil_img = scale_image(pil_img, max_size)
                frame_list.append(pil_img)
            if len(frame_list) >= max_frames:
                break

        # 音频输入
        active_audio = [a for a in audio_slots if a is not None]
        if active_audio:
            print(f"[大炮-llama] 检测到 {len(active_audio)} 个音频输入")

        _params = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            seed=seed,
        )

        def _infer(msgs, pil_frames, audio_list=None):
            user_content = []
            for pil_img in pil_frames:
                b64 = image2base64(pil_img)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            if audio_list:
                for audio_dict in audio_list:
                    try:
                        b64 = audio2base64(audio_dict)
                        user_content.append({
                            "type": "input_audio",
                            "input_audio": {"data": b64, "format": "wav"},
                        })
                    except Exception as e:
                        print(f"[大炮-llama] 音频编码失败: {e}")
            user_content.append({"type": "text", "text": user_prompt})
            msgs.append({"role": "user", "content": user_content})
            mm.throw_exception_if_processing_interrupted()
            resp = llm.create_chat_completion(messages=msgs, **_params)
            text = (resp["choices"][0]["message"]["content"] or "").removeprefix(": ").lstrip()
            import re
            # 思考模式关闭时，剥离各模型的思考块，只保留最终回答
            if not think_mode:
                # Qwen/GLM 系列: <think>...</think>
                m = re.search(r"<think>.*?</think>(.*)", text, re.DOTALL)
                if m:
                    text = m.group(1).strip()
                # Gemma4 系列: <|channel>thought\n...<channel|>
                m = re.search(r"<\|channel>thought\n.*?<channel\|>(.*)", text, re.DOTALL)
                if m:
                    text = m.group(1).strip()
            msgs.append({"role": "assistant", "content": text})
            return text

        # ── 推理 ──────────────────────────────────────────────────────────────
        print(f"[大炮-llama] 推理开始  seed={seed}  模式={inference_mode}")

        if inference_mode == "one by one" and frame_list:
            replies = []
            for pil_img in cqdm(frame_list, desc="逐帧推理"):
                frame_msgs = [{"role": "system", "content": system_prompt}]
                # one by one 模式：音频只在第一帧附带，避免重复
                audio_arg = active_audio if replies == [] else None
                replies.append(_infer(frame_msgs, [pil_img], audio_arg))
            reply = "\n\n".join(replies)
            messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt}]})
            messages.append({"role": "assistant", "content": reply})
        else:
            reply = _infer(messages, frame_list, active_audio)

        # ── 更新历史 ──────────────────────────────────────────────────────────
        if save_states:
            DapaoLlamaStorage.messages[uid] = messages
        else:
            DapaoLlamaStorage.clean_state(uid)

        # ── 推理后处理 ────────────────────────────────────────────────────────
        if force_offload:
            print("[大炮-llama] 卸载模型")
            DapaoLlamaStorage.clean()
            mm.soft_empty_cache()
        elif handler_name in ("Qwen3.5", "Qwen3.5-Thinking", "Qwen3-VL", "Qwen3-VL-Thinking"):
            # 这些模型需要手动清空 KV cache，否则下次推理会出错
            try:
                llm.n_tokens = 0
                llm._ctx.memory_clear(True)
                if llm.is_hybrid and llm._hybrid_cache_mgr is not None:
                    llm._hybrid_cache_mgr.clear()
            except Exception:
                pass

        # ── 构建历史文本 ──────────────────────────────────────────────────────
        history_lines = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                text_parts = [p["text"] for p in content if p.get("type") == "text"]
                content = " ".join(text_parts)
            history_lines.append(f"[{role}]: {content}")
        history_text = "\n".join(history_lines)

        return (reply, history_text, seed)


# ── 节点注册 ─────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "Dapao_LlamaChat": Dapao_LlamaChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Dapao_LlamaChat": "😶‍🌫️llama智能对话@炮老师的小课堂",
}


"""
llama 反推额外选项节点
输出 LLAMA_CAPTION_OPTIONS 类型，供反推节点消费
"""


# 选项定义：(显示名, 内部key, 附加到提示词的指令)
_OPTIONS = [
    ("👤 包含人物信息",     "include_person",       "如果图像中有人物/角色，请包含相关信息（如姓名等）。"),
    ("🚫 排除不可改变特征", "exclude_fixed_traits", "不要包含无法改变的人物特征信息（如种族、性别等），但可以包含可改变的属性（如发型）。"),
    ("💡 包含光照信息",     "include_lighting",     "请描述图像的光照情况。"),
    ("📐 包含相机角度",     "include_camera_angle", "请描述相机角度信息。"),
    ("📷 包含相机详情",     "include_camera_detail","如果是照片，请包含使用的相机信息和详细参数（如光圈、快门速度、ISO等）。"),
    ("💡 提及光源",         "include_light_source", "如果适用，请提及可能使用的人工或自然光源。"),
    ("🎨 包含艺术质量",     "include_art_quality",  "请评价图像的美学/艺术质量（从非常低到非常高）。"),
    ("📊 包含构图信息",     "include_composition",  "请描述图像构图信息，如三分法、引导线、对称性等。"),
    ("🌈 包含景深信息",     "include_depth_of_field","请描述景深和背景是否对焦或模糊。"),
    ("🔍 排除性感内容",     "exclude_nsfw",         "不要包含任何性感或暗示性内容的描述。"),
    ("📝 不提及文字",       "exclude_text",         "不要提及图像中的任何文字内容。"),
    ("🔇 不提及分辨率",     "exclude_resolution",   "不要提及图像的分辨率。"),
    ("🏷️ 包含水印信息",    "include_watermark",    "请说明图像是否有水印。"),
    ("🖼️ 包含JPEG伪影",   "include_jpeg_artifacts","请说明图像是否有JPEG压缩伪影。"),
    ("🌍 不使用模糊语言",   "no_vague_language",    "请使用具体、准确的语言，避免模糊的表达。"),
    ("⭐ 描述重要元素",     "describe_key_elements","请重点描述图像中最重要的元素。"),
    ("🔒 包含安全性",       "include_safety",       "请评价图像是否安全、暗示性或不安全。"),
]


class Dapao_LlamaCaptionOptions:
    CATEGORY = "🍭大炮-llama-cpp"
    RETURN_TYPES = ("LLAMA_CAPTION_OPTIONS",)
    RETURN_NAMES = ("🍭llama反推选项",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {}
        for display_name, key, _ in _OPTIONS:
            inputs[display_name] = ("BOOLEAN", {"default": False})
        return {"required": inputs}

    def run(self, **kwargs):
        # 把显示名映射回内部 key
        name_to_key = {d: k for d, k, _ in _OPTIONS}
        result = {k: False for _, k, _ in _OPTIONS}
        for display_name, val in kwargs.items():
            key = name_to_key.get(display_name)
            if key:
                result[key] = val
        return (result,)

    @staticmethod
    def build_enhanced_prompt(base_prompt: str, options: dict) -> str:
        """将启用的选项附加到基础提示词后面"""
        key_to_instruction = {k: instr for _, k, instr in _OPTIONS}
        extras = [key_to_instruction[k] for k, v in options.items() if v and k in key_to_instruction]
        if not extras:
            return base_prompt
        extra_block = "\n".join(f"- {instr}" for instr in extras)
        return f"{base_prompt}\n\n请遵循以下额外要求：\n{extra_block}"


NODE_CLASS_MAPPINGS_OPTIONS = {
    "Dapao_LlamaCaptionOptions": Dapao_LlamaCaptionOptions,
}

NODE_DISPLAY_NAME_MAPPINGS_OPTIONS = {
    "Dapao_LlamaCaptionOptions": "🍭llama反推额外选项@炮老师的小课堂",
}

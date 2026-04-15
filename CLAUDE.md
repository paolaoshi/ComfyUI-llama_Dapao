# ComfyUI-llama_Dapao 项目规则

以下规则仅适用于当前项目目录：`H:/ComfyUI_Text/ComfyUI/custom_nodes/ComfyUI-llama_Dapao`

## 1. 节点开发规范
- 所有节点必须包含完整的类定义，包括 `INPUT_TYPES`、`RETURN_TYPES`、`RETURN_NAMES`、`FUNCTION`、`CATEGORY`
- 节点分类统一使用 `CATEGORY = "🍭大炮-llama-cpp"`
- 必须在文件末尾注册节点到 `NODE_CLASS_MAPPINGS` 和 `NODE_DISPLAY_NAME_MAPPINGS`

## 2. 图像处理规范
- 输入图像的张量形状：`[B, H, W, C]`
- 输入遮罩的张量形状：`[B, H, W]`
- 确保图像值范围在 `0-1` 之间（`float32`）
- 使用 `pil2tensor()` 和 `tensor2pil()` 进行格式转换

## 3. 路径与文件管理规范
- 严禁使用绝对路径
- 所有模型（Checkpoint、VAE、Lora、LLM）的加载与下载路径，必须通过 ComfyUI 的 `folder_paths` 模块动态获取
- 插件内文件（配置、图标）必须使用相对路径锚定
- 下载逻辑中，目标路径不可硬编码，下载前必须校验文件是否存在
- 节点所需模型目录为 `ComfyUI/models/LLM`，实现时必须通过动态路径获取，不能写绝对路径

## 4. 新建节点命名与交互规范
- 节点内部参数要中文化，并尽量带 emoji，方便用户
- 节点显示名必须使用“中文节点名@炮老师的小课堂”格式
- 示例：`llama对话@炮老师的小课堂`

## 5. 执行约束
- 后续与本项目有关的规则、改动、实现、文件落地，都优先在当前项目目录下进行
- 不将此类项目规则写入全局位置，除非你明确要求

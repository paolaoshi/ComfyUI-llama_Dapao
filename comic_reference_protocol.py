"""Stable comic reference slots, independent from compacted vision picture numbers."""
import json

PREFIX = '[DAPAO_COMIC_REFERENCES_V1]'
MODES = ['自动参考', '强参考', '纯文字']
ROLES = ['自动识别', '角色', '场景', '分格布局', '道具', '姿势动作', '忽略此图']

def reference_plan(kwargs):
    mode = kwargs.get('🧲参考模式', '自动参考')
    if mode not in MODES:
        raise ValueError('参考模式无效，请重新选择。')
    slots = []
    for slot in range(1, 7):
        image = kwargs.get(f'🖼️参考图{slot}')
        role = '画风' if slot == 1 else kwargs.get(f'📎参考{slot}用途', '角色' if slot == 2 else '自动识别')
        if role != '画风' and role not in ROLES:
            raise ValueError(f'参考{slot}用途无效。')
        if mode == '纯文字' or role == '忽略此图' or image is None:
            continue
        if len(image.shape) != 4 or image.shape[0] != 1:
            raise ValueError(f'参考{slot}请使用单张图片。')
        if image.shape[1] <= 1 or image.shape[2] <= 1:
            continue
        name = str(kwargs.get(f'🏷️参考{slot}称呼', '')).strip() if slot > 1 else '画风基准'
        slots.append({'slot': slot, 'role': role, 'name': name or f'参考{slot}'})
    if mode == '强参考' and not slots:
        raise ValueError('强参考模式没有有效参考图。请上传图片并启用对应参考入口（Ctrl+B），或切换自动参考/纯文字。')
    return {'mode': mode, 'slots': slots}

def encode_plan(plan):
    return PREFIX + json.dumps(plan, ensure_ascii=False, separators=(',', ':')) + '\n'

def prepare_references(prompt, images):
    """Executed only by the comic Skill. Validate duplicated graph connections before inference."""
    if not prompt.startswith(PREFIX):
        return prompt, None
    first, _, body = prompt.partition('\n')
    plan = json.loads(first[len(PREFIX):])
    selected, bindings = [], []
    for rank, item in enumerate(plan['slots'], 1):
        slot = item['slot']
        image = (images or {}).get(f'reference_image_{slot-1}')
        if image is None:
            raise ValueError(f'参考{slot}未连接到漫画理解节点，请检查该图到两个节点的连线。')
        selected.append(image)
        bindings.append(f'<Picture {rank}> = 工作流参考图{slot}；用途={item["role"]}；称呼={item["name"]}。')
    return ('参考图实际绑定（仅作为创作依据，不印在画面上）：\n' +
            ('\n'.join(bindings) if bindings else '本次无图像参考。') + '\n\n' + body), selected

"""Descriptive lettering directions; no font installation or pixel-exact font claims."""
LETTERING = {
    '按画风自动搭配': '结合实际画风、时代与语气选择正文笔形；正文、旁白、拟声分工，关键语气适度变化。',
    '清晰黑体': '对白用笔画粗细均匀、字面开阔的中等字重黑体，清楚的横竖笔画与自然行距。',
    '柔和圆体': '每个普通对白采用圆润粗笔画、圆角端点、均匀笔粗的中文圆体，字面饱满；有软糖般的圆润轮廓，而非横细竖粗的宋体。',
    '手写钢笔体': '对白用自然手写钢笔字，轻微笔压变化与有节奏的行笔，字形清楚，不以潦草代替手写感。',
    '宋体文学感': '对白与叙述采用横细竖粗、收笔有衬线的中文宋体气质，适合安静、怀旧和文学叙事。',
    '毛笔水墨体': '普通对白也采用可见提按、粗细变化、收锋与墨迹的手写毛笔字；不是只有拟声才用毛笔。短句放大展现行笔，长段收敛笔势保持可读。',
    '美漫手写体': '采用美式漫画手工描字的厚实、紧凑、略倾斜笔形，中文保持完整汉字结构；不擅自把中文翻成英文。',
    '复古印刷体': '采用厚实印刷字形、轻微油墨边缘与复古纸面气息，正文清楚；粗糙纹理不能吞掉笔画。',
    '科幻方正体': '每个普通对白的汉字使用等粗方形笔画、方角转折和切角端点，横竖形成几何块面，接近科幻仪表盘的中文机械字；科技感直接体现于汉字轮廓，不仅装饰气泡边框。保持汉字完整可读。',
    '惊悚颤抖体': '短对白汉字本身采用轻微歪斜的细长笔画、颤动的不齐收笔与不均匀基线，有手写发抖感；不只是把气泡边框画成波浪。长段可按原句语义分成连续气泡，保持可读，情绪转暖时减少颤动。',
}
EFFECTS = {
    '按语气自动变化': '按说话力度选择字重、倾斜和尺寸；拟声随动作方向布置，轻声与爆发形成对比。',
    '克制少装饰': '字效克制，重点通过字重和留白区分语气，保留用户指定的特殊效果。',
    '粗斜爆发': '关键喊句和拟声加粗、倾斜、放大，笔形有冲击感；普通对白保持基础字形。',
    '描边立体': '重点拟声采用空心描边、外轮廓或短投影形成层次，与动作透视呼应。',
    '飞白速度': '重点拟声采用毛笔飞白、拉伸笔势与速度方向，字与动作穿插，正文仍可读。',
    '波动颤抖': '犹豫、惊恐或虚弱短句采用波动基线、细颤笔形与大小变化；不把所有对白变成同一种情绪。',
}
TEXT_FLOW = {
    '按画风与气泡自动': '按气泡形状与语言安排横排或竖排，单个气泡内阅读顺序清楚。',
    '横排': '对白采用横排中文，在气泡内自然分行；气泡先后仍服从整页阅读方向。',
    '竖排': '对白采用中文竖排，每列自上而下、列从右到左；标点随竖排方向安排，气泡顺序服从整页布局。',
}
FIELDS = {'🔤对白字形': LETTERING, '✨强调字效': EFFECTS, '↕️文字排布': TEXT_FLOW}

def final_lettering_reminder(prompt):
    """Keep explicit lettering directions near generation, without changing user's text."""
    import re
    text_mode=re.search(r'^💬文字方式：([^\n]+)', prompt, re.M)
    if text_mode and text_mode.group(1) in ('无字漫画', '留白气泡后期加字'):
        return ''
    selected=[]
    for key, values in FIELDS.items():
        match=re.search(r'^'+re.escape(key)+r'：([^\n。]+)',prompt,re.M)
        if match and match.group(1) in values and match.group(1)!=next(iter(values)):
            selected.append(values[match.group(1)])
    if not selected:
        return ''
    return ('本次已选择的文字造型：'+''.join(selected)+
            '请把具体笔画特点写入普通对白的绘制描述，不只输出字体类别名称；拟声字效另行处理。'
            '保留原句和说话人，用户原始文字中的明确覆盖要求仍优先。只输出最终绘制稿。')

def lettering_inputs():
    return {key: (list(values), {'default': next(iter(values)), 'tooltip': '引导模型绘制字形，不加载电脑字体，不保证精确字体复现。'}) for key, values in FIELDS.items()}

def lettering_direction(kwargs):
    mode = kwargs.get('💬文字方式', '自动丰富对白')
    if mode == '无字漫画':
        return '本次无字漫画：字形、字效与文字排布不增加任何文字。'
    if mode == '留白气泡后期加字':
        return '本次留白气泡：只安排可供后期加字的空间，字形字效不产生占位文字。'
    descriptions=[]
    for key, values in FIELDS.items():
        choice=kwargs.get(key, next(iter(values)))
        if choice not in values:
            raise ValueError(f'{key}选项无效，请重新选择。')
        descriptions.append(f'{key}：{choice}。{values[choice]}')
    return ('文字造型指导（引导模型绘制，不指定已加载的字体文件）：\n'+'\n'.join(descriptions)+
            '\n最终脚本把字形落实到可见对白和重点拟声；字体选择不改变用户原句、说话人或文字模式。'
            '长句可按原文语义分成同一说话人的连珠气泡，顺序拼接仍为原句。'
            '正文与音效可以不同笔形；不要把整段长对白都变成巨大装饰字。')

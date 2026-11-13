"""Poster brief and director, independent of comic and video prompt protocols."""
import json
import hashlib
import re
from pathlib import Path
from .h3_local.media import image_content, text_content
from .h3_local_prompt_node import _reply, _validate_reference_images, _sampling_settings
from .multi_turn_chat_node import _sync_model, _model_settings, _release_model, _create_completion, _reset_llm

ROOT = Path(__file__).resolve().parent
SKILL = 'moss-poster-director-local'
FIELDS = json.loads((ROOT / 'data/poster_options.json').read_text('utf-8'))
SLOTS = range(1, 7)
CATEGORY = '🍭大炮-llama-cpp'

def unsupported_event_facts(copy, source):
    """Flag logistics introduced by the copywriter; keep the source's level of precision."""
    tokens = re.findall(
        r'(?<!\d)\d{1,2}[:：]\d{2}|\d+(?:\.\d+)?[元折]|\d{1,2}月\d{1,2}日'
        r'|(?:周|星期)[一二三四五六日天]|免费|免单|包邮|零门槛|限时优惠'
        r'|(?:社区|街道|街区|邻里)(?:活动|服务|文化)?(?:中心|广场|礼堂|活动室)', copy)
    for line in copy.splitlines():
        venue = re.match(r'\s*(?:活动地点|集合地点|地点|地址|场地)[：:]\s*(.+)', line)
        if venue:
            tokens.append(venue.group(1).strip().strip('。'))
    return list(dict.fromkeys(t for t in tokens if t not in source))

def printed_once_in_order(caption, copy):
    """True if the visual caption already places every nonblank copy line exactly once."""
    cursor=0
    for line in copy.splitlines():
        if not line.strip():
            continue
        if caption.count(line) != 1:
            return False
        index=caption.find(line, cursor)
        if index < 0:
            return False
        cursor=index+len(line)
    return bool(copy.strip())

class DapaoPosterOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'📝用户需求': ('STRING', {'forceInput': True}),
                **{k: (list(v),) for k,v in FIELDS.items()},
                "🔒锁定原文": ("STRING", {"multiline":True,"default":""})},
                'optional': {f'🖼️参考图{i}': ('IMAGE',) for i in SLOTS}}
    RETURN_TYPES = ('STRING','BOOLEAN')
    RETURN_NAMES = ('海报导演需求','手动直通')
    FUNCTION = 'compose'
    CATEGORY = CATEGORY
    def compose(self, **kw):
        request = kw['📝用户需求']
        if kw['🛠️创作模式'] == '手动直通':
            if not request.strip():
                raise ValueError('手动直通需要在海报需求框填写要直接送入 MOSS 的提示词。')
            return request, True
        copy_mode=kw['💬文案方式']
        locked=kw.get('🔒锁定原文','') if copy_mode.startswith(('全部锁字','部分锁字')) else ''
        if copy_mode.startswith(('全部锁字','部分锁字')) and not locked.strip():
            raise ValueError('锁字模式需要在「文字与排版 → 锁定原文」中填写必须逐字保留的文字。')
        if not request.strip():
            if not locked.strip():
                raise ValueError('请填写海报主题或需求。使用锁字模式时，也可以只填写锁定原文。')
            request='根据锁定原稿的含义与所选用途设计成品海报，自主构思主视觉、字形、布局与色彩。'
        return json.dumps({'锁定原文':locked,'原始需求':request, '设计选择':{k:[kw[k],FIELDS[k][kw[k]][0]] for k in FIELDS if not k.startswith('📎')},
            '参考绑定':{str(i):kw[f'📎参考{i}用途'] for i in SLOTS if kw.get(f'🖼️参考图{i}') is not None and kw['🧲参考模式']!='纯文字'}},ensure_ascii=False),False

class DapaoPosterDirector:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'model':('DAPAO_LOCAL_MODEL',),'brief':('STRING',{'forceInput':True}),
                    'skill':([SKILL],),'seed':('INT',{'default':123456789,'min':0,'max':4294967295}),
                    'max_tokens':('INT',{'default':4096,'min':256,'max':16384})},
                'optional':{f'ref{i}':('IMAGE',) for i in SLOTS}}
    RETURN_TYPES = ('STRING','STRING')
    RETURN_NAMES = ('海报提示词','参考用途')
    CATEGORY = CATEGORY
    FUNCTION = 'direct'
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Edits to the active Skill must invalidate a previously cached prompt.
        return hashlib.sha256((ROOT/'skills'/SKILL/'SKILL.md').read_bytes()).hexdigest()

    def direct(self, model, brief, skill, seed, max_tokens, **refs):
        spec=json.loads(brief)
        roles=spec['参考绑定']
        locked=spec.pop('锁定原文','')
        # Use the wording for visual meaning; insert the authoritative original after generation.
        if locked:
            spec['排版预算']=[f'第{i}行约{len(line)}字符' for i,line in enumerate(locked.splitlines(),1)]
            if len(locked.strip()) <= 32:
                spec['短文案构图']='除非用户明确要预留书写栏或所选版式为续写模板，将未用于文字的版面画成主题插画；主图与局部细节形成完整画面。观察或过程类可选连续阶段、剖面、局部或不同尺度来叙事。不要画空栏目或待填写的正文框。'
        pictures = []
        for slot, role in roles.items():
            pic = refs.get('ref' + slot)
            if pic is None:
                raise ValueError(f'参考{slot}没有连接到导演。')
            _validate_reference_images([pic])
            scope = FIELDS[f'📎参考{slot}用途'][role][0]
            pictures.append((slot, role, scope, pic))
        settings=_model_settings(model)
        if roles and settings['mmproj_file'] in ('None','无',''): raise ValueError('参考图需要匹配的视觉投影 mmproj。')
        rules=(ROOT/'skills'/SKILL/'SKILL.md').read_text('utf-8')
        active=model
        try:
            active=_sync_model(model)
            observations = []
            ref_mode, mode_scope = spec['设计选择']['🧲参考模式']
            for slot, role, scope, pic in pictures:
                observe = (f'仅为平面设计师提取这张图中允许继承的可见特征。全局：{ref_mode}，{mode_scope}。'
                           f'本图用途：{role}，{scope}。只描述两者交集。'
                           '若只取布局，使用左侧文字区、右侧主图区等抽象位置与比例，省略示例主体和字句；'
                           '若只取风格，就说出具体用色、线条、印刷和材质，省略示例物体、信息栏和构图。'
                           '产品只记录可见形状、比例、部件和材质，忽略背景。不得抄写任何示例文字、日期与价格。'
                           '直接用一两句话列出观察所得，不写新海报，不猜测。')
                note, _ = _create_completion(active.llm, [
                    {'role':'system','content':observe},
                    {'role':'user','content':[text_content('观察这一张参考图。'),image_content(pic)]}],
                    {'seed':int(seed), 'temperature':0.15, 'max_tokens':400, 'stream':False})
                facts = _reply(note)
                if not facts: raise ValueError(f'参考{slot}未返回视觉特征。')
                observations.append(f'图{slot} · {role}：{facts}')
            if observations:
                spec['参考观察'] = observations
            copy_mode = spec['设计选择']['💬文案方式'][0]
            strict = copy_mode.startswith('全部锁字')
            blank = copy_mode in ('无字海报','留白后期加字')
            copy = locked
            if not strict and not blank:
                editor = (
                    '你负责决定这张海报上真正要印的文字，输出按阅读顺序排列的原文，每个独立文字区一行。'
                    '不要字段名、Markdown、引号、设计说明或编号，除非编号本身需要印刷。'
                    '主题如果很短，按用途创作有个性的实际内容。活动传单可写亮点和邀请；'
                    '科普栏写具体且准确的知识，适合年级，标题与正文可分别一行；服务告示言简意赅。'
                    '限制每栏正文长度，字号适合手持阅读。用户明确的原稿和字量优先。双语逐句交替。'
                    '若要求一个标题和一句邀请，仅两行；标题加两句歌词仅三行，不另加落款和行动语。'
                    '忠实整理和润色保留原有事实；不虚构具体时间地点、报名渠道、活动价格、免费服务、限时优惠或保证。'
                    '部分锁字时只输出获准补写的文字，不重复已锁原稿。如果不需补充，只输出 NONE。')
                if copy_mode=='AI 儿童科普短文':
                    editor+=(' 面向儿童的小栏目：除非用户要求长文，每栏用一个短标题配一句简短解释，'
                             '让标题和解释紧邻。避免用容易混淆的近义栏目名。'
                             '用日常语解释真实因果，不用含混的比喻取代科学原理。')
                copy_request={'需求':spec['原始需求'],'用途':spec['设计选择']['📌海报用途'],
                              '工作方式':spec['设计选择']['🛠️创作模式'],'文案方式':spec['设计选择']['💬文案方式'],
                              '字形与风格':spec['设计选择']['🎨视觉风格'],'已有锁稿':locked}
                layout_notes=[note for note, (_, role, _, _) in zip(observations, pictures)
                              if ref_mode=='只参考空间编排' or role in ('布局与层级','栏目密度与行距')]
                if layout_notes:
                    copy_request['参考区域结构']=layout_notes
                    editor+=' 空间参考只提供区块数量和容量。当用户要求每格对应一个事项时，给每个格子准备一项主题内容；不得复用参考原文。'
                msgs=[{'role':'system','content':editor},{'role':'user','content':json.dumps(copy_request,ensure_ascii=False)}]
                raw,_=_create_completion(active.llm,msgs,{'seed':int(seed),'temperature':0.55,'max_tokens':int(max_tokens),'stream':False})
                added=_reply(raw).strip()
                if not added: raise ValueError('海报文案生成未返回正文。')
                if added.upper()=='NONE': added=''
                facts=spec['原始需求']+' '+locked
                invented=unsupported_event_facts(added, facts)
                if invented or copy_mode=='AI 儿童科普短文':
                    review = ('只校对拟印文案，不设计画面。删去用户未给出的具体日期、地点、价格和活动承诺；'
                              '保留用户提供的时间精度。科普必须解释正确的因果，类比中各物体的运动关系应准确，'
                              '年龄较小时用短句，但不能把不同概念当成一回事。只返回全部准印文字。')
                    if invented:
                        review+=' 以下未经授权的信息应去除：'+'、'.join(invented)
                    msgs.extend([{'role':'assistant','content':added},{'role':'user','content':review}])
                    fixed,_=_create_completion(active.llm,msgs,{'seed':int(seed),'temperature':0.1,'max_tokens':int(max_tokens),'stream':False})
                    added=_reply(fixed).strip()
                    if not added:
                        raise ValueError('文案校对未返回正文。')
                    # An optional logistics line can simply be dropped if a revision still invents it.
                    added='\n'.join(line for line in added.splitlines() if not unsupported_event_facts(line,facts))
                    if not added.strip() and not locked:
                        raise ValueError('文案校对未能保留有依据的印字。请写明需要印出的活动内容。')
                if locked:
                    protected={line.strip().strip('“”"') for line in locked.splitlines() if line.strip()}
                    added='\n'.join(line for line in added.splitlines()
                                    if line.strip().strip('“”"') not in protected)
                copy='\n'.join(x for x in (locked,added) if x)
            # The last model call is always art direction. Never replace it with bare advertising copy.
            design_spec=dict(spec)
            design_spec['已经定稿的全幅印字']=copy
            design_spec['本次成品约束']=(
                '本次为无字作品，只有用户明确要后填文字时才留书写区。' if blank else
                '这份印字清单已经定稿。只为这些行安排位置、字体、大小和顺序，不加任何其他字块。'
                '全文一行就只做一处文字，绝对不虚构小号标注、统计栏、落款、英文或二维码。')
            design_rule=(
                '你最后交付的是供 MOSS 出图的成品海报画面描述。先用自然语言具体描写主视觉的形状、材质、颜色、场景与布局；'
                '再描述已定稿文字的区块如何排放。请不要重新誊写印字原文，程序会在末尾加入。'
                '不要输出纯文案、标题列表、Markdown 或任何占位标记。按照已定稿文案和本次成品约束处理文字。'
                '参考事实限于所选用途，纯布局参考不得引入参考稿上的时间、地点、活动要素。')
            final_msgs=[{'role':'system','content':rules+'\n'+design_rule},
                        {'role':'user','content':json.dumps(design_spec,ensure_ascii=False)}]
            params={**_sampling_settings(settings['think']),'seed':int(seed),'max_tokens':int(max_tokens),'stream':False}
            draft,_=_create_completion(active.llm,final_msgs,params)
            result=_reply(draft)
            if not result: raise ValueError('海报设计未返回绘制稿。')
            if copy and not blank:
                # Reference geometry may include a footer. Its source event is not our event.
                grounded=spec['原始需求']+'\n'+copy
                for quoted in re.findall(r'“([^”\n]+)”',result):
                    offending=unsupported_event_facts(quoted,grounded)
                    if offending:
                        clean=quoted
                        for token in sorted(offending,key=len,reverse=True):
                            clean=clean.replace(token,'')
                        result=result.replace('“'+quoted+'”','“'+clean.strip(' ·，、：:')+'”' if clean.strip(' ·，、：:') else '')
            if copy and (locked or not printed_once_in_order(result, copy)):
                result+='\n\n海报实际印字按以下阅读顺序排列，其他区域仅用图形表现：\n“'+copy+'”'
            return result,'\n'.join(observations) or '本次仅用文字'
        finally: _release_model(active)

NODE_CLASS_MAPPINGS={'DapaoPosterOptions':DapaoPosterOptions,'DapaoPosterDirector':DapaoPosterDirector}
NODE_DISPLAY_NAME_MAPPINGS={'DapaoPosterOptions':'🎨 海报创作大师@炮老师小课堂','DapaoPosterDirector':'🖋️ 海报提示词导演@炮老师小课堂'}

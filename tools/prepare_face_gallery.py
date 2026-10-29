"""Prepare a portable face gallery from a user-provided offline library.
Usage: python tools/prepare_face_gallery.py SOURCE DESTINATION
Never edits source files. Existing destination images are checked, not overwritten.
"""
import argparse, hashlib, json, re, shutil, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path);args=parser.parse_args()
    source=args.source;dest=args.destination
    data=json.loads((source/'data/catalog.json').read_text(encoding='utf-8-sig'))
    for kind in ('image','thumb'):(dest/kind).mkdir(parents=True,exist_ok=True)
    def convert(item):
        item=dict(item);identifier=item['id']
        if not re.fullmatch(r'[a-f0-9]{11}',identifier):raise ValueError('Invalid face ID')
        original=source/item['files']['original'];target=dest/'image'/f'{identifier}.webp';thumb=source/item['files']['thumb']
        if not all(path.resolve().is_relative_to(source.resolve()) for path in (original,thumb)):raise ValueError('Source path escapes library')
        with Image.open(original) as image:
            image.load()
            if not target.exists():image.save(target,format='WEBP',lossless=True,method=6,exact=True)
            with Image.open(target) as check:
                if not np.array_equal(np.array(image),np.array(check)):raise ValueError('Pixel mismatch: '+identifier)
            item['width'],item['height']=image.size
        thumb_target=dest/'thumb'/f'{identifier}.webp'
        if not thumb_target.exists():shutil.copy2(thumb,thumb_target)
        if hashlib.sha256(thumb.read_bytes()).digest()!=hashlib.sha256(thumb_target.read_bytes()).digest():raise ValueError('Thumbnail mismatch: '+identifier)
        item['original_bytes']=original.stat().st_size;item['bytes']=target.stat().st_size;item['sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
        item['files']={'image':f'image/{identifier}.webp','thumb':f'thumb/{identifier}.webp'}
        item['category_id']=item['hair_id'];item['category_zh']=item['hair_zh'];item['category_en']=item['hair_en'];item.pop('source_urls',None)
        return item
    started=time.monotonic();results={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(convert,item) for item in data['items']]):
            item=future.result();results[item['id']]=item
            if len(results)%100==0:print(f'{len(results)}/{len(data["items"])} verified; {time.monotonic()-started:.0f}s',flush=True)
    data['items']=[results[i['id']] for i in data['items']];data['categories']=data['hairs'];data['version']=1;data['image_encoding']='webp-lossless'
    (dest/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    shutil.copy2(source/'README.md',dest/'SOURCE-README.md')
    license_text='来源：https://ai-portrait-gallery.pages.dev/\n作者：あぎ（X: @agi_aibusi）\n原站标注 CC0 1.0：https://creativecommons.org/publicdomain/zero/1.0/\n原站同时请求尽量不要直接再分发图片（请求），此处保留该说明，不将其表述成服装库的禁止再分发条款。\n图片为 AI 生成的虚构人物。中文标签与提示词为离线库整理，不是原作者的原始提示词。\n已于 2026-10-29 核对来源页面。\n'
    (dest/'SOURCE-LICENSE.txt').write_text(license_text,encoding='utf-8')
    original=sum(i['original_bytes'] for i in data['items']);encoded=sum(i['bytes'] for i in data['items'])
    summary={'count':len(results),'original_bytes':original,'lossless_bytes':encoded,'pixel_equal_count':len(results),'total_bytes':sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),'seconds':round(time.monotonic()-started,1)}
    (dest/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()

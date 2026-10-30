async (page) => {
    await page.setViewportSize({width:1500,height:1100});
    await page.reload();
    const random=page.getByRole('button',{name:'🎲 随机组合'});
    await random.click();
    await page.waitForFunction(()=>graph.nodes.length===4&&graph.nodes.slice(1).every(n=>n.imgs?.[0]?.naturalWidth));
    const first=await page.evaluate(()=>node.widgets.filter(w=>w.name.endsWith('ID')).map(w=>w.value));
    await random.click();
    await page.waitForFunction(first=>node.widgets.filter(w=>w.name.endsWith('ID')).every((w,i)=>w.value!==first[i]),first);
    await page.waitForFunction(()=>!document.querySelector('.dmi-toolbar button').disabled);
    if(await page.evaluate(()=>graph.nodes.length)!==4)throw Error('Duplicate previews');
    await page.locator('.dmi-card').nth(2).getByRole('button',{name:'打开素材库'}).click();
    await page.getByRole('button',{name:/全部 ·/}).waitFor();
    if(await page.getByRole('button',{name:/双人/}).count())throw Error('Pair pose exposed');
    await page.locator('.dp-card').first().click();
    const apply=page.locator('dialog[open] footer button');
    await apply.click();
    await page.waitForFunction(()=>!document.querySelector('dialog[open]')&&!document.querySelector('.dmi-toolbar button').disabled);
    // Simulate a real upload with a tiny PNG; the harness stores it in a temporary directory.
    await page.getByRole('combobox',{name:'脸模来源'}).selectOption('上传图片');
    await page.waitForFunction(()=>!document.querySelector('.dmi-toolbar button').disabled);
    await page.locator('.dmi-card').first().locator('input[type=file]').setInputFiles({name:'custom-face.png',mimeType:'image/png',buffer:Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=','base64')});
    await page.waitForFunction(()=>node.widgets.find(w=>w.name==='👩脸模上传').value==='custom-face.png'&&!document.querySelector('.dmi-toolbar button').disabled);
    await random.click();
    await page.waitForFunction(()=>!document.querySelector('.dmi-toolbar button').disabled);
    if(await page.evaluate(()=>node.widgets.find(w=>w.name==='👩脸模来源').value)!=='上传图片')throw Error('Upload overridden');
    // Only text connected: updating must not recreate image previews.
    await page.evaluate(()=>{
        for(const n of [...graph.nodes])if(n!==node)graph.remove(n);
        const target={id:100,type:'PreviewAny',onExecuted(p){this.payload=p}};graph.add(target);node.connect(3,target,0);
    });
    await random.click();
    await page.waitForFunction(()=>graph.getNodeById(100).payload?.text[0]===document.querySelector('.dmi-prompt').value&&!document.querySelector('.dmi-toolbar button').disabled);
    if(await page.evaluate(()=>graph.nodes.length)!==2)throw Error('Text-only created preview');
    // Restore native serialized values, without creating previews on configure.
    await page.evaluate(()=>{node.widgets.find(w=>w.name==='🎲提示词种子').value=8;node.onConfigure();});
    await page.waitForFunction(()=>document.querySelector('.dmi-prompt').value.includes('自然漫射光')&&!document.querySelector('.dmi-toolbar button').disabled);
    await page.screenshot({path:'output/playwright/model-incubator-simulated.png',fullPage:true});
    return {random:4,autoPreviews:3,reused:true,singlePoseDialog:true,manualSelection:true,uploadPreserved:true,textOnly:true,configure:true};
}

async (page) => {
    await page.setViewportSize({width:1450,height:1100});
    await page.reload();
    for (const kind of ['Face','Costume','Pose']) {
        const host=page.locator(`section[data-kind="${kind}"]`);
        await host.locator('.dpn-card').first().waitFor();
        const canvas=host.locator('canvas');
        await canvas.click({position:{x:36,y:28}});
        await host.locator('.downstream img').waitFor();
        await host.locator('.downstream img').evaluate(i=>i.decode());
        const first=await host.locator('.downstream img').getAttribute('src');
        await canvas.click({position:{x:36,y:28}});
        await page.waitForFunction(({kind,first})=>document.querySelector(`section[data-kind="${kind}"] .downstream img`)?.getAttribute('src')!==first,{kind,first});
        await host.locator('.downstream img').evaluate(i=>i.decode());
        const ok=await host.evaluate(h=>!h.querySelector('.dpn-output-preview')&&!!h.querySelector('h3 button')&&h.querySelector('.downstream img').naturalWidth>0);
        if(!ok)throw Error(kind+' preview/layout invalid');
    }
    // Verify link dispatch without opening the external service.
    const result=await page.evaluate(()=>{let opened;const original=window.open;window.open=(...args)=>{opened=args};nodes[0].onTitleButtonClick({name:'dapao_infinite_canvas'});window.open=original;return opened;});
    if(result?.[0]!=='https://paopao.dapaoai.com/'||result?.[2]!=='noopener,noreferrer')throw Error('link invalid');
    const lifecycle=await page.evaluate(()=>{
        const target=nodes[0].target, expected=target.imgs[0];
        target.imgs=[];target.onDrawBackground();
        if(target.imgs[0]!==expected)return false;
        target.onExecuted({images:[]});target.imgs=[];target.onDrawBackground();
        return target.imgs.length===0;
    });
    if(!lifecycle)throw Error('normal execution must take over the preview');
    for(const kind of ['Face','Costume','Pose']) {
        await page.evaluate(kind=>{const n=nodes.find(n=>n.id===kind);n.outputs[0].links=[];},kind);
        const canvas=page.locator(`section[data-kind="${kind}"] canvas`);
        await canvas.click({position:{x:36,y:28}});
        await page.waitForFunction(kind=>nodes.find(n=>n.id===kind).graph.created.length===1,kind);
        await page.waitForFunction(kind=>nodes.find(n=>n.id===kind).graph.created[0].imgs?.[0]?.naturalWidth>0,kind);
        const previous=await page.evaluate(kind=>nodes.find(n=>n.id===kind).widgets[0].value,kind);
        await canvas.click({position:{x:36,y:28}});
        await page.waitForFunction(({kind,previous})=>nodes.find(n=>n.id===kind).widgets[0].value!==previous,{kind,previous});
        if(await page.evaluate(kind=>nodes.find(n=>n.id===kind).graph.created.length,kind)!==1)throw Error('duplicate automatic preview');
    }
    // Text-only downstream: use the selected language and do not add an image node.
    await page.evaluate(()=>{
        const n=nodes[0];n.outputs=Array.from({length:5},()=>({links:[]}));n.widgets[1].value='English';
        for(const [slot,type] of [[1,'PreviewAny'],[2,'ShowText|pysssss'],[3,'easy showAnything'],[4,'PreviewAny']]) {
            const t={id:'text-'+slot,type,onExecuted(payload){this.payload=payload;},setDirtyCanvas(){}};
            n.graph.nodes.push(t);n.connect(slot,t,0);
        }
    });
    await page.locator('section[data-kind="Face"] canvas').click({position:{x:36,y:28}});
    await page.waitForFunction(()=>nodes[0].graph.getNodeById('text-4').payload?.text[0]===nodes[0].widgets[0].value);
    const textOK=await page.evaluate(async()=>{
        const n=nodes[0],data=await (await fetch('/dapao/face-gallery/catalog')).json();
        const item=data.items.find(i=>i.id===n.widgets[0].value);
        return n.graph.created.length===1&&n.graph.getNodeById('text-2').payload.text[0]===item.prompt_en&&n.graph.getNodeById('text-3').payload.text[0]===item.negative_prompt_en&&n.graph.getNodeById('text-1').payload.text[0]===`${item.id} · ${item.section_zh} · ${item.hair_zh}；${item.tags_zh.join('、')}`;
    });
    if(!textOK)throw Error('text outputs mismatch or unwanted image creation');
    await page.screenshot({path:'output/playwright/reference-controls-connected.png',fullPage:true});
    return {nodes:3,randomSelections:13,autoCreated:3,textOutputs:4,imageOnly:true,link:result[0]};
}

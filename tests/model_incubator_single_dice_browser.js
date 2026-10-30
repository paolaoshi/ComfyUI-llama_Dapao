async (page) => {
    await page.setViewportSize({width:1600,height:1200});
    await page.waitForFunction(()=>!!globalThis.LiteGraph?.registered_node_types?.DapaoModelIncubator);
    let submissions=0;
    const observe=r=>{if(r.method()==='POST'&&new URL(r.url()).pathname==='/prompt')submissions++;};
    page.on('request',observe);
    try {
        await page.evaluate(async()=>{
            const {app}=await import('/scripts/app.js');
            if(app.graph._nodes.length)throw Error('Use an empty isolated test browser');
            const n=LiteGraph.createNode('DapaoModelIncubator');app.graph.add(n);n.pos=[180,220];
            app.canvas.ds.scale=.75;app.canvas.ds.offset=[0,0];window.singleDiceNode=n;
        });
        await page.getByRole('button',{name:'🎲 随机组合',exact:true}).click();
        await page.waitForFunction(()=>!document.querySelector('.dmi-toolbar button').disabled&&window.app.graph._nodes.length===4);
        for(const [slot,label] of ['脸模','服装','姿势'].entries()) {
            const before=await page.evaluate(()=>{
                const n=window.singleDiceNode;
                window.diceOldImages=window.app.graph._nodes.filter(n=>n.type==='PreviewImage').map(n=>n.imgs[0]);
                return {ids:n.widgets.filter(w=>w.name.endsWith('ID')).map(w=>w.value),seed:n.widgets.find(w=>w.name==='🎲提示词种子').value};
            });
            await page.getByRole('button',{name:`随机${label}`,exact:true}).click();
            const active=await page.locator(`.dmi-card:nth-child(${slot+1}) .dmi-single-roll`).evaluate(b=>b.classList.contains('dmi-rolling'));
            if(!active)throw Error('Missing per-card animation');
            await page.waitForFunction(()=>!document.querySelector('.dmi-toolbar button').disabled);
            const ok=await page.evaluate(({before,slot})=>{
                const n=window.singleDiceNode,ids=n.widgets.filter(w=>w.name.endsWith('ID')).map(w=>w.value);
                const previews=window.app.graph._nodes.filter(n=>n.type==='PreviewImage');
                return previews.length===3&&ids.every((id,i)=>i===slot?id!==before.ids[i]:id===before.ids[i])&&
                    n.widgets.find(w=>w.name==='🎲提示词种子').value===before.seed&&
                    previews.every((n,i)=>i===slot?n.imgs[0].naturalWidth>0&&n.imgs[0].src.includes(ids[i]):n.imgs[0]===window.diceOldImages[i]);
            },{before,slot});
            if(!ok)throw Error(`${label}: unrelated selection changed or preview did not follow`);
        }
        await page.getByRole('combobox',{name:'脸模来源'}).selectOption('上传图片');
        await page.waitForFunction(()=>!document.querySelector('.dmi-toolbar button').disabled);
        if(!await page.getByRole('button',{name:'随机脸模',exact:true}).isDisabled())throw Error('Upload dice must be disabled');
        await page.getByRole('combobox',{name:'脸模来源'}).selectOption('素材库');
        await page.waitForFunction(()=>!document.querySelector('.dmi-toolbar button').disabled);
        if(submissions)throw Error('Unexpected workflow submission');
        await page.screenshot({path:'output/playwright/model-incubator-single-dice-real.png'});
        return {singleDice:3,otherSelectionsUnchanged:true,matchingPreviewUpdated:true,otherPreviewImagesUntouched:true,seedUnchanged:true,uploadProtected:true,workflowSubmissions:submissions};
    }finally{page.off('request',observe);}
}

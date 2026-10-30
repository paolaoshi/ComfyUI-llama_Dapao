async (page) => {
    // Run only in our isolated browser tab with the incubator test node already created.
    let submissions=0;
    const observe=request=>{if(request.method()==='POST'&&new URL(request.url()).pathname==='/prompt')submissions++;};
    page.on('request',observe);
    try {
        const random=page.getByRole('button',{name:'🎲 随机组合'});
        await random.click();
        await page.waitForFunction(()=>!document.querySelector('.dmi-toolbar button').disabled&&window.app.graph._nodes.filter(n=>n.type==='PreviewImage').length===3);
        const first=await page.evaluate(()=>window.app.graph._nodes.filter(n=>n.type==='PreviewImage').map(n=>({id:n.id,src:n.imgs?.[0]?.src})));
        await random.click();
        await page.waitForFunction(first=>first.every(old=>{const n=window.app.graph.getNodeById(old.id);return n.imgs?.[0]?.naturalWidth&&n.imgs[0].src!==old.src})&&!document.querySelector('.dmi-toolbar button').disabled,first);
        const result=await page.evaluate(async()=>{
            const {app}=await import('/scripts/app.js'), n=window.incubatorTestNode;
            const workflow=app.graph.serialize();window.incubatorTestWorkflow=workflow;
            const payload=await app.graphToPrompt();
            const inputs=payload.output[n.id].inputs;
            if(!inputs['👩脸模ID']||!inputs['👗服装ID']||!inputs['🧍姿势ID'])throw Error('Hidden widgets not serialized to prompt');
            return {outputs:n.outputs.map(o=>o.name),previewCount:app.graph._nodes.filter(n=>n.type==='PreviewImage').length,serialized:true};
        });
        await page.screenshot({path:'output/playwright/model-incubator-real.png'});
        // Check a real text preview plugin, then reload the in-memory serialized workflow.
        const textType=await page.evaluate(()=>{
            const app=window.app, n=window.incubatorTestNode;
            for(const t of [...app.graph._nodes])if(t.type==='PreviewImage')app.graph.remove(t);
            const target=LiteGraph.createNode('PreviewAny');if(!target)throw Error('PreviewAny unavailable');
            app.graph.add(target);n.connect(3,target,0);window.incubatorTextTarget=target;return target.type;
        });
        await random.click();
        await page.waitForFunction(()=>!document.querySelector('.dmi-toolbar button').disabled);
        const textOK=await page.evaluate(()=>{
            if(window.app.graph._nodes.some(n=>n.type==='PreviewImage'))throw Error('Text-only created image');
            return window.incubatorTextTarget.widgets?.some(w=>typeof w.value==='string'&&w.value.includes('图片1'));
        });
        if(!textOK)throw Error('Real text preview not updated');
        await page.evaluate(()=>{window.app.graph.configure(window.incubatorTestWorkflow);window.incubatorTestNode=window.app.graph._nodes.find(n=>n.type==='DapaoModelIncubator');});
        await page.waitForFunction(()=>!!document.querySelector('.dmi-node')&&!document.querySelector('.dmi-toolbar button').disabled);
        await page.getByRole('button',{name:'更新下游预览'}).click();
        await page.waitForFunction(()=>window.app.graph._nodes.filter(n=>n.type==='PreviewImage').every(n=>n.imgs?.[0]?.naturalWidth));
        if(submissions)throw Error('Unexpected workflow submission');
        return {...result,repeatedRandom:true,textType,textOK,serializedReload:true,workflowSubmissions:submissions};
    }finally{page.off('request',observe);}
}

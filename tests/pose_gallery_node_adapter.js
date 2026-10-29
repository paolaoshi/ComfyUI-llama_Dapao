async (page) => {
    await page.reload();
    await page.evaluate(async () => {
        class Node {
            widgets = [{name:'🧍姿势ID',value:''}];
            size = [340,390];
            changes = 0;
            graph = {beforeChange:()=>this.changes++,afterChange(){}};
            addDOMWidget(name,type,root,options) {
                const host=document.createElement('section');host.id='node-host';
                host.style.cssText='width:430px;border:2px solid #559bdf;border-radius:14px;padding:12px;background:#191b20;margin:20px';
                const title=document.createElement('h3');title.textContent='精选姿势参考@泡老师的小课堂';title.style.fontSize='16px';
                host.append(title,root);root.style.height='480px';document.body.append(host);
                const result={options,root};this.dom=result;return result;
            }
            addWidget() {throw Error('Separate canvas button would reintroduce overlap');}
            setSize(size) {this.size=size; const host=document.querySelector('#node-host');if(host)host.style.width=size[0]+'px';}
            computeSize() { return [360,558]; }
            setDirtyCanvas() {}
        }
        await window.poseExtension.beforeRegisterNodeDef(Node,{name:'DapaoPoseReference'});
        window.testNode = new Node(); window.testNode.onNodeCreated();
    });
    await page.locator('.dpn-card').first().waitFor();
}

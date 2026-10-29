async (page) => {
 const root=page.locator('.dapao-pose-node');
 if(await root.locator('.dpn-card').count()!==6)throw Error('six cards');
 await root.getByRole('button',{name:'站立・双手扶腰 · 正面',exact:true}).click();
 await page.waitForFunction(()=>window.testNode.widgets[0].value==='full_single_P018_front');
 await root.getByRole('button',{name:'侧面',exact:true}).click();
 await page.waitForFunction(()=>window.testNode.widgets[0].value==='full_single_P018_side');
 if(await root.locator('.dpn-card[aria-pressed=true]').count()!==1)throw Error('selected highlight');
 const checkLayout=async()=>{
  const boxes=await root.evaluate(e=>{
   const r=s=>{const b=e.querySelector(s).getBoundingClientRect();return {top:b.top,bottom:b.bottom};};
   return {grid:r('.dpn-grid'),info:r('.dpn-selected'),angles:r('.dpn-angles'),button:r('.dpn-open'),footer:r('.dpn-footer'),bottom:e.getBoundingClientRect().bottom,overflow:e.scrollWidth>e.clientWidth};
  });
  if(boxes.overflow||boxes.grid.bottom>boxes.info.top||boxes.angles.bottom+8>boxes.button.top||boxes.button.bottom>boxes.footer.top||boxes.footer.bottom>boxes.bottom)throw Error('controls overlap or clip '+JSON.stringify(boxes));
 };
 await checkLayout();
 if(await root.locator('.dpn-grid').evaluate(e=>e.scrollHeight>e.clientHeight+1))throw Error('default grid clips second row');
 await root.screenshot({path:'output/playwright/pose-quick-default.png'});
 for(const width of [360,430,600]){
  await page.evaluate(w=>{const n=window.testNode;const s=[w,300];n.onResize(s);n.setSize(s);},width);
  await checkLayout();
 }
 await page.evaluate(()=>{window.testNode.dom.root.style.height='380px';});
 await checkLayout();
 await page.evaluate(()=>{window.testNode.dom.root.style.height='480px';window.testNode.setSize([430,650]);});
 await root.getByRole('button',{name:'打开完整姿势库',exact:false}).click();
 await page.getByRole('button',{name:'下一姿势',exact:true}).click();
 await page.getByRole('button',{name:/使用此视角/}).click();
 await page.waitForFunction(()=>!document.querySelector('dialog'));
 if(!(await root.locator('.dpn-selected').textContent()).includes('full_single_P019'))throw Error('gallery commit');
 await page.evaluate(()=>{window.testNode.widgets[0].value='full_single_P002_high';window.testNode.size=[340,390];window.testNode.onConfigure();});
 await root.locator('.dpn-selected').filter({hasText:'full_single_P002_high'}).waitFor();
 if(await root.locator('.dpn-card').count()!==6)throw Error('external selection grid');
 if(!(await root.locator('.dpn-card[aria-pressed=true]').getAttribute('aria-label')).includes('俯视'))throw Error('restored angle');
 if(await page.evaluate(()=>window.testNode.size[1]!==558))throw Error('old workflow size migration');
 await page.route('**/dapao/pose-gallery/image/full_single_P018_front',r=>r.abort());
 await root.getByRole('button',{name:'站立・双手扶腰 · 正面',exact:true}).click();
 await root.getByRole('status').filter({hasText:'原选择已保留'}).waitFor();
 if(await page.evaluate(()=>window.testNode.widgets[0].value)!=='full_single_P002_high')throw Error('failure changes output');
 await checkLayout();
 await page.unroute('**/dapao/pose-gallery/image/full_single_P018_front');
 await root.getByRole('button',{name:'站立・双手扶腰 · 正面',exact:true}).click();
 await page.waitForFunction(()=>window.testNode.widgets[0].value==='full_single_P018_front');
 await page.evaluate(()=>{window.testNode.widgets[0].value='full_single_P001_high';window.testNode.onConfigure();});
 await root.locator('.dpn-selected').filter({hasText:'full_single_P001_high'}).waitFor();
 await checkLayout();
 await root.screenshot({path:'output/playwright/pose-quick-selected.png'});
 await root.getByRole('button',{name:'打开完整姿势库',exact:false}).click();
 await page.evaluate(()=>window.testNode.onRemoved());
 await page.waitForFunction(()=>!document.querySelector('dialog')&&!document.querySelector('.dapao-pose-node'));
 console.log('PASS six existing assets; quick select; angles; 360/430/600 widths; short panel; gallery commit; old configure; errors; removal');
}
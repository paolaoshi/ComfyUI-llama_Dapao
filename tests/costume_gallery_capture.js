async (page) => {
 await page.reload(); const root=page.locator('.dapao-costume-node');await root.locator('.dpn-card').first().waitFor();
 await root.getByRole('button',{name:'SPR-0001 春季',exact:true}).click();await page.waitForFunction(()=>window.testNode.widgets[0].value==='SPR-0001');
 await root.screenshot({path:'output/playwright/costume-node.png'});
 await root.getByRole('button',{name:/打开完整服装库/}).click();
 await page.locator('.dp-detail-image').evaluate(e=>e.decode());
 await page.locator('dialog.dapao-costume').screenshot({path:'output/playwright/costume-gallery.png'});
}

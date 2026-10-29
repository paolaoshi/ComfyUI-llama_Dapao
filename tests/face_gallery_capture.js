async (page) => {
 await page.reload();const root=page.locator('.dapao-face-node');await root.locator('.dpn-card').first().waitFor();
 await root.getByRole('button',{name:'f71f980dde3 短发',exact:true}).click();await page.waitForFunction(()=>window.testNode.widgets[0].value==='f71f980dde3');
 await root.screenshot({path:'output/playwright/face-node.png'});
 await root.getByRole('button',{name:/打开完整脸模库/}).click();await page.locator('.dp-detail-image').evaluate(e=>e.decode());
 await page.locator('dialog.dapao-face').screenshot({path:'output/playwright/face-gallery.png'});
}
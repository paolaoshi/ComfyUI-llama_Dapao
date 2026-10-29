async (page) => {
    await page.reload();
    await page.getByRole('button',{name:'打开测试图库'}).click();
    await page.getByRole('button',{name:'全身单人・标准系 · 73 组 / 365 张'}).click();
    await page.getByRole('button',{name:'后仰坐・单膝立起 · 正面',exact:true}).click();
    const detail = page.getByRole('complementary', {name:'姿势详情'});
    await detail.getByRole('button',{name:'侧面',exact:true}).click();
    if (!(await detail.textContent()).includes('full_single_P001_side')) throw Error('angle selection');
    await page.keyboard.press('ArrowRight');
    if (!(await detail.textContent()).includes('full_single_P001_high')) throw Error('keyboard angle');
    await page.keyboard.press('ArrowDown');
    if (!(await detail.textContent()).includes('full_single_P002_high')) throw Error('keyboard pose');
    await page.getByRole('textbox',{name:'搜索姿势'}).fill('does-not-exist');
    if (!(await page.getByRole('region',{name:'姿势图库列表'}).textContent()).includes('没有找到')) throw Error('empty state');
    await page.getByRole('textbox',{name:'搜索姿势'}).fill('P001');
    await page.getByRole('combobox',{name:'展示方式'}).selectOption('single');
    if (await page.locator('article').count() !== 5) throw Error('single view');
    await page.getByRole('button',{name:'后仰坐・单膝立起 · 正面',exact:true}).click();
    await page.getByRole('button',{name:'使用此视角 · 正面'}).click();
    await page.waitForFunction(()=>!document.querySelector('dialog'));
    if (!(await page.locator('#selection').textContent()).includes('full_single_P001_front')) throw Error('selection');
    await page.reload();
    if (!(await page.locator('#selection').textContent()).includes('full_single_P001_front')) throw Error('harness persistence');
    await page.getByRole('button',{name:'打开测试图库'}).click();
    await page.getByRole('button',{name:'关闭',exact:true}).click();
    if (!(await page.locator('#selection').textContent()).includes('full_single_P001_front')) throw Error('cancel changed selection');
    console.log('PASS angles, keyboard, empty search, single mode, apply, persistence harness, cancel');
}

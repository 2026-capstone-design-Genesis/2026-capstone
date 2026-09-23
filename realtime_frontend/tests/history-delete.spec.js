import { test, expect } from '@playwright/test';

test('checked archive deletion removes row and reports deleted files', async ({ page }) => {
  let rows = [1, 2].map(id => ({ id, run: 'saved', file: `batch_000${id}.json`, start: 0, end: 30, count: 1 }));
  await page.route('**/api/history', route => route.fulfill({ json: rows }));
  await page.route('**/api/history/delete', route => {
    const { items } = route.request().postDataJSON();
    expect(items).toEqual([{ run: 'saved', file: 'batch_0001.json' }]);
    rows = rows.filter(row => row.file !== items[0].file);
    return route.fulfill({ json: { deleted: 1, removed_files: 3 } });
  });
  await page.goto('/#admin=local-ui-verification-20260909');
  await page.getByRole('button', { name: /영상 분석 요약/ }).click();
  await page.getByRole('button', { name: '분석 보관함', exact: true }).click();
  await page.getByRole('checkbox', { name: 'saved batch_0001.json 선택', exact: true }).check();
  await page.getByRole('button', { name: '선택 삭제 (1)', exact: true }).click();
  await expect(page.getByRole('checkbox', { name: 'saved batch_0001.json 선택', exact: true })).toHaveCount(0);
  await expect(page.getByRole('checkbox', { name: 'saved batch_0002.json 선택', exact: true })).toBeVisible();
  await expect(page.getByText('분석 1개와 파일 3개를 삭제했습니다.', { exact: true })).toBeVisible();
});

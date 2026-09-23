import { test, expect } from '@playwright/test';

test('upload tab submits video and displays analysis without camera pairing', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  let submitted = false;
  await page.route('**/api/upload-tests**', async route => {
    if (route.request().method() === 'POST') {
      expect(route.request().postDataBuffer().toString()).toBe('test video bytes');
      submitted = true;
      return route.fulfill({ status: 202, json: { id: 'test-job', status: 'analyzing', filename: 'sample.mp4' } });
    }
    await route.fulfill({ json: { id: 'test-job', run: 'test-run', status: 'done', filename: 'sample.mp4', result: {
      batch: { end: 300, skimming: { region_count: 25, delta_t_seconds: 4, gamma: .75, summary_ratio: .15, hierarchy: 'area' } },
      observations: Array.from({ length: 25 }, (_, index) => ({ id: index + 1, timestamp: index * 12, start_time: index * 12, end_time: index * 12 + 2, quality_status: index % 3 ? 'usable' : 'rejected', motion_percent: 0, image_url: 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180"><rect width="320" height="180" fill="#465574"/></svg>') })),
    } } });
  });
  await page.goto('/#admin=local-ui-verification-20260909');
  await page.getByRole('button', { name: '03 업로드 테스트' }).click();
  await expect(page.getByRole('heading', { name: '영상 업로드 테스트', exact: true })).toBeVisible();
  await page.locator('#test-video').setInputFiles({ name: 'sample.mp4', mimeType: 'video/mp4', buffer: Buffer.from('test video bytes') });
  await page.getByRole('button', { name: '분석 실행', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'sample.mp4 분석 결과' })).toBeVisible();
  expect(submitted).toBe(true);
  await expect(page.locator('.upload-test-frames button')).toHaveCount(25);
  await page.locator('.upload-test-frames button').last().click();
  await expect(page.locator('.upload-test-frames button').last()).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: '01 카메라 연동' }).click();
  await expect(page.locator('.upload-test-screen')).toBeHidden();
  await page.getByRole('button', { name: '03 업로드 테스트' }).click();
  await expect(page.locator('.upload-test-frames button')).toHaveCount(25);
  await page.locator('.upload-test-container').evaluate(element => element.scrollTop = 0);
  await page.screenshot({ path: '../.local-realtime/screenshots/upload-test.png' });
  expect(errors).toEqual([]);
});

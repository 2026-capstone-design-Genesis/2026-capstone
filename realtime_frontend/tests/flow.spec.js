import { test, expect } from '@playwright/test';

test('QR 연결 → 실시간 저장 → 영상 분석 → 연동 종료', async ({ page, browser }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/#admin=local-ui-verification-20260909');
  await expect(page.getByRole('button', { name: 'QR 연결 코드 생성' })).toBeEnabled();
  await page.getByRole('button', { name: '10분', exact: true }).click();
  await expect(page.getByRole('button', { name: '10분', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'QR 연결 코드 생성' }).click();
  await expect(page.locator('.qr-code svg')).toBeVisible();
  await page.screenshot({ path: '../.local-realtime/screenshots/pairing-dark.png', fullPage: true, animations: 'disabled' });
  await page.getByRole('button', { name: '밝은 모드로 전환' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.screenshot({ path: '../.local-realtime/screenshots/pairing-light.png', fullPage: true, animations: 'disabled' });
  const config = await (await page.request.get('/api/config')).json();
  const session = await (await page.request.get(`/api/sessions/${config.sessions.at(-1).id}`)).json();
  expect(session.interval_seconds).toBe(600);

  // 별도 컨텍스트의 가상 카메라만 사용한다. 사용자의 카메라 권한이나 기기는 접근하지 않는다.
  const phoneContext = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true });
  const phone = await phoneContext.newPage();
  phone.on('pageerror', error => errors.push(error.message));
  await phone.addInitScript(() => {
    Object.defineProperty(navigator.mediaDevices, 'getUserMedia', { value: async () => {
      const canvas = document.createElement('canvas');
      canvas.width = 640; canvas.height = 360;
      const context = canvas.getContext('2d');
      let n = 0;
      const draw = () => {
        context.fillStyle = '#1c2740'; context.fillRect(0, 0, 640, 360);
        context.fillStyle = '#a28afc'; context.fillRect((n++ * 7) % 500, 170, 140, 90);
        context.fillStyle = '#ffffff'; context.font = '26px sans-serif';
        context.fillText('TEST VIDEO / NOT A REAL CAMERA', 45, 100);
        requestAnimationFrame(draw);
      };
      draw();
      return canvas.captureStream(10);
    } });
  });
  try {
    await phone.goto(session.phone_url);
    await phone.getByRole('button', { name: '카메라 연결', exact: true }).click();
    await expect(phone.getByText('관제 PC로 영상이 전송되고 있습니다. 이 화면을 켜 두세요.')).toBeVisible();
    await phone.screenshot({ path: '../.local-realtime/screenshots/phone-test.png', fullPage: true });
    await expect(page.getByRole('button', { name: '영상 확인하기' })).toBeEnabled();
    await page.getByRole('button', { name: '영상 확인하기' }).click();
    await expect(page.getByAltText('휴대폰 실시간 카메라 영상')).toBeVisible();
    for (const label of ['30초', '1분', '30분', '1시간', '5분']) {
      await page.getByRole('button', { name: label, exact: true }).click();
      await expect(page.getByRole('button', { name: label, exact: true })).toHaveAttribute('aria-pressed', 'true');
    }
    await page.getByRole('button', { name: '지금 분석', exact: true }).click();
    await page.getByRole('button', { name: /영상 분석.*요약 · 행동 관찰 · 쇼츠/ }).click();
    const analysisPanel = page.getByRole('region', { name: '영상 분석 결과 패널' });
    await expect(analysisPanel).toBeVisible();
    await expect(analysisPanel.getByRole('heading', { name: 'AI 영상 요약' })).toBeVisible();
    await expect(analysisPanel.getByRole('heading', { name: '움직임 요약 영상' })).toBeVisible();
    await expect(page.locator('.observation-row').first()).toBeVisible({ timeout: 60000 });
    const summaryVideo = analysisPanel.locator('.analysis-media-section video');
    await expect(summaryVideo).toBeVisible();
    await expect.poll(() => summaryVideo.evaluate(video => video.error ? `error-${video.error.code}` : video.duration > 0 ? 'ready' : 'loading'), { timeout: 15000 }).toBe('ready');
    await page.locator('.observation-row').first().click();
    const shortVideo = analysisPanel.locator('.analysis-selected video');
    await expect(shortVideo).toBeVisible();
    await expect.poll(() => shortVideo.evaluate(video => video.error ? `error-${video.error.code}` : video.duration > 0 ? 'ready' : 'loading'), { timeout: 15000 }).toBe('ready');
    await expect(page.locator('.live-pill')).toHaveText('키프레임');
    await expect(page.locator('.timeline-marker.selected')).toHaveCount(1);
    await page.getByRole('button', { name: '영상 분석 패널 닫기' }).click();
    await page.getByRole('button', { name: '실시간으로', exact: true }).click();
    await page.getByRole('button', { name: '다크 모드로 전환' }).click();
    await page.screenshot({ path: '../.local-realtime/screenshots/monitor-dark-test.png', fullPage: true, animations: 'disabled' });
    const overflow = await page.evaluate(() => ({
      horizontal: document.documentElement.scrollWidth > innerWidth,
      height: document.querySelector('.app-shell').getBoundingClientRect().height,
      viewport: innerHeight,
    }));
    expect(overflow.horizontal).toBe(false);
    expect(overflow.height).toBe(overflow.viewport);
    await page.getByRole('button', { name: '연동 종료', exact: true }).click();
    await expect(page.getByText('카메라 연동이 종료되었습니다')).toBeVisible();
    await expect(phone.getByRole('button', { name: '카메라 다시 연결' })).toBeVisible();
    await expect(page.getByRole('link', { name: '분석 결과', exact: true })).toBeVisible();
    expect(errors).toEqual([]);
  } finally {
    await page.request.post(`/api/sessions/${session.id}/stop`);
    await phoneContext.close();
  }
});

test('영상 분석 패널의 요약·행동 선택·닫기와 작은 화면 접근성', async ({ page }) => {
  // 테스트 전용 장면은 브라우저 응답에만 넣으며 실제 분석 결과로 저장하지 않는다.
  const fixture = { id: 'drawer-test', status: 'ended', received_frames: 24, interval_seconds: 30, elapsed_seconds: 720,
    caption_queue_count: 3,
    batches: [{ id: 1, start: 0, end: 720, source_frames: 240, motion_frames: 24 }],
    analysis_report: { status: 'complete', ai_model: 'test-model', ai_summary: '전체 영상에서 주기적인 장면 변화와 인원 이동이 관찰되었습니다.', recorded_seconds: 720, source_frames: 240, motion_frames: 24 },
    total_keyframes: 24, keyframes: Array.from({ length: 24 }, (_, index) => ({ id: `frame-${index}`, batch_id: 1, timestamp: index * 30,
      title: `테스트 장면 ${index + 1}`, caption: `${index * 30}초 전후에서 사람의 위치 변화가 관찰됩니다.`, motion_percent: index + 1,
      caption_status: index === 0 ? 'pending' : 'ok',
      image_url: 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180"><rect width="320" height="180" fill="#26334d"/><text x="35" y="95" fill="white" font-size="22">테스트 전용 이미지</text></svg>') })) };
  await page.route('**/api/config', route => route.fulfill({ json: { sessions: [fixture] } }));
  await page.route('**/api/sessions/drawer-test', route => route.fulfill({ json: fixture }));
  await page.routeWebSocket('**/ws/viewer/drawer-test', socket => socket.send(JSON.stringify(fixture)));
  await page.setViewportSize({ width: 1280, height: 640 });
  await page.goto('/');
  const toggle = page.getByRole('button', { name: /영상 분석.*요약 · 행동 관찰 · 쇼츠/ });
  await expect(toggle).toBeInViewport();
  await toggle.click();
  const drawer = page.getByRole('region', { name: '영상 분석 결과 패널' });
  await expect(drawer).toBeVisible();
  await expect(drawer.getByText('전체 영상에서 주기적인 장면 변화와 인원 이동이 관찰되었습니다.')).toBeVisible();
  await expect(drawer.getByText(/GPU 대기열 총 3개/)).toBeVisible();
  await expect(page.locator('.observation-row')).toHaveCount(24);
  await page.locator('.observation-row').nth(3).click();
  await expect(page.locator('.observation-row.selected')).toHaveCount(1);
  await expect(page.locator('.live-pill')).toHaveText('키프레임');
  await expect(page.locator('.timeline-marker.selected')).toHaveCount(1);
  await page.screenshot({ path: '../.local-realtime/screenshots/analysis-drawer-dark.png', animations: 'disabled' });
  await page.keyboard.press('Escape');
  await expect(drawer).toHaveCount(0);
  await expect(toggle).toBeFocused();
  await page.getByRole('button', { name: '밝은 모드로 전환' }).click();
  await toggle.click();
  await page.screenshot({ path: '../.local-realtime/screenshots/analysis-drawer-light.png', animations: 'disabled' });
  await page.getByRole('button', { name: '영상 분석 패널 닫기' }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await toggle.click();
  await expect(page.locator('.observation-list')).toBeVisible();
  await expect(page.getByRole('button', { name: '영상 분석 패널 닫기' })).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('카메라 수에 따라 1·2·3·4 분할 그리드를 사용한다', async ({ page }) => {
  const cameras = count => Array.from({ length: count }, (_, index) => ({
    id: `camera-${index + 1}`, label: `카메라 ${String(index + 1).padStart(2, '0')}`, slot: index + 1,
    phone_url: `https://example.test/phone#${index}`, status: 'waiting', interval_seconds: 300,
    received_frames: 0, elapsed_seconds: 0, keyframes: [], batches: [], analysis_report: {}, total_keyframes: 0,
  }));
  for (const count of [1, 2, 3, 4]) {
    const rows = cameras(count);
    await page.route('**/api/config', route => route.fulfill({ json: { public_origin: 'https://example.test', sessions: rows } }));
    for (const camera of rows) {
      await page.route(`**/api/sessions/${camera.id}`, route => route.fulfill({ json: camera }));
      await page.routeWebSocket(`**/ws/viewer/${camera.id}`, socket => socket.send(JSON.stringify(camera)));
    }
    await page.goto('/');
    await page.getByRole('button', { name: '영상 확인', exact: true }).first().click();
    await expect(page.locator(`.camera-grid.count-${count}`)).toBeVisible();
    await expect(page.locator('.camera-grid .camera-tile')).toHaveCount(count);
    await page.unrouteAll({ behavior: 'ignoreErrors' });
  }
});

import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  workers: 1,
  timeout: 45000,
  use: { baseURL: 'http://localhost:18765', channel: 'chrome', headless: true, viewport: { width: 1600, height: 1000 } },
  reporter: 'list',
  outputDir: '../.local-realtime/ui-test-results',
  webServer: {
    command: `set REALTIME_AI_ENABLED=false&& .\\.venv-realtime\\Scripts\\python.exe -u -c "from pathlib import Path; import tempfile, uvicorn; from realtime_server import create_app; verification=tempfile.TemporaryDirectory(prefix='clullm-ui-test-'); uvicorn.run(create_app(admin_token='local-ui-verification-20260909', public_origin='http://localhost:18765', output_root=Path(verification.name)),host='127.0.0.1',port=18765,access_log=False,ws_max_size=420000)"`,
    cwd: '..',
    url: 'http://localhost:18765',
    reuseExistingServer: false,
    timeout: 30000,
  },
});

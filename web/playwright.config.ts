import { defineConfig, devices } from '@playwright/test';

// Three static servers, same tree (`dist/site`, built by scripts/build-site.mjs), three header
// profiles — see scripts/serve.mjs for exactly what each sends. Ports in the required 3000-3999
// range. `webServer` is an array: Playwright starts and health-checks all three before any test.
const ROOT = 'dist/site';
// JAI_PORT_BASE shifts all three ports — parallel agents in sibling jj workspaces otherwise
// reuse each other's servers (reuseExistingServer) and test against a foreign dist/site.
const PORT_BASE = Number(process.env.JAI_PORT_BASE ?? 3031);
export const PORTS = { plain: PORT_BASE, isolated: PORT_BASE + 1, strict: PORT_BASE + 2 };

export default defineConfig({
  testDir: 'tests',
  timeout: 60_000,
  fullyParallel: false, // decode workers share this box's CPU with other agents; avoid pileup
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  webServer: Object.entries(PORTS).map(([profile, port]) => ({
    command: `node scripts/serve.mjs ${ROOT} ${profile} ${port}`,
    port,
    reuseExistingServer: !process.env.CI,
    timeout: 20_000,
  })),
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
  ],
});

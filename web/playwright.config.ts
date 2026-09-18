import { createHash } from 'node:crypto';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig, devices } from '@playwright/test';

// Three static servers, same tree (`dist/site`, built by scripts/build-site.mjs), three header
// profiles — see scripts/serve.mjs for exactly what each sends. Ports in the required 3000-3999
// range. `webServer` is an array: Playwright starts and health-checks all three before any test.
const ROOT = 'dist/site';
// JAI_PORT_BASE shifts all three ports. Without it the base is derived deterministically from
// this checkout's absolute path (3000..3997, three consecutive ports): sibling jj workspaces get
// disjoint servers, so `reuseExistingServer` can never hand a test a foreign dist/site — the
// failure mode cache-swap.spec.ts's manifest-identity assertion guards against is a stale server
// of THIS workspace squatting on the port, not a sibling's.
const webDir = dirname(fileURLToPath(import.meta.url)); // <checkout>/web — unique per workspace
const derived = 3000 + (createHash('sha256').update(webDir).digest().readUInt16LE(0) % 997);
const PORT_BASE = Number(process.env.JAI_PORT_BASE ?? derived);
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
    // benchmark-gpu.spec.ts' rows only make sense from the WebGPU-enabled launch below,
    // scheduling.spec.ts' timing bounds are calibrated to chromium's scheduler, and
    // threads.spec.ts' scaling numbers are chromium-only (its header) — those files are
    // simply not part of the other projects rather than skipped at run time.
    { name: 'chromium', use: { ...devices['Desktop Chrome'] }, testIgnore: '**/benchmark-gpu.spec.ts' },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
      testIgnore: ['**/scheduling.spec.ts', '**/benchmark-gpu.spec.ts', '**/threads.spec.ts'],
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
      testIgnore: ['**/scheduling.spec.ts', '**/benchmark-gpu.spec.ts', '**/threads.spec.ts'],
    },
    {
      // WebGPU-enabled Chromium: `--enable-unsafe-webgpu` exposes navigator.gpu without an
      // origin trial. `WEBGPU_ADAPTER` selects the adapter policy:
      //   hardware    Vulkan + ANGLE-on-Vulkan + ignore-gpu-blocklist; Dawn must hand back a
      //               real (non-fallback) adapter — the specs FAIL when it doesn't
      //   swiftshader --use-webgpu-adapter=swiftshader for GPU-less hosts (exercises the GPU
      //               code path through Dawn's software adapter; tests report it as a software
      //               adapter, never as hardware)
      //   unset       default WebGPU behaviour — whatever Dawn hands back (non-fatal either way)
      name: 'chromium-webgpu',
      testIgnore: '**/threads.spec.ts',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--ignore-gpu-blocklist',
            ...(process.env.WEBGPU_ADAPTER === 'hardware' ? ['--use-angle=vulkan'] : []),
            ...(process.env.WEBGPU_ADAPTER === 'swiftshader' ? ['--use-webgpu-adapter=swiftshader'] : []),
          ],
        },
      },
    },
  ],
});

#!/usr/bin/env node
// Cold-load startup measurement for the demo page, against any URL — a local serve.mjs tree,
// GitHub Pages, Cloudflare Pages. Per run (a fresh browser = cold HTTP cache, Cache API and
// service-worker registry): counts every network request the page and all of its workers make
// (route interception is browser-context-wide, so dedicated-worker traffic — the rayon child
// worker's module imports included — is counted too), times navigation -> the first decoded
// card, and reports crossOriginIsolated, the build variant the pool picked, whether the coi
// service worker installed and the page reloaded.
//
// usage:
//   node web/scripts/measure-startup.mjs --url <url> --label <label> [--runs 3] [--gpu auto|off]
//                                        [--timeout 60000] [--out benchmarks/wasm_startup_<date>.tsv]
// Prints one TSV row per run; with --out also appends to the TSV and creates a .meta
// (host / commit / command / url) on first write, like tests/benchmark.spec.ts.
import { chromium } from '@playwright/test';
import { appendFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : dflt;
};
const URL_ = opt('url');
const LABEL = opt('label', URL_ ? new URL(URL_).host : 'local');
const RUNS = Number(opt('runs', 3));
const GPU = opt('gpu', 'auto');
const TIMEOUT = Number(opt('timeout', 60000));
const OUT = opt('out');
if (!URL_) {
  console.error('usage: measure-startup.mjs --url <url> --label <label> [--runs N] [--gpu auto|off] [--timeout ms] [--out file.tsv]');
  process.exit(2);
}

// A card counts as decoded once data-state says done, or — pre-data-state trees — once its
// canvas grew past the 300x150 default (same predicate as tests/scheduling.spec.ts).
const DONE_COUNT_SRC = `(() => [...document.querySelectorAll('.card')].filter((card) => {
  if (card.dataset.state) return card.dataset.state === 'done';
  const c = card.querySelector('canvas');
  return c && c.width > 300;
}).length)()`;

const classify = (url) => {
  const path = new URL(url).pathname;
  if (/\.zjb$/.test(path)) return 'models';
  if (/\.jai$/.test(path)) return 'streams';
  if (/\.wasm$/.test(path)) return 'wasm';
  if (/\.(m?js)$/.test(path) || path.endsWith('/worker.js')) return 'js';
  if (/manifest\.json$/.test(path)) return 'manifest';
  if (/\.(html?|css)$/.test(path) || path === '/' || !/\.[a-z0-9]+$/i.test(path)) return 'doc';
  return 'other';
};

const HEADER = 'date\tlabel\turl\trun\trequests\tjs_requests\tdemo_js\tswcoi_js\tmodel_requests\tstream_requests\twasm_requests\tfirst_image_ms\tisolated\tvariant\tsw_regs\tsw_reload\tgpu\n';
const rows = [];

for (let run = 0; run < RUNS; run++) {
  const launchArgs = [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--ignore-gpu-blocklist',
    ...(process.env.WEBGPU_ADAPTER === 'hardware' ? ['--use-angle=vulkan'] : []),
    ...(process.env.WEBGPU_ADAPTER === 'swiftshader' ? ['--use-webgpu-adapter=swiftshader'] : []),
  ];
  const browser = await chromium.launch({ args: launchArgs });
  const ctx = await browser.newContext();
  const requests = [];
  await ctx.route('**/*', (route) => {
    requests.push(route.request().url());
    return route.continue();
  });
  const page = await ctx.newPage();
  let navigations = 0;
  page.on('framenavigated', (frame) => {
    if (frame === page.mainFrame() && frame.url() !== 'about:blank') navigations++;
  });
  const t0 = Date.now();
  let firstImageMs = -1;
  try {
    const sep = URL_.includes('?') ? '&' : '?';
    await page.goto(`${URL_}${sep}gpu=${GPU}`, { timeout: TIMEOUT });
    await page.waitForFunction(`${DONE_COUNT_SRC} >= 1`, { timeout: TIMEOUT, polling: 100 });
    firstImageMs = Date.now() - t0;
  } catch (err) {
    console.error(`run ${run}: ${String(err && err.message || err)}`);
  }
  const probe = await page
    .evaluate(async () => {
      const regs = 'serviceWorker' in navigator ? (await navigator.serviceWorker.getRegistrations()).length : -1;
      return {
        isolated: typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated,
        swRegs: regs,
        build: [...document.querySelectorAll('#status dt')].find((d) => d.textContent === 'build')?.nextElementSibling?.textContent ?? '',
        cards: document.querySelectorAll('.card').length,
      };
    })
    .catch(() => ({ isolated: null, swRegs: -1, build: '', cards: 0 }));
  const counts = { total: requests.length, js: 0, models: 0, streams: 0, wasm: 0, manifest: 0, demoJs: 0, swCoi: 0 };
  for (const u of requests) {
    const k = classify(u);
    if (k === 'js') counts.js++;
    else if (k === 'models') counts.models++;
    else if (k === 'streams') counts.streams++;
    else if (k === 'wasm') counts.wasm++;
    else if (k === 'manifest') counts.manifest++;
    if (/\/demo\.js(\?|$)/.test(u)) counts.demoJs++;
    if (/\/sw-coi\.js(\?|$)/.test(u)) counts.swCoi++;
  }
  // sw_reload: more than one top-level navigation means the coi loader reloaded the page.
  const swReload = navigations > 1 ? 1 : 0;
  const date = new Date().toISOString().slice(0, 10);
  const row = [date, LABEL, URL_, run, counts.total, counts.js, counts.demoJs, counts.swCoi, counts.models, counts.streams, counts.wasm, firstImageMs, probe.isolated, probe.build, probe.swRegs, swReload, GPU].join('\t');
  rows.push(row);
  console.log(row);
  if (run === 0) {
    console.error(`[info] ${requests.length} requests; sw regs=${probe.swRegs}; navigations=${navigations}; cards=${probe.cards}`);
    for (const u of requests) if (classify(u) === 'js') console.error(`[js] ${u}`);
  }
  await browser.close();
}

if (OUT) {
  const tsv = OUT.startsWith('/') ? OUT : join(here, '..', '..', OUT);
  mkdirSync(dirname(tsv), { recursive: true });
  if (!existsSync(tsv)) writeFileSync(tsv, HEADER);
  appendFileSync(tsv, rows.join('\n') + '\n');
  const meta = tsv.replace(/\.tsv$/, '.meta');
  if (!existsSync(meta)) {
    const safe = (cmd) => { try { return execSync(cmd, { cwd: join(here, '..', '..'), stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim(); } catch { return 'unknown'; } };
    const commit = safe('jj log -r @ --no-graph -T "commit_id.short()"');
    writeFileSync(meta, `commit\t${commit}\nhost\t${safe('hostname')}\ncommand\tnode ${process.argv.slice(1).join(' ')}\ndate\t${new Date().toISOString()}\n`);
  }
  console.error(`appended ${rows.length} row(s) to ${tsv}`);
}

#!/usr/bin/env node
// Assembles the servable site tree at `web/dist/site/` (gitignored) from:
//   web/src/*.js              -> site/src/            (worker.js, pool.js, polyfill.js)
//   web/dist/pkg-{simd,threads} -> site/dist/pkg-*     (run scripts/build-wasm.sh first)
//   web/demo/{index.html,demo.js,demo.css,viewer.js,IMAGES.md,sw-coi.js,coi-loader.js,_headers} -> site/
//   web/.demo-assets/*.jai    -> site/streams/         (run scripts/fetch-demo-assets.mjs first)
//   web/.demo-assets/m*.zjb   -> site/models/
//   web/.demo-assets/manifest.json -> site/manifest.json
//   upstream-notices/LICENSE  -> site/upstream-notices/LICENSE
// Used both by the Playwright suite (serve.mjs points at this tree) and by the Pages workflow.
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const web = join(dirname(fileURLToPath(import.meta.url)), '..');
const repo = join(web, '..');
const site = join(web, 'dist', 'site');

function need(path, hint) {
  if (!existsSync(path)) {
    console.error(`missing ${path}\n  -> ${hint}`);
    process.exit(1);
  }
}
need(join(web, 'dist', 'pkg-simd', 'zenjpegai.js'), 'run: web/scripts/build-wasm.sh simd');
const haveThreads = existsSync(join(web, 'dist', 'pkg-threads', 'zenjpegai.js'));
if (!haveThreads) console.warn('pkg-threads not built; site will only offer the simd package (run build-wasm.sh threads to add it)');
const haveWebgpu = existsSync(join(web, 'dist', 'pkg-webgpu', 'zenjpegai.js'));
if (!haveWebgpu) console.warn('pkg-webgpu not built; WebGPU synthesis is unavailable (run build-wasm.sh webgpu to add it)');
const haveWebgpuThreads = existsSync(join(web, 'dist', 'pkg-webgpu-threads', 'zenjpegai.js'));
if (!haveWebgpuThreads) console.warn('pkg-webgpu-threads not built; isolated pages use pkg-webgpu (single-threaded CPU stages — run build-wasm.sh webgpu-threads to add it)');
const assets = join(web, '.demo-assets');
need(join(assets, 'manifest.json'), 'run: node web/scripts/fetch-demo-assets.mjs');

rmSync(site, { recursive: true, force: true });
mkdirSync(site, { recursive: true });

cpSync(join(web, 'src'), join(site, 'src'), { recursive: true });
mkdirSync(join(site, 'dist'), { recursive: true });
cpSync(join(web, 'dist', 'pkg-simd'), join(site, 'dist', 'pkg-simd'), { recursive: true });
if (haveThreads) cpSync(join(web, 'dist', 'pkg-threads'), join(site, 'dist', 'pkg-threads'), { recursive: true });
if (haveWebgpu) cpSync(join(web, 'dist', 'pkg-webgpu'), join(site, 'dist', 'pkg-webgpu'), { recursive: true });
if (haveWebgpuThreads) cpSync(join(web, 'dist', 'pkg-webgpu-threads'), join(site, 'dist', 'pkg-webgpu-threads'), { recursive: true });

for (const f of ['index.html', 'demo.js', 'demo.css', 'viewer.js', 'IMAGES.md', 'sw-coi.js', 'coi-loader.js', '_headers']) {
  cpSync(join(web, 'demo', f), join(site, f));
}

mkdirSync(join(site, 'streams'), { recursive: true });
mkdirSync(join(site, 'models'), { recursive: true });
mkdirSync(join(site, '_native'), { recursive: true }); // reference PNGs: Playwright pixel-parity oracle only
for (const f of readdirSync(assets)) {
  if (f.endsWith('.native.png')) cpSync(join(assets, f), join(site, '_native', f));
  else if (f.endsWith('.jai')) cpSync(join(assets, f), join(site, 'streams', f));
  else if (f.endsWith('.zjb')) cpSync(join(assets, f), join(site, 'models', f));
  else if (f === 'manifest.json') cpSync(join(assets, f), join(site, 'manifest.json'));
}

// `<link rel="preload" as="fetch" crossorigin="use-credentials">` for the FIRST card's model
// pair: the fetch then starts at HTML parse, ahead of even the manifest read in demo.js. The
// `?v=` digests only exist in manifest.json, so the links can only be stamped here at build
// time — never hard-code them in demo/index.html. `use-credentials` + `prefetch.js`'s
// `credentials: 'include'` are the pairing that makes the later fetch() reuse the preloaded
// response instead of double-fetching (measured: anonymous `crossorigin` does not match
// either `omit` or `same-origin` fetches).
{
  const mf = JSON.parse(readFileSync(join(site, 'manifest.json'), 'utf8'));
  const first = mf.images?.[0]?.variants?.[0];
  const preload = [];
  if (first?.modelId != null && first?.operatingPoint) {
    for (const name of [`m${first.modelId}_common.zjb`, `m${first.modelId}_${first.operatingPoint}.zjb`]) {
      const sha = mf.models?.[name]?.sha256;
      preload.push(`<link rel="preload" as="fetch" href="models/${name}${sha ? `?v=${sha}` : ''}" crossorigin="use-credentials">`);
    }
  }
  if (preload.length) {
    const htmlPath = join(site, 'index.html');
    const html = readFileSync(htmlPath, 'utf8');
    writeFileSync(htmlPath, html.replace('</head>', `${preload.join('\n')}\n</head>`));
  }
}

const fixtures = join(web, 'tests', 'fixtures');
if (existsSync(fixtures)) {
  for (const f of readdirSync(fixtures)) cpSync(join(fixtures, f), join(site, f));
}

mkdirSync(join(site, 'upstream-notices'), { recursive: true });
cpSync(join(repo, 'upstream-notices', 'LICENSE'), join(site, 'upstream-notices', 'LICENSE'));

console.log(`site built at ${site}`);

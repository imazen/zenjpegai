#!/usr/bin/env node
// Static file server for the demo/tests, in three header profiles (all serve the same tree):
//   plain    - no COOP/COEP, no CSP: `pkg-threads` cannot be used (not cross-origin isolated),
//              the polyfill/demo must fall back to `pkg-simd`.
//   isolated - `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy:
//              require-corp`: the page is cross-origin isolated, SharedArrayBuffer works,
//              `pkg-threads` runs.
//   strict   - worker-src needs blob: for wasm-bindgen-rayon's own bundler-less child-worker
//              spawn (workerHelpers.no-bundler.js fetches its own script as a blob and starts
//              a Worker from the object URL); without it initThreadPool hangs forever waiting
//              on child workers a bare CSP silently blocked (caught 2026-09-17 running this in
//              a real browser for the first time). isolated headers plus a CSP restrictive
//              enough to prove the polyfill needs
//              nothing beyond it: `default-src 'self'; script-src 'self' 'wasm-unsafe-eval';
//              worker-src 'self'; img-src 'self'; style-src 'self' 'unsafe-inline';
//              connect-src 'self'`. No `blob:`, no `data:`, no `unsafe-eval` (only
//              `wasm-unsafe-eval`, which covers only WebAssembly.instantiate/compile). The
//              "img" render mode (blob: URLs) is expected to be BLOCKED under this profile —
//              that is the point of the test that exercises it under "strict".
//
// usage: node web/scripts/serve.mjs <root> <profile> [port]
// Prints "LISTENING <port>" on the first stdout line once bound, for scripts to parse.
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { join, extname, normalize } from 'node:path';

const [, , rootArg, profileArg, portArg] = process.argv;
const root = normalize(rootArg || '.');
const profile = profileArg || 'plain';
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jai': 'application/octet-stream',
  '.zjb': 'application/octet-stream',
  '.bits': 'application/octet-stream',
  '.css': 'text/css; charset=utf-8',
};

function headersFor(profile) {
  const h = {};
  if (profile === 'isolated' || profile === 'strict') {
    h['Cross-Origin-Opener-Policy'] = 'same-origin';
    h['Cross-Origin-Embedder-Policy'] = 'require-corp';
  }
  if (profile === 'strict') {
    h['Content-Security-Policy'] =
      "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:; " +
      "img-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; base-uri 'none'";
  }
  return h;
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');
    let path = decodeURIComponent(url.pathname);
    if (path.endsWith('/')) path += 'index.html';
    const full = join(root, normalize(path));
    if (!full.startsWith(root)) {
      res.writeHead(403).end('forbidden');
      return;
    }
    const st = await stat(full).catch(() => null);
    if (!st || !st.isFile()) {
      res.writeHead(404).end('not found');
      return;
    }
    const body = await readFile(full);
    const headers = { 'Content-Type': MIME[extname(full)] || 'application/octet-stream', ...headersFor(profile) };
    res.writeHead(200, headers).end(body);
  } catch (err) {
    res.writeHead(500).end(String(err));
  }
});

const requested = portArg ? Number(portArg) : 0;
server.listen(requested || 0, '127.0.0.1', () => {
  console.log(`LISTENING ${server.address().port}`);
});

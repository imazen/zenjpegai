#!/usr/bin/env node
// Downloads the `demo-assets-v1` GitHub release assets (8 imazen-26 images x 2 rates as `.jai`
// codestreams, the 4 models' `common`/`bop` bundle halves, the reference-decoder PNGs used as
// the Playwright pixel-parity oracle, and `manifest.json`) into `web/.demo-assets/` (gitignored).
// The repo is private today (see web/README.md "GitHub Pages" / PORTING.md), so this needs `gh`
// authenticated against it — locally your own `gh auth login`, in CI the workflow's own token
// (which has read access to its own repo's releases without anything extra).
// usage: node web/scripts/fetch-demo-assets.mjs [--force]
import { existsSync, mkdirSync, readdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const dest = join(here, '..', '.demo-assets');
const force = process.argv.includes('--force');
const REPO = 'imazen/zenjpegai';
const TAG = 'demo-assets-v1';

if (existsSync(dest) && readdirSync(dest).length > 0 && !force) {
  console.log(`web/.demo-assets already populated (${readdirSync(dest).length} files); pass --force to re-download.`);
  process.exit(0);
}
mkdirSync(dest, { recursive: true });
console.log(`downloading ${TAG} assets from ${REPO} into ${dest} ...`);
execFileSync('gh', ['release', 'download', TAG, '--repo', REPO, '--dir', dest, '--clobber', '--pattern', '*'], {
  stdio: 'inherit',
});
const files = readdirSync(dest);
console.log(`got ${files.length} files`);
if (!files.includes('manifest.json')) {
  console.error('manifest.json missing from the download — release contents changed?');
  process.exit(1);
}

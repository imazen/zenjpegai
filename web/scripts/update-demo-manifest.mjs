#!/usr/bin/env node
// Re-stamps `web/.demo-assets/manifest.json` with the fields that make the published demo's
// asset URLs cache-safe: per-variant `file` + `sha256` (demo.js appends `?v=<sha256>` to the
// stream URL, so a stream swapped under the same file name gets a new cache key) and a
// `models` map of every `*.zjb` bundle's sha256/bytes (worker.js keys its Cache API entries
// on them). Run after `fetch-demo-assets.mjs`, then re-upload:
//   gh release upload demo-assets-v1 web/.demo-assets/manifest.json --clobber --repo imazen/zenjpegai
// The script also re-verifies the release's hygiene invariants: every `sourceFile` stays
// redacted to `<id>_..._<WxH>.<ext>` (the real imazen-26 filenames encode where/when a photo
// was taken — never publish them) and `bytes` still matches the file on disk.
// usage: node web/scripts/update-demo-manifest.mjs [--check]   (--check: verify, don't write)
import { readFileSync, writeFileSync, statSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const assets = join(dirname(fileURLToPath(import.meta.url)), '..', '.demo-assets');
const manifestPath = join(assets, 'manifest.json');
const checkOnly = process.argv.includes('--check');

const sha256 = (file) => createHash('sha256').update(readFileSync(join(assets, file))).digest('hex');
const bpp2 = (bpp) => String(Math.round(bpp * 100)).padStart(2, '0');
const REDACTED = /^[0-9a-z-]+_\.\.\._\d+x\d+\.[a-z0-9]+$/;

const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
let problems = 0;
const bad = (msg) => { console.error(`  ! ${msg}`); problems++; };

for (const img of manifest.images) {
  if (!REDACTED.test(img.sourceFile)) bad(`${img.slug}: sourceFile "${img.sourceFile}" is not redacted`);
  for (const v of img.variants) {
    const file = v.file || `${img.slug}_bpp${bpp2(v.bpp)}.jai`;
    let st;
    try {
      st = statSync(join(assets, file));
    } catch {
      bad(`${img.slug}: ${file} not in .demo-assets`);
      continue;
    }
    if (v.bytes !== st.size) bad(`${img.slug}/${file}: manifest bytes ${v.bytes} != ${st.size}`);
    v.file = file;
    v.sha256 = sha256(file);
  }
}

manifest.models = {};
for (const f of readdirSync(assets).filter((f) => f.endsWith('.zjb')).sort()) {
  manifest.models[f] = { sha256: sha256(f), bytes: statSync(join(assets, f)).size };
}
manifest.generated = new Date().toISOString();

if (problems) {
  console.error(`${problems} problem(s); manifest left untouched`);
  process.exit(1);
}
const out = JSON.stringify(manifest, null, 2) + '\n';
if (checkOnly) {
  console.log('manifest verified: digests computable, sourceFile redaction intact');
} else {
  writeFileSync(manifestPath, out);
  console.log(`wrote ${manifestPath} (${out.length} bytes; ${Object.keys(manifest.models).length} model bundles, ${manifest.images.length} images)`);
}

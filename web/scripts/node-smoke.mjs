// Decode one stream with the single-threaded package under node and compare with a PNG:
//   node web/scripts/node-smoke.mjs <bundle.zjb> <stream> <expected.png>
import { readFileSync } from 'node:fs';
import { decodePng, diffSamples } from '../../scripts/wasm/png.mjs';
import init, * as zj from '../dist/pkg-simd/zenjpegai.js';

const [bundle, stream, png] = process.argv.slice(2);
await init({ module_or_path: readFileSync(new URL('../dist/pkg-simd/zenjpegai_bg.wasm', import.meta.url)) });
console.log('build', zj.buildMode(), 'tier', zj.simdTier());
const bits = readFileSync(stream);
const info = zj.info(bits);
console.log('info', JSON.stringify(info));
console.log('hasModels before', zj.hasModels(info.modelId, info.operatingPoint));
zj.addModels(readFileSync(bundle));
console.log('hasModels after', zj.hasModels(info.modelId, info.operatingPoint));
const times = [];
for (let i = 0; i < Number(process.env.RUNS || 3); i++) {
  const t = performance.now();
  var img = zj.decode(bits);
  times.push(performance.now() - t);
}
times.sort((a, b) => a - b);
console.log(`decode ms: min ${times[0].toFixed(1)} median ${times[times.length >> 1].toFixed(1)} max ${times.at(-1).toFixed(1)} (n=${times.length})`);
const d = diffSamples({ width: img.width, height: img.height, channels: 4, data: img.rgba }, decodePng(readFileSync(png)));
console.log('diff vs', png, JSON.stringify(d));

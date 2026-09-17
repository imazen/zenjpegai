// `node png_diff.mjs a.png b.png`: count 8-bit samples that differ, as one TSV line.
import { readFileSync } from 'node:fs';
import { decodePng, diffSamples } from './png.mjs';

const [a, b] = process.argv.slice(2);
const d = diffSamples(decodePng(readFileSync(a)), decodePng(readFileSync(b)));
console.log(`${d.differing}\t${d.total}\t${d.max}\t${JSON.stringify(d.hist)}`);

// Run a wasm32-wasip1 binary under node's WASI: `node run_wasi.mjs prog.wasm [args...]`.
// Preopens: `/` -> `/` (read the models directory, reference vectors, write to ~/tmp).
// Doubles as the cargo test runner for the wasip1 target (see the justfile).
import { readFile } from 'node:fs/promises';
import { WASI } from 'node:wasi';

const [prog, ...args] = process.argv.slice(2);
const wasi = new WASI({
  version: 'preview1',
  args: [prog, ...args],
  env: process.env,
  preopens: { '/': '/' },
  returnOnExit: true,
});
const module = await WebAssembly.compile(await readFile(prog));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
process.exit(wasi.start(instance));

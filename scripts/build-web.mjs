/** Bundles the demo into docs/ exactly as CI rebuilds it for the diff check. */
import { build } from 'esbuild';
import { copyFileSync, cpSync, mkdirSync, rmSync } from 'node:fs';

const root = new URL('..', import.meta.url).pathname;
rmSync(`${root}docs`, { recursive: true, force: true });
mkdirSync(`${root}docs/ort`, { recursive: true });
mkdirSync(`${root}docs/data`, { recursive: true });

const shared = {
  bundle: true,
  minify: true,
  sourcemap: false,
  target: 'es2022',
  logLevel: 'warning',
};

await build({
  ...shared,
  entryPoints: [`${root}src/web/app.ts`],
  outfile: `${root}docs/app.js`,
  format: 'iife',
});

await build({
  ...shared,
  entryPoints: [`${root}src/web/worker.ts`],
  outfile: `${root}docs/worker.js`,
  format: 'iife',
});

copyFileSync(`${root}src/web/index.html`, `${root}docs/index.html`);
copyFileSync(`${root}src/web/style.css`, `${root}docs/style.css`);
for (const f of ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm']) {
  copyFileSync(`${root}node_modules/onnxruntime-web/dist/${f}`, `${root}docs/ort/${f}`);
}
cpSync(`${root}data/corpus`, `${root}docs/data/corpus`, { recursive: true });
cpSync(`${root}data/eval`, `${root}docs/data/eval`, { recursive: true });
cpSync(`${root}data/precomputed`, `${root}docs/data/precomputed`, { recursive: true });
console.log('docs/ built');

/**
 * Precomputes embeddings and rerank scores for the bundled corpus with the
 * exact models the browser uses, and writes them to data/precomputed/.
 * The demo then shows real model output instantly; the CI verify job
 * regenerates these files and fails if they drift.
 */
import { writeFileSync, mkdirSync } from 'node:fs';
import { computePrecomputed } from './precompute-core.mjs';

const root = new URL('..', import.meta.url).pathname;
const out = await computePrecomputed(root, console.log);
mkdirSync(`${root}data/precomputed`, { recursive: true });
const path = `${root}data/precomputed/night-sky.bge-small-en-v1.5.json`;
writeFileSync(path, JSON.stringify(out));
console.log(`wrote ${path} (${(JSON.stringify(out).length / 1024).toFixed(0)} KB, dims ${out.dims})`);

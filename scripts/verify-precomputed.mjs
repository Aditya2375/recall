/**
 * CI gate: recomputes the bundled precomputed artifact with the pinned
 * models and fails if the committed file has drifted. Vectors must match at
 * cosine >= 0.995 per row; rerank logits within 0.05 absolute.
 * The cosine gate is 0.995 rather than tighter because ONNX wasm SIMD yields
 * slightly different floats across CPUs (observed 0.9970 on ubuntu-latest vs
 * 1.000000 on the authoring machine); a corrupted or wrong-model artifact
 * lands orders of magnitude lower.
 */
import { readFileSync } from 'node:fs';
import { computePrecomputed } from './precompute-core.mjs';

const root = new URL('..', import.meta.url).pathname;
const stored = JSON.parse(readFileSync(`${root}data/precomputed/night-sky.bge-small-en-v1.5.json`, 'utf8'));
const fresh = await computePrecomputed(root, console.log);

const fail = (msg) => { console.error(`verify-precomputed: ${msg}`); process.exit(1); };

for (const key of ['version', 'corpus', 'embedModel', 'embedRevision', 'rerankModel', 'rerankRevision', 'queryInstruction', 'rerankTop', 'dims']) {
  if (JSON.stringify(stored[key]) !== JSON.stringify(fresh[key])) fail(`metadata mismatch on ${key}`);
}
if (JSON.stringify(stored.chunkIds) !== JSON.stringify(fresh.chunkIds)) fail('chunk id list mismatch');
if (JSON.stringify(stored.queryIds) !== JSON.stringify(fresh.queryIds)) fail('query id list mismatch');

const b64f32 = (s) => { const buf = Buffer.from(s, 'base64'); return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4); };
const cosine = (a, b) => {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
};
let worst = 1;
for (const field of ['chunkVectors', 'queryVectors']) {
  for (let i = 0; i < stored[field].length; i++) {
    const c = cosine(b64f32(stored[field][i]), b64f32(fresh[field][i]));
    if (c < worst) worst = c;
  }
}
if (worst < 0.995) fail(`worst vector cosine ${worst.toFixed(6)} < 0.995`);

let maxDiff = 0;
for (const qid of Object.keys(stored.rerankScores)) {
  const s = stored.rerankScores[qid]; const f = fresh.rerankScores[qid] ?? {};
  if (JSON.stringify(Object.keys(s).sort()) !== JSON.stringify(Object.keys(f).sort())) fail(`rerank doc set mismatch for ${qid}`);
  for (const [docId, score] of Object.entries(s)) {
    const d = Math.abs(score - f[docId]);
    if (d > maxDiff) maxDiff = d;
  }
}
if (maxDiff > 0.05) fail(`max rerank logit drift ${maxDiff.toFixed(4)} > 0.05`);
console.log(`verify-precomputed ok: worst vector cosine ${worst.toFixed(6)}, max rerank drift ${maxDiff.toFixed(4)}`);

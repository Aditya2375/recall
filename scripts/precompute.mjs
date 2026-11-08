/**
 * Precomputes embeddings and rerank scores for the bundled corpus with the
 * exact models the browser uses, and writes them to data/precomputed/.
 * The demo then shows real model output instantly; the CI verify job
 * regenerates these files and fails if they drift.
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { pipeline, AutoTokenizer, AutoModelForSequenceClassification, env } from '@huggingface/transformers';

env.cacheDir = new URL('../.cache/hf', import.meta.url).pathname;

export const EMBED_MODEL = 'Xenova/bge-small-en-v1.5';
export const EMBED_REVISION = 'ea104dacec62c0de699686887e3f920caeb4f3e3';
export const RERANK_MODEL = 'Xenova/ms-marco-MiniLM-L-6-v2';
export const RERANK_REVISION = 'a09144355adeed5f58c8ed011d209bf8ee5a1fec';
export const QUERY_INSTRUCTION = 'Represent this sentence for searching relevant passages: ';
export const RERANK_TOP = 25;

const { validateCorpus } = await import('../dist/src/core/corpus.js');
const { validateEvalSet } = await import('../dist/src/core/evaluate.js');
const { chunkStructure } = await import('../dist/src/core/chunk.js');

const root = new URL('..', import.meta.url).pathname;
const corpus = validateCorpus(JSON.parse(readFileSync(`${root}data/corpus/night-sky.json`, 'utf8')));
const evalSet = validateEvalSet(
  JSON.parse(readFileSync(`${root}data/eval/night-sky-queries.json`, 'utf8')),
  new Set(corpus.docs.map((d) => d.id)),
);

const chunks = corpus.docs.flatMap((d) => chunkStructure(d.id, d.text));
console.log(`${corpus.docs.length} docs -> ${chunks.length} chunks`);

const embedder = await pipeline('feature-extraction', EMBED_MODEL, { dtype: 'q8', revision: EMBED_REVISION });
async function embed(texts, instruct) {
  const input = instruct ? texts.map((t) => QUERY_INSTRUCTION + t) : texts;
  const out = await embedder(input, { pooling: 'mean', normalize: true });
  return out.tolist().map((row) => row.map((x) => Math.fround(x)));
}

console.log('embedding chunks...');
const chunkVectors = await embed(chunks.map((c) => c.text), false);
console.log('embedding queries...');
const queryVectors = await embed(evalSet.queries.map((q) => q.text), true);

// Rerank scores for the top-RERANK_TOP fused docs per query.
const { Retriever } = await import('../dist/src/core/pipeline.js');
const { PrecomputedEmbedder } = await import('../dist/src/core/precomputed.js');
const f32b64 = (arr) => Buffer.from(Float32Array.from(arr).buffer).toString('base64');
// Drive the real pipeline for fusion and best-chunk selection, so the
// precomputed rerank scores key onto exactly the passages the demo reranks.
const inFlight = {
  version: 1, corpus: corpus.name, embedModel: EMBED_MODEL, embedRevision: EMBED_REVISION,
  rerankModel: RERANK_MODEL, rerankRevision: RERANK_REVISION, queryInstruction: QUERY_INSTRUCTION,
  rerankTop: RERANK_TOP, dims: chunkVectors[0].length,
  chunkIds: chunks.map((c) => c.id), chunkVectors: chunkVectors.map(f32b64),
  queryIds: evalSet.queries.map((q) => q.id), queryVectors: queryVectors.map(f32b64),
  rerankScores: {},
};
const preEmbedder = new PrecomputedEmbedder(inFlight, corpus, evalSet);
const retriever = await Retriever.fromDocuments(corpus.name, corpus.docs, preEmbedder, { hnsw: null });

const tok = await AutoTokenizer.from_pretrained(RERANK_MODEL, { revision: RERANK_REVISION });
const cross = await AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL, { dtype: 'q8', revision: RERANK_REVISION });
async function rerankScore(query, passages) {
  const scores = [];
  for (const p of passages) {
    const inputs = tok(query, { text_pair: p, truncation: true, max_length: 512 });
    const { logits } = await cross(inputs);
    scores.push(Number(logits.data[0]));
  }
  return scores;
}

console.log('reranking...');
const rerankScores = {};
for (let qi = 0; qi < evalSet.queries.length; qi++) {
  const q = evalSet.queries[qi];
  const r = await retriever.query(q.text, { k: 10, fusion: 'rrf', rerankTop: RERANK_TOP }, preEmbedder);
  const head = r.head.map((h) => h.docId);
  const passages = r.head.map((h) => retriever.chunkById(h.chunkId).text);
  const scores = await rerankScore(q.text, passages);
  rerankScores[q.id] = Object.fromEntries(head.map((docId, i) => [docId, Math.fround(scores[i])]));
  if ((qi + 1) % 10 === 0) console.log(`  ${qi + 1}/${evalSet.queries.length}`);
}

const out = {
  version: 1,
  corpus: corpus.name,
  embedModel: EMBED_MODEL,
  embedRevision: EMBED_REVISION,
  rerankModel: RERANK_MODEL,
  rerankRevision: RERANK_REVISION,
  queryInstruction: QUERY_INSTRUCTION,
  rerankTop: RERANK_TOP,
  dims: chunkVectors[0].length,
  chunkIds: chunks.map((c) => c.id),
  chunkVectors: chunkVectors.map(f32b64),
  queryIds: evalSet.queries.map((q) => q.id),
  queryVectors: queryVectors.map(f32b64),
  rerankScores,
};
mkdirSync(`${root}data/precomputed`, { recursive: true });
const path = `${root}data/precomputed/night-sky.bge-small-en-v1.5.json`;
writeFileSync(path, JSON.stringify(out));
console.log(`wrote ${path} (${(JSON.stringify(out).length / 1024).toFixed(0)} KB, dims ${out.dims})`);

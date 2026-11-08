/**
 * Runs the labeled evaluation over the bundled corpus with the precomputed
 * model vectors. This is the same evaluation the live Eval tab runs; the
 * numbers it prints are the only numbers the README may quote.
 */
import { readFileSync } from 'node:fs';

const { validateCorpus } = await import('../dist/src/core/corpus.js');
const { validateEvalSet, runEvaluation } = await import('../dist/src/core/evaluate.js');
const { loadPrecomputed, PrecomputedEmbedder, PrecomputedReranker } = await import('../dist/src/core/precomputed.js');
const { Retriever } = await import('../dist/src/core/pipeline.js');

const root = new URL('..', import.meta.url).pathname;
const corpus = validateCorpus(JSON.parse(readFileSync(`${root}data/corpus/night-sky.json`, 'utf8')));
const evalSet = validateEvalSet(
  JSON.parse(readFileSync(`${root}data/eval/night-sky-queries.json`, 'utf8')),
  new Set(corpus.docs.map((d) => d.id)),
);
const pre = loadPrecomputed(JSON.parse(readFileSync(`${root}data/precomputed/night-sky.bge-small-en-v1.5.json`, 'utf8')));

const embedder = new PrecomputedEmbedder(pre, corpus, evalSet);
const retriever = await Retriever.fromDocuments(corpus.name, corpus.docs, embedder, { hnsw: { seed: 42 } });
const reranker = new PrecomputedReranker(pre, corpus, evalSet);

const K = 10;
const configs = [
  { name: 'bm25', run: async (q) => { const r = await retriever.query(q, { k: K, dense: null }); return { ranked: r.docs.map((d) => d.docId), latencyMs: r.latencyMs.total }; } },
  { name: 'dense (flat)', run: async (q) => { const r = await retriever.query(q, { k: K, lexical: false, dense: 'flat' }, embedder); return { ranked: r.docs.map((d) => d.docId), latencyMs: r.latencyMs.total }; } },
  { name: 'dense (hnsw ef=50)', run: async (q) => { const r = await retriever.query(q, { k: K, lexical: false, dense: 'hnsw', efSearch: 50 }, embedder); return { ranked: r.docs.map((d) => d.docId), latencyMs: r.latencyMs.total }; } },
  { name: 'hybrid rrf', run: async (q) => { const r = await retriever.query(q, { k: K, fusion: 'rrf' }, embedder); return { ranked: r.docs.map((d) => d.docId), latencyMs: r.latencyMs.total }; } },
  { name: 'hybrid linear a=0.5', run: async (q) => { const r = await retriever.query(q, { k: K, fusion: 'linear', alpha: 0.5 }, embedder); return { ranked: r.docs.map((d) => d.docId), latencyMs: r.latencyMs.total }; } },
  { name: 'hybrid rrf + rerank', run: async (q) => { const r = await retriever.query(q, { k: K, fusion: 'rrf', rerankTop: 25 }, embedder, reranker); return { ranked: r.docs.map((d) => d.docId), latencyMs: r.latencyMs.total }; } },
];

const reports = await runEvaluation(configs, evalSet);
const pct = (x) => (x * 100).toFixed(1).padStart(6);
console.log('config'.padEnd(24) + 'R@1   R@5   R@10  MRR   nDCG@10  p50ms');
for (const r of reports) {
  const a = r.aggregate;
  console.log(
    r.name.padEnd(24) +
    `${pct(a.recall[1])}${pct(a.recall[5])}${pct(a.recall[10])}${pct(a.mrr)}${pct(a.ndcg[10]).padStart(7)}  ${a.latencyP50.toFixed(2).padStart(6)}`,
  );
}

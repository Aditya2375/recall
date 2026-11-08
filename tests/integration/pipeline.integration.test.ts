/**
 * Integration test over the shipped data: the real 50-document corpus and
 * the real 40-query labeled eval set, indexed and ranked end to end with the
 * deterministic HashEmbedder (no model downloads - CI-safe). Asserts the
 * whole pipeline holds together: indexing, all six retrieval configs, the
 * eval runner, answer extraction, determinism, and import-limit behavior.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { HashEmbedder } from '../../src/core/embed.js';
import { Retriever } from '../../src/core/pipeline.js';
import { validateCorpus } from '../../src/core/corpus.js';
import { runEvaluation, validateEvalSet, type EvalConfig } from '../../src/core/evaluate.js';
import { composeAnswer } from '../../src/core/answer.js';
import type { Reranker } from '../../src/core/rerank.js';

const root = new URL('../../..', import.meta.url).pathname; // dist/tests/integration -> repo root
const corpus = validateCorpus(JSON.parse(readFileSync(`${root}data/corpus/night-sky.json`, 'utf8')));
const evalSet = validateEvalSet(
  JSON.parse(readFileSync(`${root}data/eval/night-sky-queries.json`, 'utf8')),
  new Set(corpus.docs.map((d) => d.id)),
);

/** Deterministic overlap reranker: real reranker plumbing, no neural net. */
const overlapReranker: Reranker = {
  id: 'overlap-test-reranker',
  async score(query: string, passages: string[]): Promise<number[]> {
    const terms = new Set(query.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length > 2));
    return passages.map((p) => {
      const words = p.toLowerCase().split(/[^a-z0-9]+/);
      return words.filter((w) => terms.has(w)).length;
    });
  },
};

test('shipped data validates and sizes match the README', () => {
  assert.equal(corpus.docs.length, 50);
  assert.equal(evalSet.queries.length, 40);
  const judgments = evalSet.queries.reduce((n, q) => n + Object.keys(q.judgments).length, 0);
  assert.equal(judgments, 71);
});

test('all six configs rank end to end with sane, in-range metrics', async () => {
  const embedder = new HashEmbedder(11, 384);
  const retriever = await Retriever.fromDocuments(corpus.name, corpus.docs, embedder, { hnsw: { seed: 42 } });
  const docIds = new Set(corpus.docs.map((d) => d.id));

  const configs: EvalConfig[] = [
    { name: 'bm25', run: async (q) => timed(() => retriever.query(q, { k: 10, dense: null }, embedder), 'bm25') },
    { name: 'dense (flat)', run: async (q) => timed(() => retriever.query(q, { k: 10, lexical: false, dense: 'flat' }, embedder), 'dense (flat)') },
    { name: 'dense (hnsw ef=50)', run: async (q) => timed(() => retriever.query(q, { k: 10, lexical: false, dense: 'hnsw', efSearch: 50 }, embedder), 'dense (hnsw)') },
    { name: 'hybrid rrf', run: async (q) => timed(() => retriever.query(q, { k: 10, fusion: 'rrf' }, embedder), 'hybrid rrf') },
    { name: 'hybrid linear a=0.5', run: async (q) => timed(() => retriever.query(q, { k: 10, fusion: 'linear', alpha: 0.5 }, embedder), 'hybrid linear') },
    { name: 'hybrid rrf + rerank', run: async (q) => timed(() => retriever.query(q, { k: 10, fusion: 'rrf', rerankTop: 25 }, embedder, overlapReranker), 'rrf+rerank') },
  ];
  const timed = async (fn: () => Promise<{ docs: { docId: string }[]; latencyMs: { total: number } }>, name = '') => {
    const r = await fn();
    const ranked = r.docs.map((d) => d.docId);
    // BM25 legitimately returns fewer than k docs when few match; dense fills to k.
    assert.ok(ranked.length >= 1 && ranked.length <= 10, `${name} returned ${ranked.length}`);
    for (const id of ranked) assert.ok(docIds.has(id), `${name} ranked unknown doc ${id}`);
    assert.equal(new Set(ranked).size, ranked.length, `${name} duplicated a doc`);
    return { ranked, latencyMs: r.latencyMs.total };
  };

  const reports = await runEvaluation(configs, evalSet);
  assert.equal(reports.length, 6);
  for (const report of reports) {
    assert.equal(report.rows.length, 40, report.name);
    const agg = report.aggregate;
    assert.ok(agg.mrr >= 0 && agg.mrr <= 1, `${report.name} mrr=${agg.mrr}`);
    assert.ok(agg.map >= 0 && agg.map <= 1, `${report.name} map=${agg.map}`);
    for (const k of [1, 5, 10]) {
      assert.ok(agg.recall[k]! >= 0 && agg.recall[k]! <= 1, `${report.name} recall@${k}`);
      assert.ok(agg.precision[k]! >= 0 && agg.precision[k]! <= 1, `${report.name} precision@${k}`);
      assert.ok(agg.ndcg[k]! >= 0 && agg.ndcg[k]! <= 1, `${report.name} ndcg@${k}`);
    }
  }
  // BM25 alone must beat pure chance by a wide margin on keyword queries.
  const bm25 = reports.find((r) => r.name === 'bm25')!;
  assert.ok(bm25.aggregate.recall[10]! > 0.5, `bm25 recall@10 suspiciously low: ${bm25.aggregate.recall[10]}`);
});

test('retrieval is deterministic across runs', async () => {
  const embedder = new HashEmbedder(11, 384);
  const a = await Retriever.fromDocuments(corpus.name, corpus.docs, embedder, { hnsw: { seed: 42 } });
  const b = await Retriever.fromDocuments(corpus.name, corpus.docs, embedder, { hnsw: { seed: 42 } });
  for (const q of ['Bortle scale', 'Olympus Mons height', 'Sagittarius A*']) {
    const ra = await a.query(q, { k: 10, fusion: 'rrf' }, embedder);
    const rb = await b.query(q, { k: 10, fusion: 'rrf' }, embedder);
    assert.deepEqual(ra.docs.map((d) => d.docId), rb.docs.map((d) => d.docId));
  }
});

test('composed answers quote the corpus verbatim', async () => {
  const embedder = new HashEmbedder(11, 384);
  const retriever = await Retriever.fromDocuments(corpus.name, corpus.docs, embedder);
  const allText = corpus.docs.map((d) => d.text).join('\n');
  for (const q of ['what causes the northern lights', 'how old is the universe']) {
    const r = await retriever.query(q, { k: 5, fusion: 'rrf' }, embedder);
    const inputs = r.head.slice(0, 3).map((h, i) => {
      const chunk = retriever.chunkById(h.chunkId);
      assert.ok(chunk, `head references unknown chunk ${h.chunkId}`);
      return { chunkId: h.chunkId, docId: h.docId, text: chunk.text, citation: i + 1 };
    });
    const answer = composeAnswer(q, inputs, retriever.bm25Index);
    assert.ok(answer.sentences.length > 0, `no answer sentences for "${q}"`);
    for (const s of answer.sentences) {
      assert.ok(allText.includes(s.text), `answer sentence not verbatim in corpus: "${s.text.slice(0, 60)}..."`);
    }
  }
});

test('import limits reject oversized corpora', () => {
  const tooManyDocs = { version: 1, name: 'x', docs: Array.from({ length: 501 }, (_, i) => ({ id: `d${i}`, title: 't', text: 'x' })) };
  assert.throws(() => validateCorpus(tooManyDocs), /too many documents/);
  const tooBig = { version: 1, name: 'x', docs: [{ id: 'd', title: 't', text: 'x'.repeat(2_000_001) }] };
  assert.throws(() => validateCorpus(tooBig), /2,000,000|2000000/);
});

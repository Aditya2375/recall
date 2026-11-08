import test from 'node:test';
import assert from 'node:assert/strict';
import { HashEmbedder } from '../../src/core/embed.js';
import { Retriever } from '../../src/core/pipeline.js';
import type { Reranker } from '../../src/core/rerank.js';

const docs = [
  { id: 'stars', title: 'Stars', text: '# Stars\n\nStars twinkle because air moves in the atmosphere. Starlight bends through turbulent layers.' },
  { id: 'planets', title: 'Planets', text: '# Planets\n\nPlanets shine by reflected sunlight. Planets usually do not twinkle because they are tiny disks, not points.' },
  { id: 'galaxies', title: 'Galaxies', text: '# Galaxies\n\nGalaxies are vast collections of stars. The Milky Way is our galaxy.' },
];

async function makeRetriever(hnsw: boolean) {
  return Retriever.fromDocuments('t', docs, new HashEmbedder(7, 128), { hnsw: hnsw ? { seed: 42 } : null });
}

test('bm25-only mode ranks the lexical match first', async () => {
  const r = await makeRetriever(false);
  const res = await r.query('twinkle atmosphere', { dense: null, k: 3 });
  assert.equal(res.docs[0]!.docId, 'stars');
  assert.equal(res.docs[0]!.ranks.bm25, 1);
  assert.equal(res.docs[0]!.scores.dense, undefined);
  assert.ok(res.latencyMs.total >= 0);
});

test('dense mode needs an embedder at query time', async () => {
  const r = await makeRetriever(false);
  await assert.rejects(() => r.query('stars', { lexical: false, dense: 'flat' }), /embedder/);
});

test('both channels off is rejected', async () => {
  const r = await makeRetriever(false);
  await assert.rejects(() => r.query('stars', { lexical: false, dense: null }), /both be off/);
});

test('hybrid mode fuses both lists and reports provenance', async () => {
  const r = await makeRetriever(true);
  const res = await r.query('twinkle', { k: 3, fusion: 'rrf' }, new HashEmbedder(7, 128));
  assert.ok(res.docs.length > 0);
  const first = res.docs[0]!;
  assert.ok(first.scores.fused > 0);
  assert.ok(first.rank === 1);
  assert.ok(res.mode.includes('rrf'));
  const chunk = r.chunkById(first.chunkId)!;
  assert.equal(chunk.docId, first.docId);
});

test('hnsw mode requires the index to have been built', async () => {
  const r = await makeRetriever(false);
  await assert.rejects(
    () => r.query('stars', { lexical: false, dense: 'hnsw' }, new HashEmbedder(7, 128)),
    /no hnsw index/,
  );
});

test('reranker reorders the top N and records rerank scores', async () => {
  const r = await makeRetriever(false);
  const fake: Reranker = {
    id: 'fake-reranker',
    score: (_q: string, passages: string[]) => Promise.resolve(passages.map((p) => (p.includes('Planets') ? 100 : 0))),
  };
  const res = await r.query('twinkle', { k: 3, rerankTop: 3 }, new HashEmbedder(7, 128), fake);
  assert.equal(res.docs[0]!.docId, 'planets');
  assert.equal(res.docs[0]!.scores.rerank, 100);
  assert.ok(res.latencyMs.rerank >= 0);
});

test('document ranking aggregates chunks with the best chunk per document', async () => {
  const many = [
    { id: 'multi', title: 'Multi', text: 'alpha beta gamma.\n\ndelta epsilon zeta.\n\nalpha alpha alpha beta.' },
    { id: 'other', title: 'Other', text: 'completely unrelated words about nothing in particular.' },
  ];
  const r = await Retriever.fromDocuments('t', many, new HashEmbedder(7, 64), { hnsw: null });
  const res = await r.query('alpha', { dense: null, k: 5 });
  const multi = res.docs.find((d) => d.docId === 'multi')!;
  assert.ok(multi.chunkId.startsWith('multi#c'));
  // one entry per document, never one per chunk
  assert.equal(res.docs.filter((d) => d.docId === 'multi').length, 1);
});

test('empty corpus is rejected at build time', async () => {
  await assert.rejects(() => Retriever.fromDocuments('t', [], new HashEmbedder()), /no documents/);
});

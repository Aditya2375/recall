import test from 'node:test';
import assert from 'node:assert/strict';
import { HnswIndex } from '../../src/core/hnsw.js';
import { FlatIndex } from '../../src/core/vector.js';
import { mulberry32 } from '../../src/core/rng.js';

function randomVectors(count: number, dims: number, seed: number): Float32Array[] {
  const rand = mulberry32(seed);
  return Array.from({ length: count }, () => {
    const v = new Float32Array(dims);
    for (let i = 0; i < dims; i++) v[i] = rand() * 2 - 1;
    return v;
  });
}

test('hnsw recall@10 vs exact search is high on random vectors', () => {
  const dims = 64;
  const vectors = randomVectors(2000, dims, 1234);
  const queries = randomVectors(100, dims, 9876);
  const flat = new FlatIndex(dims);
  const hnsw = new HnswIndex(dims, { M: 16, efConstruction: 200, seed: 42 });
  vectors.forEach((v, i) => {
    flat.add(`v${i}`, v);
    hnsw.add(`v${i}`, v);
  });
  const recallAt = (ef: number): number => {
    let hits = 0;
    for (const q of queries) {
      const exact = new Set(flat.search(q, 10).map((h) => h.id));
      const approx = hnsw.search(q, 10, ef);
      assert.equal(approx.length, 10);
      for (const h of approx) if (exact.has(h.id)) hits += 1;
    }
    return hits / (queries.length * 10);
  };
  const ef50 = recallAt(50);
  const ef100 = recallAt(100);
  assert.ok(ef50 >= 0.9, `recall@10 ef=50 = ${ef50}`);
  assert.ok(ef100 >= 0.98, `recall@10 ef=100 = ${ef100}`);
  assert.ok(ef100 > ef50, `recall should rise with ef: ${ef50} -> ${ef100}`);
});

test('hnsw build is deterministic for a fixed seed', () => {
  const vectors = randomVectors(300, 32, 55);
  const build = () => {
    const idx = new HnswIndex(32, { seed: 9 });
    vectors.forEach((v, i) => idx.add(`v${i}`, v));
    return idx;
  };
  const a = build();
  const z = build();
  const q = randomVectors(1, 32, 77)[0]!;
  assert.deepEqual(a.search(q, 10), z.search(q, 10));
  assert.deepEqual(a.stats(), z.stats());
});

test('hnsw handles edge cases: empty, single node, k > n, duplicate vectors', () => {
  const idx = new HnswIndex(4, { seed: 1 });
  assert.deepEqual(idx.search(new Float32Array([1, 0, 0, 0]), 5), []);
  idx.add('only', new Float32Array([1, 0, 0, 0]));
  assert.equal(idx.search(new Float32Array([1, 0, 0, 0]), 5).length, 1);
  idx.add('copy', new Float32Array([1, 0, 0, 0]));
  idx.add('other', new Float32Array([0, 1, 0, 0]));
  const hits = idx.search(new Float32Array([1, 0, 0, 0]), 10);
  assert.equal(hits.length, 3);
  assert.ok(hits[0]!.score >= hits[1]!.score);
});

test('hnsw rejects duplicates and wrong dims', () => {
  const idx = new HnswIndex(2, { seed: 1 });
  idx.add('a', new Float32Array([1, 0]));
  assert.throws(() => idx.add('a', new Float32Array([1, 0])), /duplicate/);
  assert.throws(() => idx.add('b', new Float32Array([1, 0, 0])), /dims/);
  assert.throws(() => new HnswIndex(2, { M: 1 }), /M must be/);
});

test('hnsw results are sorted by score descending', () => {
  const vectors = randomVectors(200, 16, 31);
  const idx = new HnswIndex(16, { seed: 2 });
  vectors.forEach((v, i) => idx.add(`v${i}`, v));
  const hits = idx.search(randomVectors(1, 16, 33)[0]!, 20);
  for (let i = 1; i < hits.length; i++) {
    assert.ok(hits[i - 1]!.score >= hits[i]!.score);
  }
});

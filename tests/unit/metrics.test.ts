import test from 'node:test';
import assert from 'node:assert/strict';
import {
  aggregate, averagePrecision, measureQuery, ndcgAtK, precisionAtK, recallAtK, reciprocalRank,
} from '../../src/core/metrics.js';

const J = new Map([['a', 2], ['b', 1], ['c', 0]]);

test('precision@k counts relevant in the top k', () => {
  assert.equal(precisionAtK(['a', 'c', 'b'], J, 3), 2 / 3);
  assert.equal(precisionAtK(['c'], J, 1), 0);
  assert.equal(precisionAtK(['a', 'b'], J, 5), 2 / 5);
  assert.equal(precisionAtK([], J, 0), 0);
});

test('recall@k divides by total relevant', () => {
  assert.equal(recallAtK(['a'], J, 1), 1 / 2);
  assert.equal(recallAtK(['a', 'b'], J, 2), 1);
  assert.equal(recallAtK(['c', 'x'], J, 2), 0);
});

test('reciprocal rank uses the first relevant position', () => {
  assert.equal(reciprocalRank(['c', 'a'], J), 1 / 2);
  assert.equal(reciprocalRank(['a'], J), 1);
  assert.equal(reciprocalRank(['c'], J), 0);
});

test('average precision matches a hand-computed case', () => {
  // ranked: a(rel) b(rel) c(not) ; relevant = {a,b}
  // AP = (1/1 + 2/2) / 2 = 1
  assert.equal(averagePrecision(['a', 'b', 'c'], J), 1);
  // ranked: c(not) a(rel) b(rel) ; AP = (1/2 + 2/3) / 2 = 7/12
  const ap = averagePrecision(['c', 'a', 'b'], J);
  assert.ok(Math.abs(ap - 7 / 12) < 1e-12);
});

test('ndcg@k matches a hand-computed case', () => {
  // ranked [b(1), a(2)] k=2: DCG = 1/log2(2) + 3/log2(3); ideal [2,1]: 3/1 + 1/log2(3)
  const dcg = (g: number[]) => g.reduce((s, x, i) => s + (2 ** x - 1) / Math.log2(i + 2), 0);
  const expected = dcg([1, 2]) / dcg([2, 1]);
  assert.ok(Math.abs(ndcgAtK(['b', 'a'], J, 2) - expected) < 1e-12);
  assert.equal(ndcgAtK(['a', 'b'], J, 2), 1);
});

test('measureQuery fills every metric field', () => {
  const m = measureQuery('q', ['a', 'c', 'b'], J, 12);
  assert.equal(m.latencyMs, 12);
  for (const k of [1, 5, 10]) {
    assert.ok(Number.isFinite(m.precision[k]!));
    assert.ok(Number.isFinite(m.recall[k]!));
    assert.ok(Number.isFinite(m.ndcg[k]!));
  }
});

test('aggregate skips queries with no relevant document', () => {
  const rows = [
    measureQuery('q1', ['a', 'b'], J, 10),
    measureQuery('q2', ['x'], new Map([['x', 0]]), 10),
  ];
  const byQuery = new Map([['q1', J], ['q2', new Map([['x', 0]])]]);
  const agg = aggregate(rows, byQuery);
  assert.equal(agg.queries, 1);
  assert.equal(agg.skipped, 1);
  assert.equal(agg.mrr, 1);
  assert.equal(agg.map, 1);
});

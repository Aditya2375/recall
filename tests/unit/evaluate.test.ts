import test from 'node:test';
import assert from 'node:assert/strict';
import { runEvaluation, validateEvalSet, type EvalSet } from '../../src/core/evaluate.js';

const docIds = new Set(['a', 'b', 'c']);

const evalSet: EvalSet = {
  version: 1,
  corpus: 't',
  queries: [
    { id: 'q1', text: 'first', judgments: { a: 2, b: 1 } },
    { id: 'q2', text: 'second', judgments: { c: 1 } },
  ],
};

test('validateEvalSet accepts a good set and checks doc ids', () => {
  const parsed = validateEvalSet(structuredClone(evalSet), docIds);
  assert.equal(parsed.queries.length, 2);
  assert.throws(() => validateEvalSet({ version: 1, corpus: 't', queries: [
    { id: 'q', text: 'x', judgments: { nope: 1 } },
  ] }, docIds), /unknown document/);
  assert.throws(() => validateEvalSet({ version: 1, corpus: 't', queries: [
    { id: 'q', text: 'x', judgments: { a: 3 } },
  ] }, docIds), /0, 1 or 2/);
});

test('runEvaluation scores a perfect config perfectly', async () => {
  const reports = await runEvaluation([
    { name: 'perfect', run: (q: string) => Promise.resolve({ ranked: q === 'first' ? ['a', 'b', 'c'] : ['c', 'a', 'b'], latencyMs: 1 }) },
  ], evalSet);
  const agg = reports[0]!.aggregate;
  assert.equal(agg.queries, 2);
  assert.equal(agg.mrr, 1);
  assert.equal(agg.recall[10], 1);
  assert.equal(agg.ndcg[10], 1);
});

test('runEvaluation scores a bad config worse than a good one', async () => {
  const reports = await runEvaluation([
    { name: 'good', run: (q: string) => Promise.resolve({ ranked: q === 'first' ? ['a', 'b', 'c'] : ['c', 'a', 'b'], latencyMs: 1 }) },
    { name: 'bad', run: (q: string) => Promise.resolve({ ranked: q === 'first' ? ['c', 'b', 'a'] : ['a', 'b', 'c'], latencyMs: 1 }) },
  ], evalSet);
  const good = reports.find((r) => r.name === 'good')!.aggregate;
  const bad = reports.find((r) => r.name === 'bad')!.aggregate;
  assert.ok(good.mrr > bad.mrr);
  assert.ok(good.ndcg[10]! > bad.ndcg[10]!);
});

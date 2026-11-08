import test from 'node:test';
import assert from 'node:assert/strict';
import { linear, rrf, sortScored } from '../../src/core/fusion.js';

test('rrf sums reciprocal ranks across lists with 1-based ranks', () => {
  const fused = rrf([['a', 'b', 'c'], ['b', 'a']], 60);
  // a: 1/61 + 1/62 ; b: 1/62 + 1/61 ; c: 1/63
  assert.ok(Math.abs(fused.get('a')! - (1 / 61 + 1 / 62)) < 1e-12);
  assert.ok(Math.abs(fused.get('c')! - 1 / 63) < 1e-12);
});

test('rrf output order is deterministic: score desc, then id asc', () => {
  const fused = rrf([['b', 'a'], ['a', 'b']], 60); // identical scores
  assert.deepEqual([...fused.keys()], ['a', 'b']);
});

test('rrf ignores duplicate ids within one list', () => {
  const fused = rrf([['a', 'a', 'b']], 60);
  assert.ok(Math.abs(fused.get('a')! - 1 / 61) < 1e-12);
  assert.ok(Math.abs(fused.get('b')! - 1 / 62) < 1e-12);
});

test('rrf validates k', () => {
  assert.throws(() => rrf([['a']], 0), /k must be/);
});

test('linear fusion min-max normalizes each list before weighting', () => {
  const fused = linear([
    { ids: ['a', 'b'], scores: new Map([['a', 10], ['b', 0]]), weight: 1 },
    { ids: ['b', 'c'], scores: new Map([['b', 5], ['c', 5]]), weight: 0.5 },
  ]);
  // list 1: a -> 1, b -> 0. list 2: equal scores -> both 1.
  assert.ok(Math.abs(fused.get('a')! - 1) < 1e-12);
  assert.ok(Math.abs(fused.get('b')! - 0.5) < 1e-12);
  assert.ok(Math.abs(fused.get('c')! - 0.5) < 1e-12);
});

test('linear fusion rejects negative weights', () => {
  assert.throws(() => linear([{ ids: ['a'], scores: new Map([['a', 1]]), weight: -0.1 }]), /weights/);
});

test('sortScored breaks ties by id ascending', () => {
  const sorted = sortScored(new Map([['z', 1], ['a', 1], ['m', 2]]));
  assert.deepEqual(sorted.map(([id]) => id), ['m', 'a', 'z']);
});

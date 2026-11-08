import test from 'node:test';
import assert from 'node:assert/strict';
import { dot, FlatIndex, normalize } from '../../src/core/vector.js';

test('normalize produces unit length; zero vector stays zero', () => {
  const v = normalize(new Float32Array([3, 4]));
  assert.ok(Math.abs(dot(v, v) - 1) < 1e-5);
  const z = normalize(new Float32Array([0, 0]));
  assert.equal(dot(z, z), 0);
});

test('flat index returns cosine-sorted hits with deterministic ties', () => {
  const idx = new FlatIndex(2);
  idx.add('x', new Float32Array([1, 0]));
  idx.add('y', new Float32Array([0, 1]));
  idx.add('z', new Float32Array([1, 1]));
  const hits = idx.search(new Float32Array([1, 0]), 3);
  assert.equal(hits[0]!.id, 'x');
  assert.ok(Math.abs(hits[0]!.score - 1) < 1e-5);
  assert.ok(hits[1]!.score > hits[2]!.score);
});

test('flat index rejects duplicates and wrong dims', () => {
  const idx = new FlatIndex(2);
  idx.add('x', new Float32Array([1, 0]));
  assert.throws(() => idx.add('x', new Float32Array([1, 0])), /duplicate/);
  assert.throws(() => idx.add('y', new Float32Array([1, 0, 0])), /dims/);
});

test('flat index k larger than the corpus returns everything', () => {
  const idx = new FlatIndex(2);
  idx.add('x', new Float32Array([1, 0]));
  assert.equal(idx.search(new Float32Array([1, 0]), 99).length, 1);
  assert.equal(idx.search(new Float32Array([1, 0]), 0).length, 0);
});

import test from 'node:test';
import assert from 'node:assert/strict';
import { BM25Index } from '../../src/core/bm25.js';

const TOL = 1e-9;

function approx(actual: number, expected: number, tol = TOL): void {
  assert.ok(Math.abs(actual - expected) <= tol, `${actual} != ${expected}`);
}

test('hand-computed score matches the formula', () => {
  // Two docs; query term appears in one. Check idf and score arithmetic by hand.
  const idx = new BM25Index({ k1: 1.2, b: 0.75 });
  idx.addDocument('a', 'apple apple banana'); // dl 3, tf(apple)=2
  idx.addDocument('z', 'cherry date elderberry fig grape'); // dl 5
  // N=2, df(apple)=1 -> idf = ln(1 + (2-1+0.5)/(1+0.5)) = ln(2)
  const idf = Math.log(2);
  approx(idx.idf('apple'), idf);
  const avgdl = 4;
  const tf = 2;
  const denom = tf + 1.2 * (1 - 0.75 + (0.75 * 3) / avgdl);
  const expected = (idf * tf * 2.2) / denom;
  const scores = idx.score('apple');
  approx(scores.get('a')!, expected);
  assert.equal(scores.get('z'), undefined);
});

test('idf is non-negative and shrinks as a term gets more common', () => {
  const idx = new BM25Index();
  idx.addDocument('a', 'common rare');
  idx.addDocument('b', 'common');
  idx.addDocument('c', 'common');
  assert.ok(idx.idf('common') < idx.idf('rare'));
  assert.ok(idx.idf('common') >= 0);
  assert.equal(idx.idf('absent'), 0);
});

test('longer documents score lower for the same tf (length normalization)', () => {
  const idx = new BM25Index({ k1: 1.2, b: 0.75 });
  idx.addDocument('short', 'planet');
  idx.addDocument('long', 'planet ' + 'filler '.repeat(50));
  const scores = idx.score('planet');
  assert.ok(scores.get('short')! > scores.get('long')!);
});

test('b=0 disables length normalization', () => {
  const idx = new BM25Index({ k1: 1.2, b: 0 });
  idx.addDocument('short', 'planet');
  idx.addDocument('long', 'planet ' + 'filler '.repeat(50));
  const scores = idx.score('planet');
  approx(scores.get('short')!, scores.get('long')!);
});

test('duplicate document ids are rejected', () => {
  const idx = new BM25Index();
  idx.addDocument('a', 'one');
  assert.throws(() => idx.addDocument('a', 'two'), /duplicate/);
});

test('empty query and unknown terms return no scores', () => {
  const idx = new BM25Index();
  idx.addDocument('a', 'one two');
  assert.equal(idx.score('').size, 0);
  assert.equal(idx.score('zzzz').size, 0);
  assert.equal(new BM25Index().score('one').size, 0);
});

test('explain decomposes the score into per-term contributions that sum to the total', () => {
  const idx = new BM25Index();
  idx.addDocument('a', 'star light star bright night star');
  idx.addDocument('b', 'galaxy far away');
  const ex = idx.explain('star night', 'a')!;
  const sum = ex.terms.reduce((s, t) => s + t.contribution, 0);
  approx(ex.score, sum);
  approx(ex.score, idx.score('star night').get('a')!);
  const star = ex.terms.find((t) => t.term === 'star')!;
  assert.equal(star.tf, 3);
  const night = ex.terms.find((t) => t.term === 'night')!;
  assert.equal(night.tf, 1);
});

test('explain returns null for unknown documents', () => {
  const idx = new BM25Index();
  idx.addDocument('a', 'one');
  assert.equal(idx.explain('one', 'nope'), null);
});

test('stopword option drops listed terms from indexing and queries', () => {
  const idx = new BM25Index({ stopwords: true });
  idx.addDocument('a', 'the stars and the sky');
  assert.equal(idx.score('the').size, 0);
  assert.ok(idx.score('sky').get('a')! > 0);
});

test('invalid parameters throw', () => {
  assert.throws(() => new BM25Index({ k1: -1 }), /k1/);
  assert.throws(() => new BM25Index({ b: 1.5 }), /b must be/);
});

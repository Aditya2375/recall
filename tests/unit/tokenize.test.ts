import test from 'node:test';
import assert from 'node:assert/strict';
import { removeStopwords, termFreq, tokenize, tokenizeWithSpans, STOPWORDS } from '../../src/core/tokenize.js';

test('tokenize lowercases and splits on non-word characters', () => {
  assert.deepEqual(tokenize('The Night-Sky, 2026!'), ['the', 'night', 'sky', '2026']);
});

test('tokenize handles unicode letters', () => {
  assert.deepEqual(tokenize('Café naïve Über'), ['café', 'naïve', 'über']);
});

test('empty and separator-only input yield no tokens', () => {
  assert.deepEqual(tokenize(''), []);
  assert.deepEqual(tokenize('  --- ... '), []);
});

test('spans map back to the original text exactly', () => {
  const text = 'Alpha, BETA gamma';
  for (const t of tokenizeWithSpans(text)) {
    assert.equal(text.slice(t.start, t.end).toLowerCase(), t.text);
  }
});

test('stopword removal only drops listed words', () => {
  const tokens = tokenize('the quick brown fox and the stars');
  const kept = removeStopwords(tokens);
  assert.deepEqual(kept, ['quick', 'brown', 'fox', 'stars']);
  for (const w of kept) assert.ok(!STOPWORDS.has(w));
});

test('termFreq counts occurrences', () => {
  const tf = termFreq(['a', 'b', 'a', 'c', 'a', 'b']);
  assert.equal(tf.get('a'), 3);
  assert.equal(tf.get('b'), 2);
  assert.equal(tf.get('c'), 1);
  assert.equal(tf.get('zz'), undefined);
});

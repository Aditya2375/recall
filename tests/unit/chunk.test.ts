import test from 'node:test';
import assert from 'node:assert/strict';
import { chunkFixed, chunkStructure } from '../../src/core/chunk.js';
import { tokenize } from '../../src/core/tokenize.js';

const lorem = Array.from({ length: 100 }, (_, i) => `word${i}`).join(' ');

test('fixed chunker respects the token budget and overlap', () => {
  const chunks = chunkFixed('d', lorem, { maxTokens: 20, overlap: 5 });
  assert.equal(chunks.length, Math.ceil((100 - 20) / 15) + 1);
  for (const c of chunks) {
    assert.ok(tokenize(c.text).length <= 20);
  }
  // consecutive chunks share exactly `overlap` tokens
  const first = tokenize(chunks[0]!.text);
  const second = tokenize(chunks[1]!.text);
  assert.deepEqual(first.slice(-5), second.slice(0, 5));
});

test('fixed chunker spans map back to the source text', () => {
  const chunks = chunkFixed('d', lorem, { maxTokens: 10, overlap: 0 });
  for (const c of chunks) {
    assert.equal(lorem.slice(c.start, c.end), c.text);
  }
  // full coverage with no overlap
  assert.equal(chunks.map((c) => c.text).join(' '), lorem);
});

test('fixed chunker: empty text yields no chunks', () => {
  assert.deepEqual(chunkFixed('d', ''), []);
  assert.deepEqual(chunkFixed('d', ' \n\t '), []);
});

test('fixed chunker validates options', () => {
  assert.throws(() => chunkFixed('d', lorem, { maxTokens: 10, overlap: 10 }), /overlap/);
  assert.throws(() => chunkFixed('d', lorem, { maxTokens: 0 }), /maxTokens/);
});

test('chunk ids are stable and sequential', () => {
  const chunks = chunkFixed('doc', lorem, { maxTokens: 30, overlap: 0 });
  assert.deepEqual(chunks.map((c) => c.id), ['doc#c0', 'doc#c1', 'doc#c2', 'doc#c3']);
  assert.deepEqual(chunks.map((c) => c.index), [0, 1, 2, 3]);
});

test('structure chunker keeps heading context with each chunk', () => {
  const text = [
    '# Stars',
    'Stars are luminous plasma balls.',
    '',
    '## Red giants',
    'Red giants are swollen old stars. Betelgeuse is one.',
    '',
    '# Planets',
    'Planets orbit stars. Earth orbits the Sun.',
  ].join('\n');
  const chunks = chunkStructure('d', text, { maxTokens: 16, overlap: 4 });
  assert.ok(chunks.length >= 2);
  const redGiant = chunks.find((c) => c.text.includes('Betelgeuse'))!;
  assert.deepEqual(redGiant.headings, ['Stars', 'Red giants']);
  const planets = chunks.find((c) => c.text.includes('Earth'))!;
  assert.deepEqual(planets.headings, ['Planets']);
});

test('structure chunker splits oversized blocks with fixed windows', () => {
  const big = '# Big\n\n' + Array.from({ length: 500 }, (_, i) => `tok${i}`).join(' ');
  const chunks = chunkStructure('d', big, { maxTokens: 100, overlap: 10 });
  assert.ok(chunks.length >= 5);
  for (const c of chunks) {
    assert.ok(tokenize(c.text).length <= 100);
    assert.deepEqual(c.headings, ['Big']);
  }
});

test('structure chunker spans map back to the source text', () => {
  const text = '# A\n\npara one words here.\n\n# B\n\npara two more words.';
  for (const c of chunkStructure('d', text)) {
    assert.equal(text.slice(c.start, c.end), c.text);
  }
});

test('structure chunker handles text without headings or blank lines', () => {
  const chunks = chunkStructure('d', 'just one paragraph of plain words');
  assert.equal(chunks.length, 1);
  assert.deepEqual(chunks[0]!.headings, []);
});

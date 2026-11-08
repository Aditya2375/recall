import test from 'node:test';
import assert from 'node:assert/strict';
import { BM25Index } from '../../src/core/bm25.js';
import { composeAnswer, type AnswerInput } from '../../src/core/answer.js';

const bm25 = new BM25Index();
bm25.addDocument('c0', 'Stars twinkle because air moves. The atmosphere refracts starlight.');
bm25.addDocument('c1', 'Planets do not twinkle much. They appear as tiny disks.');

const chunks: AnswerInput[] = [
  { chunkId: 'c0', docId: 'd0', text: 'Stars twinkle because air moves. The atmosphere refracts starlight.', citation: 1 },
  { chunkId: 'c1', docId: 'd1', text: 'Planets do not twinkle much. They appear as tiny disks.', citation: 2 },
];

test('every answer sentence is a verbatim substring of its source chunk', () => {
  const answer = composeAnswer('why do stars twinkle', chunks, bm25);
  assert.ok(answer.sentences.length > 0);
  for (const s of answer.sentences) {
    const source = chunks.find((c) => c.chunkId === s.chunkId)!;
    assert.ok(source.text.includes(s.text), `"${s.text}" not found verbatim in chunk`);
  }
});

test('sentences carry citation numbers matching the input ordering', () => {
  const answer = composeAnswer('stars twinkle atmosphere', chunks, bm25);
  for (const s of answer.sentences) {
    const source = chunks.find((c) => c.chunkId === s.chunkId)!;
    assert.equal(s.citation, source.citation);
  }
});

test('no overlap means no fabricated answer', () => {
  const answer = composeAnswer('quantum entanglement', chunks, bm25);
  assert.equal(answer.sentences.length, 0);
});

test('redundant near-duplicate sentences are dropped', () => {
  const dup: AnswerInput[] = [
    { chunkId: 'x0', docId: 'd', text: 'Stars twinkle due to moving air.', citation: 1 },
    { chunkId: 'x1', docId: 'd', text: 'Stars twinkle due to moving air currents. Comets are icy.', citation: 2 },
  ];
  const bm = new BM25Index();
  bm.addDocument('x0', dup[0]!.text);
  bm.addDocument('x1', dup[1]!.text);
  const answer = composeAnswer('stars twinkle air', dup, bm);
  const twinkles = answer.sentences.filter((s) => s.text.toLowerCase().includes('twinkle'));
  assert.equal(twinkles.length, 1);
});

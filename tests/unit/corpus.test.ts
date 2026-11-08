import test from 'node:test';
import assert from 'node:assert/strict';
import { IMPORT_LIMITS, parseCorpusJson, validateCorpus } from '../../src/core/corpus.js';

const good = {
  version: 1,
  name: 'demo',
  docs: [
    { id: 'a', title: 'Alpha', text: 'some words' },
    { id: 'b-2', title: 'Beta', text: 'more words' },
  ],
};

test('a well-formed corpus validates', () => {
  const c = validateCorpus(structuredClone(good));
  assert.equal(c.docs.length, 2);
  assert.equal(c.name, 'demo');
});

test('rejects non-objects, wrong version, empty docs', () => {
  assert.throws(() => validateCorpus(null), /not an object/);
  assert.throws(() => validateCorpus({ version: 2, name: 'x', docs: [{}] }), /version/);
  assert.throws(() => validateCorpus({ version: 1, name: 'x', docs: [] }), /empty/);
});

test('rejects duplicate ids, bad ids, missing fields', () => {
  assert.throws(() => validateCorpus({ version: 1, name: 'x', docs: [
    { id: 'a', title: 't', text: 'x' }, { id: 'a', title: 't', text: 'y' },
  ] }), /duplicate/);
  assert.throws(() => validateCorpus({ version: 1, name: 'x', docs: [
    { id: 'bad id!', title: 't', text: 'x' },
  ] }), /invalid characters/);
  assert.throws(() => validateCorpus({ version: 1, name: 'x', docs: [
    { id: 'a', title: '', text: 'x' },
  ] }), /title/);
});

test('enforces size limits', () => {
  assert.throws(() => validateCorpus({ version: 1, name: 'x', docs: [
    { id: 'a', title: 't', text: 'x'.repeat(IMPORT_LIMITS.maxTotalChars + 1) },
  ] }), /characters total/);
});

test('parseCorpusJson reports invalid JSON distinctly', () => {
  assert.throws(() => parseCorpusJson('{not json'), /invalid JSON/);
  assert.equal(parseCorpusJson(JSON.stringify(good)).docs.length, 2);
});

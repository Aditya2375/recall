/**
 * Corpus model and import validation. Import is deliberately strict: the
 * demo indexes whatever it is given, so size, shape and id rules are
 * enforced up front and every rejection says why.
 */

export interface CorpusDoc {
  id: string;
  title: string;
  text: string;
}

export interface Corpus {
  version: 1;
  name: string;
  docs: CorpusDoc[];
}

export const IMPORT_LIMITS = {
  maxDocs: 500,
  maxTotalChars: 2_000_000,
  maxTitleChars: 200,
  maxIdChars: 120,
  maxNameChars: 80,
} as const;

const ID_PATTERN = /^[\w\-:.]+$/;

function fail(reason: string): never {
  throw new Error(`corpus: ${reason}`);
}

export function validateCorpus(input: unknown): Corpus {
  if (typeof input !== 'object' || input === null) fail('not an object');
  const raw = input as Record<string, unknown>;
  if (raw['version'] !== 1) fail('version must be 1');
  if (typeof raw['name'] !== 'string' || raw['name'].trim() === '') fail('name must be a non-empty string');
  const name = raw['name'].trim();
  if (name.length > IMPORT_LIMITS.maxNameChars) fail(`name exceeds ${IMPORT_LIMITS.maxNameChars} characters`);
  if (!Array.isArray(raw['docs'])) fail('docs must be an array');
  const docsRaw = raw['docs'] as unknown[];
  if (docsRaw.length === 0) fail('docs must not be empty');
  if (docsRaw.length > IMPORT_LIMITS.maxDocs) fail(`too many documents (max ${IMPORT_LIMITS.maxDocs})`);

  const docs: CorpusDoc[] = [];
  const seen = new Set<string>();
  let totalChars = 0;
  for (let i = 0; i < docsRaw.length; i++) {
    const d = docsRaw[i];
    if (typeof d !== 'object' || d === null) fail(`docs[${i}] is not an object`);
    const r = d as Record<string, unknown>;
    if (typeof r['id'] !== 'string' || r['id'] === '') fail(`docs[${i}].id must be a non-empty string`);
    const id = r['id'];
    if (id.length > IMPORT_LIMITS.maxIdChars) fail(`docs[${i}].id exceeds ${IMPORT_LIMITS.maxIdChars} characters`);
    if (!ID_PATTERN.test(id)) fail(`docs[${i}].id "${id}" has invalid characters (allowed: letters, digits, _ - : .)`);
    if (seen.has(id)) fail(`duplicate document id "${id}"`);
    seen.add(id);
    if (typeof r['title'] !== 'string' || r['title'].trim() === '') fail(`docs[${i}].title must be a non-empty string`);
    const title = r['title'].trim();
    if (title.length > IMPORT_LIMITS.maxTitleChars) fail(`docs[${i}].title exceeds ${IMPORT_LIMITS.maxTitleChars} characters`);
    if (typeof r['text'] !== 'string' || r['text'].trim() === '') fail(`docs[${i}].text must be a non-empty string`);
    const text = r['text'];
    totalChars += text.length;
    if (totalChars > IMPORT_LIMITS.maxTotalChars) fail(`corpus exceeds ${IMPORT_LIMITS.maxTotalChars} characters total`);
    docs.push({ id, title, text });
  }
  return { version: 1, name, docs };
}

export function parseCorpusJson(raw: string): Corpus {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    fail('invalid JSON');
  }
  return validateCorpus(parsed);
}

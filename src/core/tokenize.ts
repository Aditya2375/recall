/**
 * Tokenizer: Unicode-aware, lowercased word tokens with character spans.
 * A token is any run of Unicode letters or numbers; everything else is a
 * separator. Case folding is plain toLowerCase, which is locale-independent
 * and deterministic.
 */

const WORD = /[\p{L}\p{N}]+/gu;

export interface Token {
  text: string;
  start: number;
  end: number;
}

export function tokenizeWithSpans(text: string): Token[] {
  const out: Token[] = [];
  WORD.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = WORD.exec(text)) !== null) {
    const raw = m[0];
    out.push({ text: raw.toLowerCase(), start: m.index, end: m.index + raw.length });
  }
  return out;
}

export function tokenize(text: string): string[] {
  return tokenizeWithSpans(text).map((t) => t.text);
}

/**
 * Common English stopwords. Keeping or dropping them is a visible ablation
 * in the demo: BM25's IDF already suppresses terms that occur everywhere,
 * so the effect is small - which is exactly the point of showing it.
 */
export const STOPWORDS: ReadonlySet<string> = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by',
  'can', 'could', 'did', 'do', 'does', 'for', 'from', 'had', 'has',
  'have', 'he', 'her', 'here', 'hers', 'him', 'his', 'how', 'i', 'if',
  'in', 'into', 'is', 'it', 'its', 'just', 'like', 'me', 'more', 'most',
  'my', 'no', 'not', 'now', 'of', 'on', 'one', 'or', 'our', 'ours',
  'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such',
  'than', 'that', 'the', 'their', 'theirs', 'them', 'then', 'there',
  'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under',
  'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where',
  'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would',
  'you', 'your', 'yours', 'about', 'above', 'after', 'again', 'against',
  'all', 'also', 'am', 'any', 'because', 'before', 'being', 'below',
  'between', 'both', 'came', 'come', 'each', 'few', 'further', 'get',
  'got', 'here', 'how', 'itself', 'many', 'may', 'might', 'much',
  'must', 'nor', 'off', 'old', 'once', 'only', 'other', 'others',
  'otherwise', 'per', 'put', 'rather', 'said', 'say', 'says', 'since',
  'still', 'take', 'things', 'thing', 'way', 'well', 'went', 'yet',
]);

export function removeStopwords(tokens: string[]): string[] {
  return tokens.filter((t) => !STOPWORDS.has(t));
}

export function termFreq(tokens: string[]): Map<string, number> {
  const tf = new Map<string, number>();
  for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
  return tf;
}

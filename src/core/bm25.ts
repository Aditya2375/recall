/**
 * BM25 (Okapi) lexical ranking with Lucene-style non-negative IDF:
 *
 *   idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))
 *   score(d, q) = sum_t idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
 *
 * The index exposes full per-term score decomposition so the UI can show
 * exactly why a document ranked where it did.
 */

import { removeStopwords, termFreq, tokenize } from './tokenize.js';

export interface BM25Options {
  k1?: number;
  b?: number;
  stopwords?: boolean;
}

export interface TermContribution {
  term: string;
  idf: number;
  tf: number;
  docLen: number;
  avgDocLen: number;
  contribution: number;
}

export interface BM25Explanation {
  docId: string;
  score: number;
  terms: TermContribution[];
}

export class BM25Index {
  readonly k1: number;
  readonly b: number;
  readonly stopwords: boolean;

  private readonly ids: string[] = [];
  private readonly posById = new Map<string, number>();
  private readonly tfByDoc: Array<Map<string, number>> = [];
  private readonly docLen: number[] = [];
  private readonly df = new Map<string, number>();
  private totalLen = 0;

  constructor(options: BM25Options = {}) {
    this.k1 = options.k1 ?? 1.2;
    this.b = options.b ?? 0.75;
    this.stopwords = options.stopwords ?? false;
    if (!(this.k1 >= 0)) throw new Error('bm25: k1 must be >= 0');
    if (!(this.b >= 0 && this.b <= 1)) throw new Error('bm25: b must be in [0, 1]');
  }

  get docCount(): number {
    return this.ids.length;
  }

  get vocabSize(): number {
    return this.df.size;
  }

  get avgDocLen(): number {
    return this.ids.length === 0 ? 0 : this.totalLen / this.ids.length;
  }

  has(id: string): boolean {
    return this.posById.has(id);
  }

  private terms(text: string): string[] {
    const tokens = tokenize(text);
    return this.stopwords ? removeStopwords(tokens) : tokens;
  }

  addDocument(id: string, text: string): void {
    if (this.posById.has(id)) {
      throw new Error(`bm25: duplicate document id "${id}"`);
    }
    const tokens = this.terms(text);
    const tf = termFreq(tokens);
    this.posById.set(id, this.ids.length);
    this.ids.push(id);
    this.tfByDoc.push(tf);
    this.docLen.push(tokens.length);
    this.totalLen += tokens.length;
    for (const term of tf.keys()) {
      this.df.set(term, (this.df.get(term) ?? 0) + 1);
    }
  }

  idf(term: string): number {
    const n = this.ids.length;
    const df = this.df.get(term) ?? 0;
    if (n === 0 || df === 0) return 0;
    return Math.log(1 + (n - df + 0.5) / (df + 0.5));
  }

  /** Full-corpus scoring. Deterministic output order is the caller's job. */
  score(query: string): Map<string, number> {
    const out = new Map<string, number>();
    const qTerms = this.terms(query);
    if (qTerms.length === 0 || this.ids.length === 0) return out;
    const avgdl = this.avgDocLen || 1;
    const seen = new Set<string>();
    for (const term of qTerms) {
      if (seen.has(term)) continue;
      seen.add(term);
      const idf = this.idf(term);
      if (idf === 0) continue;
      for (let d = 0; d < this.ids.length; d++) {
        const tf = this.tfByDoc[d]!.get(term);
        if (tf === undefined) continue;
        const dl = this.docLen[d]!;
        const denom = tf + this.k1 * (1 - this.b + (this.b * dl) / avgdl);
        const add = (idf * tf * (this.k1 + 1)) / denom;
        const id = this.ids[d]!;
        out.set(id, (out.get(id) ?? 0) + add);
      }
    }
    return out;
  }

  explain(query: string, docId: string): BM25Explanation | null {
    const pos = this.posById.get(docId);
    if (pos === undefined) return null;
    const avgdl = this.avgDocLen || 1;
    const terms: TermContribution[] = [];
    let score = 0;
    const seen = new Set<string>();
    for (const term of this.terms(query)) {
      if (seen.has(term)) continue;
      seen.add(term);
      const idf = this.idf(term);
      const tf = this.tfByDoc[pos]!.get(term) ?? 0;
      const dl = this.docLen[pos]!;
      const denom = tf + this.k1 * (1 - this.b + (this.b * dl) / avgdl);
      const contribution = denom === 0 ? 0 : (idf * tf * (this.k1 + 1)) / denom;
      if (idf > 0) {
        terms.push({ term, idf, tf, docLen: dl, avgDocLen: this.avgDocLen, contribution });
        score += contribution;
      }
    }
    terms.sort((a, z) => z.contribution - a.contribution || a.term.localeCompare(z.term));
    return { docId, score, terms };
  }
}

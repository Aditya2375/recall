/**
 * Extractive answer composer. Instead of generating text, it selects the
 * best supporting sentences from the top-ranked chunks and quotes them
 * verbatim with citations. Every sentence in the output is a substring of
 * the corpus - the unit tests assert exactly that, and the UI links each
 * citation to the source chunk. No generative model, no hallucination path.
 */

import type { BM25Index } from './bm25.js';
import { tokenize } from './tokenize.js';

export interface AnswerInput {
  chunkId: string;
  docId: string;
  text: string;
  /** 1-based position of this chunk in the fused result list; used as the citation number. */
  citation: number;
}

export interface AnswerSentence {
  text: string;
  chunkId: string;
  docId: string;
  citation: number;
  score: number;
}

export interface ComposedAnswer {
  sentences: AnswerSentence[];
}

const SENTENCE = /[^.!?]+[.!?]+(?=\s|$)|[^.!?]+$/g;
const MAX_SENTENCE_CHARS = 400;

function splitSentences(text: string): string[] {
  const out: string[] = [];
  SENTENCE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = SENTENCE.exec(text)) !== null) {
    const s = m[0].replace(/\s+/g, ' ').trim();
    if (s.length > 0 && s.length <= MAX_SENTENCE_CHARS) out.push(s);
  }
  return out;
}

function sentenceScore(sentence: string, queryTerms: Set<string>, bm25: BM25Index): number {
  const terms = new Set(tokenize(sentence));
  if (terms.size === 0) return 0;
  let idfSum = 0;
  for (const term of queryTerms) {
    if (terms.has(term)) idfSum += bm25.idf(term);
  }
  return idfSum / Math.sqrt(terms.size);
}

function overlapRatio(a: Set<string>, z: Set<string>): number {
  if (a.size === 0) return 0;
  let shared = 0;
  for (const t of a) if (z.has(t)) shared += 1;
  return shared / a.size;
}

export function composeAnswer(
  query: string,
  topChunks: AnswerInput[],
  bm25: BM25Index,
  maxSentences = 4,
): ComposedAnswer {
  const queryTerms = new Set(tokenize(query));
  if (queryTerms.size === 0 || topChunks.length === 0) return { sentences: [] };

  interface Candidate extends AnswerSentence {
    order: number;
    terms: Set<string>;
  }
  const candidates: Candidate[] = [];
  topChunks.forEach((chunk, chunkOrder) => {
    splitSentences(chunk.text).forEach((sentence, sentenceOrder) => {
      const score = sentenceScore(sentence, queryTerms, bm25);
      if (score <= 0) return;
      candidates.push({
        text: sentence,
        chunkId: chunk.chunkId,
        docId: chunk.docId,
        citation: chunk.citation,
        score: score / (1 + 0.1 * chunkOrder),
        order: chunkOrder * 1000 + sentenceOrder,
        terms: new Set(tokenize(sentence)),
      });
    });
  });
  candidates.sort((a, z) => z.score - a.score || a.order - z.order);

  const picked: Candidate[] = [];
  for (const cand of candidates) {
    if (picked.length >= maxSentences) break;
    const redundant = picked.some((p) => overlapRatio(cand.terms, p.terms) > 0.6);
    if (!redundant) picked.push(cand);
  }
  picked.sort((a, z) => a.order - z.order);
  return {
    sentences: picked.map(({ text, chunkId, docId, citation, score }) => ({ text, chunkId, docId, citation, score })),
  };
}

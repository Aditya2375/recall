/**
 * Loader and adapters for precomputed model artifacts. The demo can run the
 * entire pipeline - dense search, HNSW, fusion, reranking - from vectors and
 * scores produced by the exact models at build time, verified in CI. Texts
 * that are not part of the bundled corpus or eval set are rejected, so
 * precomputed data can never silently masquerade as live model output.
 */

import { chunkStructure } from './chunk.js';
import type { Corpus } from './corpus.js';
import type { Embedder } from './embed.js';
import type { EvalSet } from './evaluate.js';
import type { Reranker } from './rerank.js';

export interface Precomputed {
  version: 1;
  corpus: string;
  embedModel: string;
  embedRevision: string;
  rerankModel: string;
  rerankRevision: string;
  queryInstruction: string;
  rerankTop: number;
  dims: number;
  chunkIds: string[];
  chunkVectors: string[];
  queryIds: string[];
  queryVectors: string[];
  rerankScores: Record<string, Record<string, number>>;
}

function b64ToBytes(b64: string): Uint8Array {
  // Buffer in Node, atob in the browser; the bundle must not depend on Node globals.
  if (typeof Buffer !== 'undefined') {
    const buf = Buffer.from(b64, 'base64');
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function b64ToF32(b64: string, dims: number): Float32Array {
  const bytes = b64ToBytes(b64);
  const vec = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  if (vec.length !== dims) throw new Error(`precomputed: expected ${dims} dims, got ${vec.length}`);
  return new Float32Array(vec);
}

export function loadPrecomputed(input: unknown): Precomputed {
  const fail = (reason: string): never => {
    throw new Error(`precomputed: ${reason}`);
  };
  if (typeof input !== 'object' || input === null) fail('not an object');
  const p = input as Precomputed;
  if (p.version !== 1) fail('version must be 1');
  if (!Number.isInteger(p.dims) || p.dims <= 0) fail('dims invalid');
  if (!Array.isArray(p.chunkIds) || !Array.isArray(p.chunkVectors)) fail('chunk vectors missing');
  if (p.chunkIds.length !== p.chunkVectors.length) fail('chunk id/vector count mismatch');
  if (!Array.isArray(p.queryIds) || !Array.isArray(p.queryVectors)) fail('query vectors missing');
  if (p.queryIds.length !== p.queryVectors.length) fail('query id/vector count mismatch');
  for (const b64 of [...p.chunkVectors, ...p.queryVectors]) b64ToF32(b64, p.dims);
  return p;
}

export class PrecomputedEmbedder implements Embedder {
  readonly id: string;
  readonly dims: number;
  private readonly byText = new Map<string, Float32Array>();

  constructor(pre: Precomputed, corpus: Corpus, evalSet?: EvalSet) {
    this.dims = pre.dims;
    this.id = `${pre.embedModel}@${pre.embedRevision.slice(0, 8)} (precomputed)`;
    const chunks = corpus.docs.flatMap((d) => chunkStructure(d.id, d.text));
    if (chunks.length !== pre.chunkIds.length) {
      throw new Error(`precomputed: corpus has ${chunks.length} chunks, artifact has ${pre.chunkIds.length}`);
    }
    chunks.forEach((chunk, i) => {
      if (chunk.id !== pre.chunkIds[i]) throw new Error(`precomputed: chunk order mismatch at ${i}`);
      this.byText.set(chunk.text, b64ToF32(pre.chunkVectors[i]!, pre.dims));
    });
    if (evalSet !== undefined) {
      if (evalSet.queries.length !== pre.queryIds.length) throw new Error('precomputed: query count mismatch');
      evalSet.queries.forEach((q, i) => {
        if (q.id !== pre.queryIds[i]) throw new Error(`precomputed: query order mismatch at ${i}`);
        this.byText.set(q.text, b64ToF32(pre.queryVectors[i]!, pre.dims));
      });
    }
  }

  embed(texts: string[]): Promise<Float32Array[]> {
    return Promise.resolve(
      texts.map((t) => {
        const vec = this.byText.get(t);
        if (vec === undefined) throw new Error('precomputed embedder: text not in the bundled corpus or eval set');
        return vec;
      }),
    );
  }
}

export class PrecomputedReranker implements Reranker {
  readonly id: string;
  private readonly scoresByQueryText = new Map<string, Record<string, number>>();
  private readonly docByChunkText = new Map<string, string>();

  constructor(pre: Precomputed, corpus: Corpus, evalSet: EvalSet) {
    this.id = `${pre.rerankModel}@${pre.rerankRevision.slice(0, 8)} (precomputed)`;
    for (const q of evalSet.queries) {
      const scores = pre.rerankScores[q.id];
      if (scores === undefined) throw new Error(`precomputed: no rerank scores for query ${q.id}`);
      this.scoresByQueryText.set(q.text, scores);
    }
    for (const doc of corpus.docs) {
      for (const chunk of chunkStructure(doc.id, doc.text)) {
        this.docByChunkText.set(chunk.text, chunk.docId);
      }
    }
  }

  score(query: string, passages: string[]): Promise<number[]> {
    const scores = this.scoresByQueryText.get(query);
    if (scores === undefined) throw new Error('precomputed reranker: unknown query');
    return Promise.resolve(
      passages.map((p) => {
        const docId = this.docByChunkText.get(p);
        const score = docId === undefined ? undefined : scores[docId];
        if (score === undefined) throw new Error('precomputed reranker: passage outside the precomputed top set');
        return score;
      }),
    );
  }
}

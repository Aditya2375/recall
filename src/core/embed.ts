/**
 * Embedding interface plus the deterministic test embedder.
 *
 * The real model (bge-small-en-v1.5 via ONNX Runtime) implements this same
 * interface in the web bundle and in the precompute script. HashEmbedder
 * exists so unit tests and CI run hermetically: it is a signed random
 * projection of token hashes - good enough to exercise every code path,
 * never presented as a quality signal.
 */

import { termFreq, tokenize } from './tokenize.js';
import { normalize } from './vector.js';

export interface Embedder {
  readonly id: string;
  readonly dims: number;
  embed(texts: string[]): Promise<Float32Array[]>;
}

function fnv1a(input: string, seed: number): number {
  let h = (0x811c9dc5 ^ seed) >>> 0;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

export class HashEmbedder implements Embedder {
  readonly id: string;
  readonly dims: number;
  private readonly seed: number;

  constructor(seed = 7, dims = 256) {
    if (!Number.isInteger(dims) || dims <= 0) throw new Error('hash-embedder: dims must be a positive integer');
    this.seed = seed >>> 0;
    this.dims = dims;
    this.id = `hash-embedder(seed=${this.seed},dims=${dims})`;
  }

  embed(texts: string[]): Promise<Float32Array[]> {
    return Promise.resolve(texts.map((t) => this.embedOne(t)));
  }

  private embedOne(text: string): Float32Array {
    const v = new Float32Array(this.dims);
    const tf = termFreq(tokenize(text));
    for (const [term, count] of tf) {
      const h1 = fnv1a(term, this.seed);
      const h2 = fnv1a(term, this.seed ^ 0x9e3779b9);
      const weight = Math.sqrt(count);
      v[h1 % this.dims]! += (h1 & 1 ? 1 : -1) * weight;
      v[h2 % this.dims]! += (h2 & 1 ? 1 : -1) * weight * 0.5;
    }
    return normalize(v);
  }
}

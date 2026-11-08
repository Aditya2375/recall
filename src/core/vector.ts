/**
 * Exact (flat) vector index: brute-force cosine similarity over L2-normalized
 * vectors. This is the ground truth that the HNSW index is measured against.
 */

export interface VectorHit {
  id: string;
  score: number;
}

export function normalize(v: Float32Array): Float32Array {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i]! * v[i]!;
  const norm = Math.sqrt(sum);
  const out = new Float32Array(v.length);
  if (norm === 0) return out;
  for (let i = 0; i < v.length; i++) out[i] = v[i]! / norm;
  return out;
}

export function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i]! * b[i]!;
  return sum;
}

export class FlatIndex {
  readonly dims: number;
  private readonly ids: string[] = [];
  private readonly posById = new Map<string, number>();
  private readonly vectors: Float32Array[] = [];

  constructor(dims: number) {
    if (!Number.isInteger(dims) || dims <= 0) throw new Error('flat: dims must be a positive integer');
    this.dims = dims;
  }

  get size(): number {
    return this.ids.length;
  }

  has(id: string): boolean {
    return this.posById.has(id);
  }

  vector(id: string): Float32Array | undefined {
    const pos = this.posById.get(id);
    return pos === undefined ? undefined : this.vectors[pos];
  }

  add(id: string, vector: Float32Array): void {
    if (this.posById.has(id)) throw new Error(`flat: duplicate id "${id}"`);
    if (vector.length !== this.dims) throw new Error(`flat: expected ${this.dims} dims, got ${vector.length}`);
    this.posById.set(id, this.ids.length);
    this.ids.push(id);
    this.vectors.push(normalize(vector));
  }

  /** Cosine similarity search; hits sorted by score desc, ties by id asc. */
  search(query: Float32Array, k: number): VectorHit[] {
    if (k <= 0) return [];
    const q = normalize(query);
    const hits: VectorHit[] = [];
    for (let i = 0; i < this.ids.length; i++) {
      hits.push({ id: this.ids[i]!, score: dot(q, this.vectors[i]!) });
    }
    hits.sort((a, z) => z.score - a.score || (a.id < z.id ? -1 : a.id > z.id ? 1 : 0));
    return hits.slice(0, k);
  }
}

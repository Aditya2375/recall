/**
 * HNSW (Hierarchical Navigable Small World) approximate nearest neighbour
 * index, after Malkov & Yashunin (2016). Cosine distance on L2-normalized
 * vectors (distance = 1 - dot). Level assignment and graph construction are
 * driven by a seeded RNG, so a given (corpus, seed) pair always builds the
 * same graph - the demo's recall measurements are reproducible.
 */

import { mulberry32 } from './rng.js';
import { dot, normalize } from './vector.js';

export interface HnswOptions {
  M?: number;
  efConstruction?: number;
  seed?: number;
}

interface Node {
  id: string;
  vec: Float32Array;
  links: number[][];
}

/** Binary heap of node indices ordered by distance, ties by index. */
class Heap {
  private items: number[] = [];
  constructor(
    private readonly distOf: (idx: number) => number,
    private readonly farthestFirst: boolean,
  ) {}

  get size(): number {
    return this.items.length;
  }

  peek(): number {
    return this.items[0]!;
  }

  private before(a: number, z: number): boolean {
    const da = this.distOf(a);
    const dz = this.distOf(z);
    if (da !== dz) return this.farthestFirst ? da > dz : da < dz;
    return this.farthestFirst ? a > z : a < z;
  }

  push(item: number): void {
    const items = this.items;
    items.push(item);
    let i = items.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.before(items[parent]!, items[i]!)) break;
      [items[parent], items[i]] = [items[i]!, items[parent]!];
      i = parent;
    }
  }

  pop(): number {
    const items = this.items;
    const top = items[0]!;
    const last = items.pop()!;
    if (items.length > 0) {
      items[0] = last;
      let i = 0;
      for (;;) {
        const left = 2 * i + 1;
        const right = left + 1;
        let best = i;
        if (left < items.length && this.before(items[left]!, items[best]!)) best = left;
        if (right < items.length && this.before(items[right]!, items[best]!)) best = right;
        if (best === i) break;
        [items[best], items[i]] = [items[i]!, items[best]!];
        i = best;
      }
    }
    return top;
  }

  values(): number[] {
    return this.items.slice();
  }
}

export class HnswIndex {
  readonly dims: number;
  readonly M: number;
  readonly efConstruction: number;
  private readonly mMax0: number;
  private readonly mL: number;
  private readonly rand: () => number;
  private readonly nodes: Node[] = [];
  private readonly posById = new Map<string, number>();
  private enterPoint = -1;
  private maxLevel = -1;

  constructor(dims: number, options: HnswOptions = {}) {
    if (!Number.isInteger(dims) || dims <= 0) throw new Error('hnsw: dims must be a positive integer');
    this.dims = dims;
    this.M = options.M ?? 16;
    this.efConstruction = options.efConstruction ?? 200;
    if (!(this.M >= 2)) throw new Error('hnsw: M must be >= 2');
    if (!(this.efConstruction >= 1)) throw new Error('hnsw: efConstruction must be >= 1');
    this.mMax0 = this.M * 2;
    this.mL = 1 / Math.log(this.M);
    this.rand = mulberry32(options.seed ?? 42);
  }

  get size(): number {
    return this.nodes.length;
  }

  has(id: string): boolean {
    return this.posById.has(id);
  }

  vector(id: string): Float32Array | undefined {
    const pos = this.posById.get(id);
    return pos === undefined ? undefined : this.nodes[pos]!.vec;
  }

  private dist(a: Float32Array, b: Float32Array): number {
    return 1 - dot(a, b);
  }

  private level(): number {
    let r = this.rand();
    if (r <= 0) r = Number.EPSILON;
    return Math.floor(-Math.log(r) * this.mL);
  }

  add(id: string, vector: Float32Array): void {
    if (this.posById.has(id)) throw new Error(`hnsw: duplicate id "${id}"`);
    if (vector.length !== this.dims) throw new Error(`hnsw: expected ${this.dims} dims, got ${vector.length}`);
    const vec = normalize(vector);
    const top = this.level();
    const idx = this.nodes.length;
    const node: Node = { id, vec, links: Array.from({ length: top + 1 }, () => []) };
    this.nodes.push(node);
    this.posById.set(id, idx);

    if (this.enterPoint === -1) {
      this.enterPoint = idx;
      this.maxLevel = top;
      return;
    }

    let ep: number[] = [this.enterPoint];
    for (let lc = this.maxLevel; lc > top; lc--) {
      const nearest = this.searchLayer(vec, ep, 1, lc);
      ep = [nearest[0]!];
    }
    for (let lc = Math.min(top, this.maxLevel); lc >= 0; lc--) {
      const candidates = this.searchLayer(vec, ep, this.efConstruction, lc);
      const chosen = this.selectNeighbors(vec, candidates, this.M);
      node.links[lc] = chosen;
      const mMax = lc === 0 ? this.mMax0 : this.M;
      for (const nb of chosen) {
        const nbNode = this.nodes[nb]!;
        nbNode.links[lc]!.push(idx);
        if (nbNode.links[lc]!.length > mMax) {
          nbNode.links[lc] = this.selectNeighbors(nbNode.vec, nbNode.links[lc]!, mMax);
        }
      }
      ep = candidates;
    }
    if (top > this.maxLevel) {
      this.enterPoint = idx;
      this.maxLevel = top;
    }
  }

  /** Nearest `ef` nodes to `query` on `layer`, sorted nearest-first. */
  private searchLayer(query: Float32Array, entryPoints: number[], ef: number, layer: number): number[] {
    const distCache = new Map<number, number>();
    const distOf = (idx: number): number => {
      let d = distCache.get(idx);
      if (d === undefined) {
        d = this.dist(query, this.nodes[idx]!.vec);
        distCache.set(idx, d);
      }
      return d;
    };
    const visited = new Set<number>(entryPoints);
    const candidates = new Heap(distOf, false);
    const results = new Heap(distOf, true);
    for (const ep of entryPoints) {
      distOf(ep);
      candidates.push(ep);
      results.push(ep);
    }
    while (candidates.size > 0) {
      const c = candidates.pop();
      const f = results.peek();
      if (distOf(c) > distOf(f)) break;
      for (const nb of this.nodes[c]!.links[layer] ?? []) {
        if (visited.has(nb)) continue;
        visited.add(nb);
        const fDist = distOf(results.peek());
        const d = distOf(nb);
        if (d < fDist || results.size < ef) {
          candidates.push(nb);
          results.push(nb);
          if (results.size > ef) results.pop();
        }
      }
    }
    return results.values().sort((a, z) => distOf(a) - distOf(z) || a - z);
  }

  /** Diversity heuristic, paper algorithm 4 (keepPrunedConnections = false). */
  private selectNeighbors(query: Float32Array, candidates: number[], m: number): number[] {
    const pool = candidates
      .map((idx) => ({ idx, d: this.dist(query, this.nodes[idx]!.vec) }))
      .sort((a, z) => a.d - z.d || a.idx - z.idx);
    const chosen: number[] = [];
    for (const cand of pool) {
      if (chosen.length >= m) break;
      let closer = true;
      for (const other of chosen) {
        const dToOther = this.dist(this.nodes[cand.idx]!.vec, this.nodes[other]!.vec);
        if (dToOther < cand.d) {
          closer = false;
          break;
        }
      }
      if (closer) chosen.push(cand.idx);
    }
    return chosen.sort((a, z) => a - z);
  }

  /** Approximate cosine search; hits sorted by score desc, ties by id asc. */
  search(query: Float32Array, k: number, ef = 50): Array<{ id: string; score: number }> {
    if (k <= 0 || this.nodes.length === 0) return [];
    const q = normalize(query);
    let ep: number[] = [this.enterPoint];
    for (let lc = this.maxLevel; lc >= 1; lc--) {
      const nearest = this.searchLayer(q, ep, 1, lc);
      ep = [nearest[0]!];
    }
    const found = this.searchLayer(q, ep, Math.max(ef, k), 0);
    return found.slice(0, k).map((idx) => ({
      id: this.nodes[idx]!.id,
      score: 1 - this.dist(q, this.nodes[idx]!.vec),
    }));
  }

  /** Graph shape summary for the Inspect view. */
  stats(): { size: number; maxLevel: number; avgDegree0: number } {
    let degree = 0;
    for (const n of this.nodes) degree += n.links[0]?.length ?? 0;
    return {
      size: this.nodes.length,
      maxLevel: this.maxLevel,
      avgDegree0: this.nodes.length === 0 ? 0 : degree / this.nodes.length,
    };
  }
}

/**
 * Retriever: the full hybrid pipeline.
 *
 *   documents -> chunks -> BM25 index + vector index (flat and/or HNSW)
 *   query     -> per-list chunk rankings -> aggregate to documents
 *             -> fuse (RRF or weighted linear) -> optional neural rerank
 *             -> ranked documents with full score provenance
 *
 * Chunk-to-document aggregation takes the best-scoring chunk per document
 * (MaxP) and the document's rank from its first appearance in the list.
 */

import { BM25Index, type BM25Options } from './bm25.js';
import { chunkFixed, chunkStructure, type Chunk, type ChunkOptions } from './chunk.js';
import type { CorpusDoc } from './corpus.js';
import type { Embedder } from './embed.js';
import { linear, rrf } from './fusion.js';
import { HnswIndex, type HnswOptions } from './hnsw.js';
import type { Reranker } from './rerank.js';
import { FlatIndex } from './vector.js';

export interface RetrieverOptions {
  chunker?: 'fixed' | 'structure';
  chunkOptions?: ChunkOptions;
  bm25?: BM25Options;
  /** null disables the ANN index; flat search then serves every dense query. */
  hnsw?: HnswOptions | null;
}

export interface RankedDoc {
  docId: string;
  chunkId: string;
  rank: number;
  scores: {
    bm25?: number;
    dense?: number;
    fused: number;
    rerank?: number;
  };
  ranks: {
    bm25?: number;
    dense?: number;
  };
}

export interface QueryOptions {
  k?: number;
  lexical?: boolean;
  dense?: 'flat' | 'hnsw' | null;
  fusion?: 'rrf' | 'linear';
  /** Dense-list weight for linear fusion; BM25 gets 1 - alpha. */
  alpha?: number;
  efSearch?: number;
  /** Cross-encoder over the top N fused docs; 0 disables reranking. */
  rerankTop?: number;
}

export interface LatencyBreakdown {
  embed: number;
  lexical: number;
  dense: number;
  fusion: number;
  rerank: number;
  total: number;
}

export interface QueryResult {
  query: string;
  mode: string;
  docs: RankedDoc[];
  latencyMs: LatencyBreakdown;
}

interface DocAggregate {
  docId: string;
  chunkId: string;
  score: number;
  rank: number;
}

function aggregateChunks(chunkIds: string[], scores: ReadonlyMap<string, number>, docOf: (chunkId: string) => string): DocAggregate[] {
  const out: DocAggregate[] = [];
  const seenDocs = new Set<string>();
  for (const chunkId of chunkIds) {
    const docId = docOf(chunkId);
    if (seenDocs.has(docId)) continue;
    seenDocs.add(docId);
    out.push({ docId, chunkId, score: scores.get(chunkId) ?? 0, rank: out.length + 1 });
  }
  return out;
}

export class Retriever {
  readonly name: string;
  readonly docs: CorpusDoc[];
  readonly chunks: Chunk[];
  readonly embedderId: string;
  readonly bm25Index: BM25Index;
  readonly flat: FlatIndex | null;
  readonly hnsw: HnswIndex | null;

  private readonly docById = new Map<string, CorpusDoc>();
  private readonly chunkByIdMap = new Map<string, Chunk>();

  private constructor(
    name: string,
    docs: CorpusDoc[],
    chunks: Chunk[],
    embedderId: string,
    bm25: BM25Index,
    flat: FlatIndex | null,
    hnsw: HnswIndex | null,
  ) {
    this.name = name;
    this.docs = docs;
    this.chunks = chunks;
    this.embedderId = embedderId;
    this.bm25Index = bm25;
    this.flat = flat;
    this.hnsw = hnsw;
    for (const d of docs) this.docById.set(d.id, d);
    for (const c of chunks) this.chunkByIdMap.set(c.id, c);
  }

  static async fromDocuments(
    name: string,
    docs: CorpusDoc[],
    embedder: Embedder,
    options: RetrieverOptions = {},
    onProgress?: (stage: string, done: number, total: number) => void,
  ): Promise<Retriever> {
    if (docs.length === 0) throw new Error('retriever: no documents');
    const chunkOpts = options.chunkOptions ?? {};
    const chunker = options.chunker ?? 'structure';
    onProgress?.('chunking', 0, docs.length);
    const chunks: Chunk[] = [];
    docs.forEach((doc, i) => {
      const made = chunker === 'fixed' ? chunkFixed(doc.id, doc.text, chunkOpts) : chunkStructure(doc.id, doc.text, chunkOpts);
      if (made.length === 0) {
        chunks.push({
          id: `${doc.id}#c0`,
          docId: doc.id,
          index: 0,
          text: doc.title,
          start: 0,
          end: 0,
          headings: [],
        });
      } else {
        chunks.push(...made);
      }
      onProgress?.('chunking', i + 1, docs.length);
    });

    const bm25 = new BM25Index(options.bm25 ?? {});
    for (const c of chunks) bm25.addDocument(c.id, c.text);

    onProgress?.('embedding', 0, chunks.length);
    const embeddings = await embedder.embed(chunks.map((c) => c.text));
    onProgress?.('embedding', chunks.length, chunks.length);

    const flat = new FlatIndex(embedder.dims);
    embeddings.forEach((vec, i) => flat.add(chunks[i]!.id, vec));

    let hnsw: HnswIndex | null = null;
    if (options.hnsw !== null) {
      hnsw = new HnswIndex(embedder.dims, options.hnsw ?? {});
      embeddings.forEach((vec, i) => hnsw!.add(chunks[i]!.id, vec));
    }
    onProgress?.('indexing', 1, 1);
    return new Retriever(name, docs, chunks, embedder.id, bm25, flat, hnsw);
  }

  docByIdGet(id: string): CorpusDoc | undefined {
    return this.docById.get(id);
  }

  chunkById(id: string): Chunk | undefined {
    return this.chunkByIdMap.get(id);
  }

  private docOfChunk(chunkId: string): string {
    const chunk = this.chunkByIdMap.get(chunkId);
    if (chunk === undefined) throw new Error(`retriever: unknown chunk "${chunkId}"`);
    return chunk.docId;
  }

  async query(queryText: string, opts: QueryOptions = {}, embedder?: Embedder, reranker?: Reranker): Promise<QueryResult> {
    const t0 = performance.now();
    const k = opts.k ?? 10;
    const useLexical = opts.lexical ?? true;
    const denseMode = opts.dense === undefined ? 'flat' : opts.dense;
    const fusion = opts.fusion ?? 'rrf';
    const alpha = opts.alpha ?? 0.5;
    const rerankTop = opts.rerankTop ?? 0;
    if (!(k > 0)) throw new Error('retriever: k must be > 0');
    if (!(alpha >= 0 && alpha <= 1)) throw new Error('retriever: alpha must be in [0, 1]');
    if (!useLexical && denseMode === null) throw new Error('retriever: lexical and dense cannot both be off');

    const latency: LatencyBreakdown = { embed: 0, lexical: 0, dense: 0, fusion: 0, rerank: 0, total: 0 };

    let bm25Agg: DocAggregate[] = [];
    let bm25Scores = new Map<string, number>();
    if (useLexical) {
      const t = performance.now();
      const chunkScores = this.bm25Index.score(queryText);
      bm25Scores = chunkScores;
      const sortedIds = [...chunkScores.entries()]
        .sort((a, z) => z[1] - a[1] || (a[0] < z[0] ? -1 : 1))
        .map(([id]) => id);
      bm25Agg = aggregateChunks(sortedIds, chunkScores, (c) => this.docOfChunk(c));
      latency.lexical = performance.now() - t;
    }

    let denseAgg: DocAggregate[] = [];
    let denseScores = new Map<string, number>();
    if (denseMode !== null) {
      if (embedder === undefined) throw new Error('retriever: dense search needs an embedder');
      const t = performance.now();
      const [qvec] = await embedder.embed([queryText]);
      latency.embed = performance.now() - t;
      const t2 = performance.now();
      const index = denseMode === 'hnsw' ? this.hnsw : this.flat;
      if (index === null || index === undefined) throw new Error(`retriever: no ${denseMode} index built`);
      const hits = index.search(qvec!, Math.max(k * 4, 25), opts.efSearch ?? 50);
      denseScores = new Map(hits.map((h) => [h.id, h.score]));
      denseAgg = aggregateChunks(hits.map((h) => h.id), denseScores, (c) => this.docOfChunk(c));
      latency.dense = performance.now() - t2;
    }

    const t3 = performance.now();
    const docLists: string[][] = [];
    const weighted: Array<{ ids: string[]; scores: ReadonlyMap<string, number>; weight: number }> = [];
    if (bm25Agg.length > 0) {
      docLists.push(bm25Agg.map((d) => d.docId));
      const m = new Map<string, number>();
      for (const d of bm25Agg) m.set(d.docId, d.score);
      weighted.push({ ids: bm25Agg.map((d) => d.docId), scores: m, weight: 1 - alpha });
    }
    if (denseAgg.length > 0) {
      docLists.push(denseAgg.map((d) => d.docId));
      const m = new Map<string, number>();
      for (const d of denseAgg) m.set(d.docId, d.score);
      weighted.push({ ids: denseAgg.map((d) => d.docId), scores: m, weight: alpha });
    }
    const fused = fusion === 'rrf' ? rrf(docLists) : linear(weighted);
    const fusedEntries = [...fused.entries()];
    latency.fusion = performance.now() - t3;

    const bm25ByDoc = new Map(bm25Agg.map((d) => [d.docId, d]));
    const denseByDoc = new Map(denseAgg.map((d) => [d.docId, d]));
    const bestChunkFor = (docId: string): string => {
      const b = bm25ByDoc.get(docId);
      const d = denseByDoc.get(docId);
      if (b === undefined) return d!.chunkId;
      if (d === undefined) return b.chunkId;
      // Both lists found the document: show the chunk from whichever list
      // ranked the document higher; ties go to the dense list.
      return b.rank < d.rank ? b.chunkId : d.chunkId;
    };

    let order = fusedEntries.slice(0, Math.max(k, rerankTop)).map(([docId]) => docId);
    const rerankScores = new Map<string, number>();
    if (rerankTop > 0 && order.length > 0) {
      if (reranker === undefined) throw new Error('retriever: rerankTop > 0 needs a reranker');
      const t4 = performance.now();
      const head = order.slice(0, rerankTop);
      const tail = order.slice(rerankTop);
      const passages = head.map((docId) => this.chunkByIdMap.get(bestChunkFor(docId))!.text);
      const scores = await reranker.score(queryText, passages);
      const reranked = head
        .map((docId, i) => ({ docId, score: scores[i]!, pos: i }))
        .sort((a, z) => z.score - a.score || a.pos - z.pos);
      for (const r of reranked) rerankScores.set(r.docId, r.score);
      order = [...reranked.map((r) => r.docId), ...tail];
      latency.rerank = performance.now() - t4;
    }

    const docs: RankedDoc[] = order.slice(0, k).map((docId, i) => {
      const b = bm25ByDoc.get(docId);
      const d = denseByDoc.get(docId);
      const entry: RankedDoc = {
        docId,
        chunkId: bestChunkFor(docId),
        rank: i + 1,
        scores: { fused: fused.get(docId) ?? 0 },
        ranks: {},
      };
      if (b !== undefined) {
        entry.scores.bm25 = b.score;
        entry.ranks.bm25 = b.rank;
      }
      if (d !== undefined) {
        entry.scores.dense = d.score;
        entry.ranks.dense = d.rank;
      }
      const rr = rerankScores.get(docId);
      if (rr !== undefined) entry.scores.rerank = rr;
      return entry;
    });

    const parts: string[] = [];
    if (useLexical) parts.push('bm25');
    if (denseMode !== null) parts.push(`dense-${denseMode}`);
    if (parts.length === 2) parts.push(fusion === 'rrf' ? 'rrf' : `linear-${alpha.toFixed(2)}`);
    if (rerankTop > 0) parts.push('rerank');

    latency.total = performance.now() - t0;
    return { query: queryText, mode: parts.join(' + ') || 'none', docs, latencyMs: latency };
  }
}

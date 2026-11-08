/**
 * Compute worker. All indexing, searching and evaluation run here so the UI
 * thread never blocks. The bundled corpus runs from precomputed vectors;
 * custom queries and user documents lazily load the real models.
 *
 * Protocol (postMessage):
 *   in:  {kind:'init', corpus, evalSet, precomputed}
 *   in:  {kind:'query', id, text, options, corpusId}
 *   in:  {kind:'eval', id}
 *   in:  {kind:'efSweep', id}
 *   in:  {kind:'loadModel', id, which:'embed'|'rerank'}
 *   in:  {kind:'indexCorpus', id, corpus}
 *   out: {kind:'ready', stats, chunks, docs}
 *   out: {kind:'result', id, result} | {kind:'evalResult', id, reports}
 *   out: {kind:'efResult', id, points} | {kind:'modelReady', id, which}
 *   out: {kind:'corpusReady', id, corpusId, stats, chunks, docs}
 *   out: {kind:'progress', id, stage, fraction} | {kind:'error', id, message}
 */

import { validateCorpus, type Corpus, type CorpusDoc } from '../core/corpus.js';
import { validateEvalSet, runEvaluation, type EvalSet } from '../core/evaluate.js';
import { loadPrecomputed, PrecomputedEmbedder, PrecomputedReranker } from '../core/precomputed.js';
import { Retriever, type QueryOptions } from '../core/pipeline.js';
import type { Chunk } from '../core/chunk.js';
import { composeAnswer, type AnswerInput } from '../core/answer.js';
import { CrossEncoderReranker, TransformersEmbedder } from './models.js';

interface CorpusState {
  corpus: Corpus;
  retriever: Retriever;
  precomputed: PrecomputedEmbedder | null;
  reranker: PrecomputedReranker | null;
  evalSet: EvalSet | null;
}

const states = new Map<string, CorpusState>();
const liveEmbedder = new TransformersEmbedder();
const liveReranker = new CrossEncoderReranker();

const post = (message: unknown) => (self as unknown as { postMessage: (m: unknown) => void }).postMessage(message);

function statsOf(state: CorpusState): Record<string, number | string> {
  const r = state.retriever;
  return {
    docs: r.docs.length,
    chunks: r.chunks.length,
    vocab: r.bm25Index.vocabSize,
    avgChunkTokens: Math.round(r.bm25Index.avgDocLen),
    embedder: r.embedderId,
  };
}

function serializableChunks(retriever: Retriever): Chunk[] {
  return retriever.chunks.map((c) => ({ ...c }));
}

async function handleInit(id: number, msg: { corpus: unknown; evalSet: unknown; precomputed: unknown }): Promise<void> {
  const corpus = validateCorpus(msg.corpus);
  const evalSet = validateEvalSet(msg.evalSet, new Set(corpus.docs.map((d) => d.id)));
  const pre = loadPrecomputed(msg.precomputed);
  if (pre.corpus !== corpus.name) throw new Error(`precomputed artifact is for corpus "${pre.corpus}", not "${corpus.name}"`);
  const embedder = new PrecomputedEmbedder(pre, corpus, evalSet);
  const retriever = await Retriever.fromDocuments(corpus.name, corpus.docs, embedder, { hnsw: { seed: 42 } });
  states.set(corpus.name, {
    corpus,
    retriever,
    precomputed: embedder,
    reranker: new PrecomputedReranker(pre, corpus, evalSet),
    evalSet,
  });
  post({ kind: 'ready', id, corpusId: corpus.name, stats: statsOf(states.get(corpus.name)!), chunks: serializableChunks(retriever), docs: corpus.docs });
}

async function handleQuery(id: number, msg: { text: string; options: QueryOptions; corpusId: string }): Promise<void> {
  const state = states.get(msg.corpusId);
  if (state === undefined) throw new Error(`unknown corpus "${msg.corpusId}"`);
  const modelsUsed: string[] = [];
  let embedder = state.precomputed ?? undefined;
  if (msg.options.dense !== null) {
    if (state.precomputed === null) {
      await liveEmbedder.load((f) => post({ kind: 'progress', id, stage: 'loading embedding model', fraction: f }));
      embedder = liveEmbedder;
      modelsUsed.push('embed');
    } else {
      // Precomputed covers bundled queries; anything else needs the live model.
      try {
        await state.precomputed.embed([msg.text]);
      } catch {
        await liveEmbedder.load((f) => post({ kind: 'progress', id, stage: 'loading embedding model', fraction: f }));
        embedder = new LiveQueryEmbedder(state.precomputed, liveEmbedder);
        modelsUsed.push('embed');
      }
    }
  }
  let reranker = state.reranker ?? undefined;
  if ((msg.options.rerankTop ?? 0) > 0) {
    if (state.reranker === null || !(await precomputedRerankCovers(state, msg.text))) {
      await liveReranker.load((f) => post({ kind: 'progress', id, stage: 'loading reranker model', fraction: f }));
      reranker = liveReranker;
      modelsUsed.push('rerank');
    }
  }
  const result = await state.retriever.query(msg.text, msg.options, embedder, reranker);
  const topChunks: AnswerInput[] = result.docs.slice(0, 3).map((doc) => ({
    chunkId: doc.chunkId,
    docId: doc.docId,
    text: state.retriever.chunkById(doc.chunkId)!.text,
    citation: doc.rank,
  }));
  const answer = composeAnswer(msg.text, topChunks, state.retriever.bm25Index);
  post({ kind: 'result', id, result: { ...result, answer }, modelsUsed });
}

function handleExplain(id: number, msg: { corpusId: string; chunkId: string; query: string }): void {
  const state = states.get(msg.corpusId);
  if (state === undefined) throw new Error(`unknown corpus "${msg.corpusId}"`);
  post({ kind: 'explanation', id, explanation: state.retriever.bm25Index.explain(msg.query, msg.chunkId) });
}

async function precomputedRerankCovers(state: CorpusState, text: string): Promise<boolean> {
  if (state.reranker === null) return false;
  try {
    await state.reranker.score(text, ['probe']);
    return true;
  } catch {
    return false;
  }
}

/** Bundled corpus: precomputed chunk vectors, live model for unseen queries. */
class LiveQueryEmbedder extends TransformersEmbedder {
  constructor(
    private readonly fallback: PrecomputedEmbedder,
    private readonly live: TransformersEmbedder,
  ) {
    super();
  }
  override async embed(texts: string[]): Promise<Float32Array[]> {
    const out: Float32Array[] = [];
    const missing: number[] = [];
    for (let i = 0; i < texts.length; i++) {
      try {
        out[i] = (await this.fallback.embed([texts[i]!]))[0]!;
      } catch {
        out[i] = new Float32Array();
        missing.push(i);
      }
    }
    if (missing.length > 0) {
      const liveVecs = await this.live.embed(missing.map((i) => texts[i]!));
      missing.forEach((idx, j) => {
        out[idx] = liveVecs[j]!;
      });
    }
    return out;
  }
}

async function handleEval(id: number, corpusId: string): Promise<void> {
  const state = states.get(corpusId);
  if (state?.evalSet == null || state.precomputed == null || state.reranker == null) {
    throw new Error('evaluation is only available for the bundled corpus');
  }
  const r = state.retriever;
  const embedder = state.precomputed;
  const K = 10;
  const configs = [
    { name: 'bm25', run: async (q: string) => { const x = await r.query(q, { k: K, dense: null }); return { ranked: x.docs.map((d) => d.docId), latencyMs: x.latencyMs.total }; } },
    { name: 'dense (flat)', run: async (q: string) => { const x = await r.query(q, { k: K, lexical: false, dense: 'flat' }, embedder); return { ranked: x.docs.map((d) => d.docId), latencyMs: x.latencyMs.total }; } },
    { name: 'dense (hnsw ef=50)', run: async (q: string) => { const x = await r.query(q, { k: K, lexical: false, dense: 'hnsw', efSearch: 50 }, embedder); return { ranked: x.docs.map((d) => d.docId), latencyMs: x.latencyMs.total }; } },
    { name: 'hybrid rrf', run: async (q: string) => { const x = await r.query(q, { k: K, fusion: 'rrf' }, embedder); return { ranked: x.docs.map((d) => d.docId), latencyMs: x.latencyMs.total }; } },
    { name: 'hybrid linear a=0.5', run: async (q: string) => { const x = await r.query(q, { k: K, fusion: 'linear', alpha: 0.5 }, embedder); return { ranked: x.docs.map((d) => d.docId), latencyMs: x.latencyMs.total }; } },
    { name: 'hybrid rrf + rerank', run: async (q: string) => { const x = await r.query(q, { k: K, fusion: 'rrf', rerankTop: 25 }, embedder, state.reranker!); return { ranked: x.docs.map((d) => d.docId), latencyMs: x.latencyMs.total }; } },
  ];
  const reports = await runEvaluation(configs, state.evalSet);
  post({ kind: 'evalResult', id, reports });
}

async function handleEfSweep(id: number, corpusId: string): Promise<void> {
  const state = states.get(corpusId);
  if (state?.evalSet == null || state.precomputed == null) throw new Error('ef sweep needs the bundled corpus');
  const r = state.retriever;
  if (r.flat === null || r.hnsw === null) throw new Error('both indexes required');
  const points: Array<{ ef: number; recall: number; latencyMs: number }> = [];
  for (const ef of [10, 25, 50, 100, 200]) {
    let hits = 0;
    let total = 0;
    const t0 = performance.now();
    for (const q of state.evalSet.queries) {
      const [qvec] = await state.precomputed.embed([q.text]);
      const exact = new Set(r.flat.search(qvec!, 10).map((h) => h.id));
      const approx = r.hnsw.search(qvec!, 10, ef);
      total += exact.size;
      for (const h of approx) if (exact.has(h.id)) hits += 1;
    }
    const latencyMs = (performance.now() - t0) / state.evalSet.queries.length;
    points.push({ ef, recall: total === 0 ? 1 : hits / total, latencyMs });
    post({ kind: 'progress', id, stage: `ef=${ef}`, fraction: points.length / 5 });
  }
  post({ kind: 'efResult', id, points });
}

async function handleIndexCorpus(id: number, raw: unknown): Promise<void> {
  const corpus = validateCorpus(raw);
  await liveEmbedder.load((f) => post({ kind: 'progress', id, stage: 'loading embedding model', fraction: f }));
  const retriever = await Retriever.fromDocuments(
    corpus.name,
    corpus.docs as CorpusDoc[],
    liveEmbedder,
    { hnsw: { seed: 42 } },
    (stage, done, total) => post({ kind: 'progress', id, stage, fraction: total === 0 ? 0 : done / total }),
  );
  states.set(corpus.name, { corpus, retriever, precomputed: null, reranker: null, evalSet: null });
  post({ kind: 'corpusReady', id, corpusId: corpus.name, stats: statsOf(states.get(corpus.name)!), chunks: serializableChunks(retriever), docs: corpus.docs });
}

self.onmessage = (event: MessageEvent) => {
  const msg = event.data as { kind: string; id?: number } & Record<string, unknown>;
  const id = msg.id ?? 0;
  const run = async () => {
    switch (msg.kind) {
      case 'init':
        return handleInit(id, msg as never);
      case 'query':
        return handleQuery(id, msg as never);
      case 'eval':
        return handleEval(id, msg['corpusId'] as string);
      case 'efSweep':
        return handleEfSweep(id, msg['corpusId'] as string);
      case 'loadModel': {
        const which = msg['which'] as string;
        if (which === 'embed') await liveEmbedder.load((f) => post({ kind: 'progress', id, stage: 'loading embedding model', fraction: f }));
        else await liveReranker.load((f) => post({ kind: 'progress', id, stage: 'loading reranker model', fraction: f }));
        post({ kind: 'modelReady', id, which });
        return;
      }
      case 'indexCorpus':
        return handleIndexCorpus(id, msg['corpus']);
      case 'explain':
        handleExplain(id, msg as never);
        return;
      default:
        throw new Error(`unknown message kind "${msg.kind}"`);
    }
  };
  run().catch((err: unknown) => {
    post({ kind: 'error', id, message: err instanceof Error ? err.message : String(err) });
  });
};

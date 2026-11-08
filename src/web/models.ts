/**
 * Live model adapters for the browser: the same Embedder and Reranker
 * interfaces the core defines, backed by ONNX Runtime Web through
 * transformers.js. Models are pinned to exact revisions and downloaded
 * lazily - only when the user runs a custom query or indexes their own
 * documents. The bundled corpus and eval numbers never need a download.
 */

import type { Embedder } from '../core/embed.js';
import type { Reranker } from '../core/rerank.js';

export const EMBED_MODEL = 'Xenova/bge-small-en-v1.5';
export const EMBED_REVISION = 'ea104dacec62c0de699686887e3f920caeb4f3e3';
export const RERANK_MODEL = 'Xenova/ms-marco-MiniLM-L-6-v2';
export const RERANK_REVISION = 'a09144355adeed5f58c8ed011d209bf8ee5a1fec';
/** bge models expect this instruction prefix on queries, not on passages. */
export const QUERY_INSTRUCTION = 'Represent this sentence for searching relevant passages: ';

type FeatureExtractionPipeline = (texts: string[], options: Record<string, unknown>) => Promise<{ tolist: () => number[][] }>;

/**
 * Points ONNX Runtime Web at the wasm files the repo ships under docs/ort/
 * instead of its jsdelivr CDN default, so the demo's CSP only ever talks to
 * this site and the pinned Hugging Face hosts. Must run before the first
 * pipeline/session is created.
 */
type OrtWasmEnv = { wasmPaths?: { mjs: string; wasm: string }; proxy?: boolean };
let ortConfigured = false;
async function configureOrt(): Promise<void> {
  if (ortConfigured) return;
  const { env } = await import('@huggingface/transformers');
  const onnx = (env.backends as Record<string, unknown> | undefined)?.['onnx'] as { wasm?: OrtWasmEnv } | undefined;
  if (onnx?.wasm) {
    onnx.wasm.wasmPaths = {
      mjs: './ort/ort-wasm-simd-threaded.mjs',
      wasm: './ort/ort-wasm-simd-threaded.wasm',
    };
    onnx.wasm.proxy = false;
  }
  ortConfigured = true;
}

export class TransformersEmbedder implements Embedder {
  readonly id = `${EMBED_MODEL}@${EMBED_REVISION.slice(0, 8)}`;
  readonly dims = 384;
  private pipe: FeatureExtractionPipeline | null = null;

  async load(onProgress?: (fraction: number) => void): Promise<void> {
    if (this.pipe !== null) return;
    await configureOrt();
    const { pipeline } = await import('@huggingface/transformers');
    const p = await pipeline('feature-extraction', EMBED_MODEL, {
      dtype: 'q8',
      revision: EMBED_REVISION,
      progress_callback: (info: { status?: string; progress?: number }) => {
        if (info.status === 'progress' && typeof info.progress === 'number') onProgress?.(info.progress / 100);
      },
    });
    this.pipe = p as unknown as FeatureExtractionPipeline;
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    if (this.pipe === null) throw new Error('embedder not loaded');
    const out = await this.pipe(texts, { pooling: 'mean', normalize: true });
    return out.tolist().map((row) => Float32Array.from(row));
  }

  /** bge query-side embedding with the instruction prefix. */
  async embedQuery(text: string): Promise<Float32Array> {
    const [vec] = await this.embed([QUERY_INSTRUCTION + text]);
    return vec!;
  }
}

interface CrossModel {
  (inputs: unknown): Promise<{ logits: { data: ArrayLike<number> } }>;
}

export class CrossEncoderReranker implements Reranker {
  readonly id = `${RERANK_MODEL}@${RERANK_REVISION.slice(0, 8)}`;
  private tokenizer: ((query: string, options: Record<string, unknown>) => unknown) | null = null;
  private model: CrossModel | null = null;

  async load(onProgress?: (fraction: number) => void): Promise<void> {
    if (this.model !== null) return;
    await configureOrt();
    const { AutoTokenizer, AutoModelForSequenceClassification } = await import('@huggingface/transformers');
    const progress = (info: { status?: string; progress?: number }) => {
      if (info.status === 'progress' && typeof info.progress === 'number') onProgress?.(info.progress / 100);
    };
    this.tokenizer = (await AutoTokenizer.from_pretrained(RERANK_MODEL, {
      revision: RERANK_REVISION,
      progress_callback: progress,
    })) as typeof this.tokenizer;
    this.model = (await AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL, {
      dtype: 'q8',
      revision: RERANK_REVISION,
      progress_callback: progress,
    })) as unknown as CrossModel;
  }

  async score(query: string, passages: string[]): Promise<number[]> {
    if (this.tokenizer === null || this.model === null) throw new Error('reranker not loaded');
    const scores: number[] = [];
    for (const passage of passages) {
      const inputs = this.tokenizer(query, { text_pair: passage, truncation: true, max_length: 512 });
      const { logits } = await this.model(inputs);
      scores.push(Number(logits.data[0]));
    }
    return scores;
  }
}

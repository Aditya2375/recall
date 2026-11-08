# recall

Hybrid search you can take apart: BM25, dense vector search with a hand-rolled
HNSW index, reciprocal-rank and linear fusion, and a cross-encoder reranker -
all in TypeScript, all running in your browser. No server, no API keys, no
build-step magic between you and the numbers.

**Live demo: https://aditya2375.github.io/recall/**

The demo bundles a 50-document astronomy corpus with a 40-query labeled eval
set. Every model output it shows for the bundled corpus is real: the embedding
and reranker scores are precomputed with the exact pinned models the browser
would run, and CI regenerates them and fails if they drift.

## What you can do in the demo

- **Search** the bundled corpus across six retrieval configurations and read
  the per-stage latency breakdown (embed / BM25 / dense / fuse / total).
- **Compare** modes side by side on the same query, with per-document scores
  and ranks from each stage.
- **Evaluate** the full 40-query labeled set in the browser and watch the
  metrics table fill in - the same runner that prints the table below.
- **Explore HNSW**: an ef_search sweep chart shows the recall/latency tradeoff
  of the approximate index against brute force.
- **Upload your own documents** (.txt, .md, or the JSON format below). The
  embedding model downloads on first use and runs locally in a Web Worker;
  your documents never leave the browser (they persist in IndexedDB).

## Measured results

40 queries, 71 graded relevance judgments over the 50-document corpus,
k = 10. Reproduce with `node scripts/evaluate.mjs`.

| config                 | R@1  | R@5  | R@10 | MRR  | nDCG@10 |
|------------------------|------|------|------|------|---------|
| bm25                   | 55.8 | 75.0 | 81.7 | 86.5 | 81.9    |
| dense (flat)           | 53.7 | 88.3 | 95.0 | 89.5 | 87.3    |
| dense (hnsw ef=50)     | 53.7 | 88.3 | 95.0 | 89.5 | 87.3    |
| hybrid rrf             | 54.6 | 85.4 | 90.8 | 91.3 | 88.1    |
| hybrid linear a=0.5    | 60.8 | 86.7 | 93.3 | 94.9 | 91.4    |
| hybrid rrf + rerank    | 64.2 | 88.8 | 92.9 | 97.8 | 93.8    |

Honest wrinkles, visible in the demo rather than hidden:

- **Dense retrieval has the best recall at 10** (95.0). Hybrid RRF actually
  dilutes it here (90.8) because BM25's top ranks on paraphrase queries push
  irrelevant keyword matches into the fused list. Fusion is not a free lunch.
- **Linear fusion beats RRF on this corpus** once BM25's raw scores are
  min-max normalized against cosine similarity. RRF's rank-only view throws
  away score magnitude, and it shows.
- **The reranker wins where it matters**: +10 points of R@1 and +6.5 MRR over
  plain RRF. Cross-encoder attention over query and passage together catches
  what bi-encoder dot products miss.
- The corpus contains a deliberate near-duplicate pair ("Why stars twinkle"
  vs "Why planets do not twinkle"). BM25 alone ranks the wrong one first for
  "why do stars twinkle"; the dense and reranked configurations get it right.

## HNSW from scratch

The approximate index (src/core/hnsw.ts) is a from-scratch hierarchical
navigable small world graph: seeded RNG for deterministic builds, layered
skip-list structure, greedy layer descent with beam search. Measured recall@10
against brute-force flat search on 2,000 random 64-dimensional vectors:

| ef_search | 25   | 50   | 100  | 200  |
|-----------|------|------|------|------|
| recall    | 80.7 | 94.3 | 99.1 | 100  |

On the bundled corpus, ef=50 matches flat search exactly on all 40 eval
queries. The demo's ef sweep chart renders this tradeoff live.

## Models

Pinned to exact revisions, quantized, downloaded lazily (only for custom
queries or your own documents - never for the bundled demo):

- Embeddings: `Xenova/bge-small-en-v1.5` @ `ea104dac`, 384-dim, q8 ONNX,
  with the bge query instruction prefix applied to queries only.
- Reranker: `Xenova/ms-marco-MiniLM-L-6-v2` @ `a0914435`, q8 ONNX, raw
  single-logit scores (no softmax).

ONNX Runtime Web runs single-threaded from wasm files the repo ships itself
(docs/ort/), because GitHub Pages cannot send the cross-origin isolation
headers threaded wasm requires. Everything works under the page's strict
Content-Security-Policy.

## Architecture

```
src/core/    pure TypeScript, zero dependencies, unit tested
  tokenize.ts    Unicode-aware tokenizer
  bm25.ts        BM25 inverted index
  vector.ts      flat brute-force cosine index
  hnsw.ts        HNSW approximate index (from scratch)
  fusion.ts      reciprocal-rank + min-max linear fusion, deterministic ties
  chunk.ts       fixed-window and structure-aware chunkers, char offsets
  metrics.ts     precision, recall, MRR, AP, nDCG (graded)
  pipeline.ts    Retriever: chunk aggregation, candidate breadth, rerank head
  evaluate.ts    eval-set runner (used by CLI and the browser tab)
  answer.ts      extractive answer composer - quotes corpus sentences verbatim
  precomputed.ts loader + adapters for the bundled model artifacts
src/web/     browser shell
  worker.ts      Web Worker: all indexing, search, evaluation off-thread
  models.ts      live model adapters (lazy, pinned, self-hosted wasm)
  app.ts         UI: textContent-only rendering, IndexedDB corpora, uploads
scripts/     precompute / verify / evaluate / build-web / warm
docs/        the built site (what GitHub Pages serves)
```

The Web Worker keeps indexing and inference off the UI thread. Answers are
extractive: every sentence is quoted verbatim from a retrieved chunk with its
source linked, so the answer panel cannot hallucinate.

## Bring your own corpus

Upload .txt or .md files directly, or a JSON file in this shape:

```json
{
  "name": "my-corpus",
  "docs": [{ "id": "doc-1", "title": "Title", "text": "Full text..." }]
}
```

Import limits are enforced up front: 500 documents, 2,000,000 characters
total, strict id/title validation. The first query downloads the embedding
model (~33 MB); after that everything, including the index, lives in your
browser's IndexedDB.

## Develop and verify

```sh
npm ci
npm run build                  # typecheck + compile core and web
npm test                       # 68 unit tests
node tests/browser-smoke.mjs   # 8 end-to-end checks in headless Chrome
node scripts/evaluate.mjs      # prints the metrics table above
node scripts/precompute.mjs    # regenerate the bundled model artifacts
node scripts/verify-precomputed.mjs  # fails if the artifacts drifted
node scripts/build-web.mjs     # rebuild docs/ exactly as CI checks it
sh tests/check-docs.sh         # README claims must match the code
```

CI runs all of the above plus a docs/ rebuild diff check, dependency audit,
and secret scanning.

## Security

No backend, no cookies, no analytics, no keys. The page ships a strict CSP
(`default-src 'self'`, model downloads allowed only from the pinned Hugging
Face hosts), renders all corpus and query text through `textContent` (never
`innerHTML`), validates every uploaded corpus against hard limits, and keeps
user data in IndexedDB on the user's machine only. See SECURITY_REVIEW.md for
the full item-by-item review.

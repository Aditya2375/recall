/**
 * Reranker interface. The neural cross-encoder (ms-marco-MiniLM, raw
 * logits) implements this in the web bundle and the precompute script;
 * core stays model-free so tests and CI never download anything.
 */

export interface Reranker {
  readonly id: string;
  score(query: string, passages: string[]): Promise<number[]>;
}

/** Stable descending sort by score, ties broken by key ascending. */
export function sortByScore<T>(items: T[], scores: number[], key: (item: T) => string): T[] {
  if (items.length !== scores.length) throw new Error('rerank: items and scores length mismatch');
  return items
    .map((item, i) => ({ item, score: scores[i]! }))
    .sort((a, z) => z.score - a.score || (key(a.item) < key(z.item) ? -1 : 1))
    .map((x) => x.item);
}

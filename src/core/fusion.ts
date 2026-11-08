/**
 * Rank fusion. Two methods, both deterministic:
 *
 * - Reciprocal rank fusion (Cormack et al. 2009): score = sum 1 / (k + rank),
 *   ranks 1-based. Needs no score calibration, which is why it is the
 *   default for mixing BM25 with cosine similarity.
 * - Weighted linear fusion over min-max normalized scores. When every score
 *   in a list is identical, each entry normalizes to 1 (they are equally
 *   good; mapping them all to 0 would erase the list).
 *
 * Every map returned is sorted by score descending, ties broken by id
 * ascending, so downstream ranks never depend on insertion order.
 */

export function sortScored(map: Map<string, number>): Array<[string, number]> {
  return [...map.entries()].sort((a, z) => z[1] - a[1] || (a[0] < z[0] ? -1 : a[0] > z[0] ? 1 : 0));
}

export function rrf(lists: Array<ReadonlyArray<string>>, k = 60): Map<string, number> {
  if (!(k > 0)) throw new Error('fusion: rrf k must be > 0');
  const out = new Map<string, number>();
  for (const ids of lists) {
    const seenInList = new Set<string>();
    let rank = 0;
    for (const id of ids) {
      if (seenInList.has(id)) continue;
      seenInList.add(id);
      rank += 1;
      out.set(id, (out.get(id) ?? 0) + 1 / (k + rank));
    }
  }
  return new Map(sortScored(out));
}

export interface WeightedList {
  ids: ReadonlyArray<string>;
  scores: ReadonlyMap<string, number>;
  weight: number;
}

export function linear(lists: WeightedList[]): Map<string, number> {
  const out = new Map<string, number>();
  for (const list of lists) {
    if (!(list.weight >= 0)) throw new Error('fusion: weights must be >= 0');
    let min = Infinity;
    let max = -Infinity;
    const seen = new Set<string>();
    for (const id of list.ids) {
      if (seen.has(id)) continue;
      seen.add(id);
      const s = list.scores.get(id);
      if (s === undefined) continue;
      if (s < min) min = s;
      if (s > max) max = s;
    }
    if (!Number.isFinite(min)) continue;
    const span = max - min;
    for (const id of seen) {
      const s = list.scores.get(id)!;
      const norm = span === 0 ? 1 : (s - min) / span;
      out.set(id, (out.get(id) ?? 0) + list.weight * norm);
    }
  }
  return new Map(sortScored(out));
}

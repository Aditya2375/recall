/**
 * Ranking metrics over graded relevance judgments (grade 0 = irrelevant,
 * 1 = relevant, 2 = highly relevant). Binary metrics treat grade > 0 as
 * relevant. Queries with no relevant document at all are excluded from
 * aggregates by `aggregate`, matching standard IR evaluation practice.
 */

export type Judgments = ReadonlyMap<string, number>;

function relevantSet(judgments: Judgments): Set<string> {
  const out = new Set<string>();
  for (const [id, grade] of judgments) if (grade > 0) out.add(id);
  return out;
}

export function precisionAtK(ranked: ReadonlyArray<string>, judgments: Judgments, k: number): number {
  if (k <= 0) return 0;
  const rel = relevantSet(judgments);
  let hits = 0;
  for (let i = 0; i < Math.min(k, ranked.length); i++) {
    if (rel.has(ranked[i]!)) hits += 1;
  }
  return hits / k;
}

export function recallAtK(ranked: ReadonlyArray<string>, judgments: Judgments, k: number): number {
  const rel = relevantSet(judgments);
  if (rel.size === 0) return 0;
  let hits = 0;
  for (let i = 0; i < Math.min(k, ranked.length); i++) {
    if (rel.has(ranked[i]!)) hits += 1;
  }
  return hits / rel.size;
}

export function reciprocalRank(ranked: ReadonlyArray<string>, judgments: Judgments): number {
  const rel = relevantSet(judgments);
  for (let i = 0; i < ranked.length; i++) {
    if (rel.has(ranked[i]!)) return 1 / (i + 1);
  }
  return 0;
}

export function averagePrecision(ranked: ReadonlyArray<string>, judgments: Judgments): number {
  const rel = relevantSet(judgments);
  if (rel.size === 0) return 0;
  let hits = 0;
  let sum = 0;
  for (let i = 0; i < ranked.length; i++) {
    if (rel.has(ranked[i]!)) {
      hits += 1;
      sum += hits / (i + 1);
    }
  }
  return sum / rel.size;
}

function dcg(grades: number[]): number {
  let sum = 0;
  for (let i = 0; i < grades.length; i++) {
    sum += (Math.pow(2, grades[i]!) - 1) / Math.log2(i + 2);
  }
  return sum;
}

export function ndcgAtK(ranked: ReadonlyArray<string>, judgments: Judgments, k: number): number {
  if (k <= 0) return 0;
  const actual: number[] = [];
  for (let i = 0; i < Math.min(k, ranked.length); i++) {
    actual.push(judgments.get(ranked[i]!) ?? 0);
  }
  const ideal = [...judgments.values()].sort((a, z) => z - a).slice(0, k);
  const idcg = dcg(ideal);
  if (idcg === 0) return 0;
  return dcg(actual) / idcg;
}

export interface QueryMetrics {
  query: string;
  precision: Record<number, number>;
  recall: Record<number, number>;
  rr: number;
  ap: number;
  ndcg: Record<number, number>;
  latencyMs: number;
}

const KS = [1, 5, 10] as const;

export function measureQuery(
  query: string,
  ranked: ReadonlyArray<string>,
  judgments: Judgments,
  latencyMs: number,
): QueryMetrics {
  const precision: Record<number, number> = {};
  const recall: Record<number, number> = {};
  const ndcg: Record<number, number> = {};
  for (const k of KS) {
    precision[k] = precisionAtK(ranked, judgments, k);
    recall[k] = recallAtK(ranked, judgments, k);
    ndcg[k] = ndcgAtK(ranked, judgments, k);
  }
  return { query, precision, recall, rr: reciprocalRank(ranked, judgments), ap: averagePrecision(ranked, judgments), ndcg, latencyMs };
}

export interface AggregateMetrics {
  queries: number;
  skipped: number;
  precision: Record<number, number>;
  recall: Record<number, number>;
  mrr: number;
  map: number;
  ndcg: Record<number, number>;
  latencyP50: number;
  latencyP95: number;
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[idx]!;
}

export function aggregate(rows: QueryMetrics[], judgmentsByQuery: ReadonlyMap<string, Judgments>): AggregateMetrics {
  const precision: Record<number, number> = { 1: 0, 5: 0, 10: 0 };
  const recall: Record<number, number> = { 1: 0, 5: 0, 10: 0 };
  const ndcg: Record<number, number> = { 1: 0, 5: 0, 10: 0 };
  let mrr = 0;
  let map = 0;
  let used = 0;
  let skipped = 0;
  const latencies: number[] = [];
  for (const row of rows) {
    const judgments = judgmentsByQuery.get(row.query);
    const hasRelevant = judgments !== undefined && [...judgments.values()].some((g) => g > 0);
    if (!hasRelevant) {
      skipped += 1;
      continue;
    }
    used += 1;
    for (const k of KS) {
      precision[k]! += row.precision[k]!;
      recall[k]! += row.recall[k]!;
      ndcg[k]! += row.ndcg[k]!;
    }
    mrr += row.rr;
    map += row.ap;
    latencies.push(row.latencyMs);
  }
  latencies.sort((a, z) => a - z);
  if (used > 0) {
    for (const k of KS) {
      precision[k]! /= used;
      recall[k]! /= used;
      ndcg[k]! /= used;
    }
    mrr /= used;
    map /= used;
  }
  return {
    queries: used,
    skipped,
    precision,
    recall,
    mrr,
    map,
    ndcg,
    latencyP50: percentile(latencies, 50),
    latencyP95: percentile(latencies, 95),
  };
}

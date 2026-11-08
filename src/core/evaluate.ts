/**
 * Evaluation runner: runs a set of retrieval configurations over a labeled
 * query set and produces per-query rows plus aggregates for each config.
 * This is the same code path that produces the numbers in the README and
 * the live Eval view - one runner, one truth.
 */

import { aggregate, measureQuery, type AggregateMetrics, type Judgments, type QueryMetrics } from './metrics.js';

export interface EvalQuery {
  id: string;
  text: string;
  /** docId -> grade (0, 1 or 2) */
  judgments: Record<string, number>;
}

export interface EvalSet {
  version: 1;
  corpus: string;
  queries: EvalQuery[];
}

export function validateEvalSet(input: unknown, docIds: ReadonlySet<string>): EvalSet {
  function fail(reason: string): never {
    throw new Error(`evalset: ${reason}`);
  }
  if (typeof input !== 'object' || input === null) fail('not an object');
  const raw = input as Record<string, unknown>;
  if (raw['version'] !== 1) fail('version must be 1');
  if (typeof raw['corpus'] !== 'string' || raw['corpus'] === '') fail('corpus must be a non-empty string');
  if (!Array.isArray(raw['queries']) || raw['queries'].length === 0) fail('queries must be a non-empty array');
  const queriesRaw = raw['queries'] as unknown[];
  const queries: EvalQuery[] = [];
  const seenIds = new Set<string>();
  for (let i = 0; i < queriesRaw.length; i++) {
    const elem = queriesRaw[i];
    if (typeof elem !== 'object' || elem === null) fail(`queries[${i}] is not an object`);
    const q = elem as Record<string, unknown>;
    const id = q['id'];
    const text = q['text'];
    const judgmentsRaw = q['judgments'];
    if (typeof id !== 'string' || id === '') fail(`queries[${i}].id missing`);
    const qid: string = id;
    if (seenIds.has(qid)) fail(`duplicate query id "${qid}"`);
    seenIds.add(qid);
    if (typeof text !== 'string' || text.trim() === '') fail(`queries[${i}].text missing`);
    const qtext: string = text;
    if (typeof judgmentsRaw !== 'object' || judgmentsRaw === null) fail(`queries[${i}].judgments missing`);
    const judgments: Record<string, number> = {};
    for (const [docId, grade] of Object.entries(judgmentsRaw as Record<string, unknown>)) {
      if (!docIds.has(docId)) fail(`queries[${i}] judges unknown document "${docId}"`);
      if (typeof grade !== 'number' || (grade !== 0 && grade !== 1 && grade !== 2)) {
        fail(`queries[${i}] grade for "${docId}" must be 0, 1 or 2`);
      }
      judgments[docId] = grade;
    }
    if (Object.keys(judgments).length === 0) fail(`queries[${i}] has no judgments`);
    queries.push({ id: qid, text: qtext.trim(), judgments });
  }
  const corpusName = raw['corpus'];
  return { version: 1, corpus: corpusName as string, queries };
}

export interface ConfigRun {
  ranked: string[];
  latencyMs: number;
}

export interface EvalConfig {
  name: string;
  run: (query: string) => Promise<ConfigRun>;
}

export interface ConfigReport {
  name: string;
  rows: QueryMetrics[];
  aggregate: AggregateMetrics;
}

export async function runEvaluation(configs: EvalConfig[], evalSet: EvalSet): Promise<ConfigReport[]> {
  const judgmentsByQuery = new Map<string, Judgments>();
  for (const q of evalSet.queries) judgmentsByQuery.set(q.text, new Map(Object.entries(q.judgments)));

  const reports: ConfigReport[] = [];
  for (const config of configs) {
    const rows: QueryMetrics[] = [];
    for (const q of evalSet.queries) {
      const { ranked, latencyMs } = await config.run(q.text);
      rows.push(measureQuery(q.text, ranked, judgmentsByQuery.get(q.text)!, latencyMs));
    }
    reports.push({ name: config.name, rows, aggregate: aggregate(rows, judgmentsByQuery) });
  }
  return reports;
}

/**
 * Main-thread UI controller. All heavy work lives in the worker; this file
 * only renders. Rule zero: corpus text, query text and model output are
 * rendered with textContent and createElement - never innerHTML.
 */

import { tokenize, tokenizeWithSpans } from '../core/tokenize.js';
import type { QueryResult, RankedDoc } from '../core/pipeline.js';
import type { Chunk } from '../core/chunk.js';
import type { CorpusDoc } from '../core/corpus.js';
import { validateCorpus, IMPORT_LIMITS } from '../core/corpus.js';
import type { ConfigReport } from '../core/evaluate.js';
import type { BM25Explanation } from '../core/bm25.js';
import type { ComposedAnswer } from '../core/answer.js';

interface CorpusState {
  id: string;
  docs: CorpusDoc[];
  chunks: Chunk[];
  stats: Record<string, number | string>;
}

type ProgressFn = (stage: string, fraction: number) => void;

class WorkerClient {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, { resolve: (v: never) => void; reject: (e: Error) => void; onProgress?: ProgressFn }>();

  constructor() {
    this.worker = new Worker('./worker.js');
    this.worker.onmessage = (event: MessageEvent) => {
      const msg = event.data as { kind: string; id?: number; message?: string };
      if (msg.kind === 'progress') {
        const p = this.pending.get(msg.id ?? 0);
        p?.onProgress?.((event.data as { stage: string }).stage, (event.data as { fraction: number }).fraction);
        return;
      }
      const id = msg.id ?? 0;
      const entry = this.pending.get(id);
      if (entry === undefined) return;
      this.pending.delete(id);
      if (msg.kind === 'error') entry.reject(new Error(msg.message ?? 'worker error'));
      else entry.resolve(event.data as never);
    };
    this.worker.onerror = (event) => {
      for (const [, entry] of this.pending) entry.reject(new Error(event.message));
      this.pending.clear();
    };
  }

  request<T>(message: Record<string, unknown>, onProgress?: ProgressFn): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      const entry: { resolve: (v: never) => void; reject: (e: Error) => void; onProgress?: ProgressFn } = { resolve, reject };
      if (onProgress !== undefined) entry.onProgress = onProgress;
      this.pending.set(id, entry);
      this.worker.postMessage({ ...message, id });
    });
  }
}

function el(tag: string, className?: string, text?: string): HTMLElement {
  const node = document.createElement(tag);
  if (className !== undefined) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const client = new WorkerClient();
const corpora = new Map<string, CorpusState>();
let activeCorpus = '';
let evalSetQueries: Array<{ id: string; text: string; judgments: Record<string, number> }> = [];
let evalJudgmentsByText = new Map<string, Record<string, number>>();
let liveEmbedderLoaded = false;
let liveRerankerLoaded = false;

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;

function setModelPill(): void {
  const pill = $('model-pill');
  const parts: string[] = [];
  parts.push(liveEmbedderLoaded ? 'live embedding model loaded' : 'instant mode: precomputed vectors');
  if (liveRerankerLoaded) parts.push('reranker loaded');
  pill.textContent = parts.join(' · ');
  pill.classList.toggle('live', liveEmbedderLoaded);
}

function showProgress(prefix: string): ProgressFn {
  const bar = $(prefix + '-bar') as HTMLElement;
  const label = $(prefix + '-label');
  $(prefix).classList.remove('hidden');
  return (stage, fraction) => {
    bar.style.width = `${Math.round(fraction * 100)}%`;
    label.textContent = stage;
  };
}

function hideProgress(prefix: string): void {
  $(prefix).classList.add('hidden');
}

function highlightInto(container: HTMLElement, text: string, query: string): void {
  const qTerms = new Set(tokenize(query));
  if (qTerms.size === 0) {
    container.textContent = text;
    return;
  }
  const spans = tokenizeWithSpans(text);
  let cursor = 0;
  for (const span of spans) {
    if (span.start > cursor) container.appendChild(document.createTextNode(text.slice(cursor, span.start)));
    if (qTerms.has(span.text)) {
      const mark = document.createElement('mark');
      mark.textContent = text.slice(span.start, span.end);
      container.appendChild(mark);
    } else {
      container.appendChild(document.createTextNode(text.slice(span.start, span.end)));
    }
    cursor = span.end;
  }
  if (cursor < text.length) container.appendChild(document.createTextNode(text.slice(cursor)));
}

function fmt(x: number | undefined, digits = 3): string {
  return x === undefined ? '-' : x.toFixed(digits);
}

// ---------- tabs ----------

for (const button of document.querySelectorAll<HTMLButtonElement>('#tabs button')) {
  button.addEventListener('click', () => {
    for (const b of document.querySelectorAll('#tabs button')) b.classList.remove('active');
    for (const s of document.querySelectorAll('.tab')) s.classList.remove('active');
    button.classList.add('active');
    $(`tab-${button.dataset['tab'] ?? 'search'}`).classList.add('active');
  });
}

// ---------- search ----------

function currentQueryOptions(mode: string, rerank: boolean): Record<string, unknown> {
  const base = { k: 10, rerankTop: rerank ? 25 : 0 };
  switch (mode) {
    case 'bm25': return { ...base, dense: null };
    case 'dense': return { ...base, lexical: false, dense: 'flat' };
    case 'linear': return { ...base, fusion: 'linear', alpha: 0.5 };
    default: return { ...base, fusion: 'rrf' };
  }
}

function renderAnswer(answer: ComposedAnswer): void {
  const box = $('answer');
  box.replaceChildren();
  if (answer.sentences.length === 0) {
    box.classList.add('hidden');
    return;
  }
  box.classList.remove('hidden');
  box.appendChild(el('h4', undefined, 'Grounded answer - quoted from the corpus'));
  for (const s of answer.sentences) {
    const p = el('p');
    p.appendChild(document.createTextNode(s.text + ' '));
    const cite = el('a', 'cite', `[${s.citation}]`);
    cite.setAttribute('href', '#');
    cite.addEventListener('click', (e) => {
      e.preventDefault();
      document.querySelector(`[data-chunk="${s.chunkId}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
    p.appendChild(cite);
    box.appendChild(p);
  }
}

async function attachExplain(details: HTMLElement, chunkId: string, query: string): Promise<void> {
  const reply = await client.request<{ explanation: BM25Explanation | null }>({ kind: 'explain', corpusId: activeCorpus, chunkId, query });
  const ex = reply.explanation;
  if (ex === null) {
    details.appendChild(el('p', 'hint', 'No BM25 breakdown for this chunk.'));
    return;
  }
  const table = el('table', 'term-table');
  const head = el('tr');
  for (const h of ['term', 'idf', 'tf', 'score']) head.appendChild(el('th', undefined, h));
  table.appendChild(head);
  for (const t of ex.terms) {
    const row = el('tr');
    row.appendChild(el('td', undefined, t.term));
    row.appendChild(el('td', undefined, fmt(t.idf)));
    row.appendChild(el('td', undefined, String(t.tf)));
    row.appendChild(el('td', undefined, fmt(t.contribution)));
    table.appendChild(row);
  }
  details.appendChild(table);
}

function renderResults(result: QueryResult & { answer?: ComposedAnswer }, query: string): void {
  const list = $('results');
  list.replaceChildren();
  renderAnswer(result.answer ?? { sentences: [] });
  const l = result.latencyMs;
  $('results-meta').textContent =
    `${result.docs.length} documents · mode ${result.mode} · ` +
    `embed ${l.embed.toFixed(0)}ms · bm25 ${l.lexical.toFixed(1)}ms · dense ${l.dense.toFixed(1)}ms · fuse ${l.fusion.toFixed(1)}ms` +
    (l.rerank > 0 ? ` · rerank ${l.rerank.toFixed(0)}ms` : '') + ` · total ${l.total.toFixed(1)}ms`;
  const state = corpora.get(activeCorpus)!;
  for (const doc of result.docs) {
    renderResult(list, doc, state, query);
  }
}

function renderResult(list: HTMLOListElement, doc: RankedDoc, state: CorpusState, query: string): void {
  const meta = state.docs.find((d) => d.id === doc.docId);
  const chunk = state.chunks.find((c) => c.id === doc.chunkId);
  const li = el('li');
  li.setAttribute('data-chunk', doc.chunkId);
  const head = el('div', 'head');
  head.appendChild(el('span', 'title', `${doc.rank}. ${meta?.title ?? doc.docId}`));
  const bits = [
    doc.scores.bm25 !== undefined ? `bm25 ${fmt(doc.scores.bm25, 2)} #${doc.ranks.bm25}` : null,
    doc.scores.dense !== undefined ? `dense ${fmt(doc.scores.dense)} #${doc.ranks.dense}` : null,
    `fused ${fmt(doc.scores.fused, 4)}`,
    doc.scores.rerank !== undefined ? `rerank ${fmt(doc.scores.rerank, 2)}` : null,
  ].filter((b): b is string => b !== null);
  head.appendChild(el('span', 'scoreline', bits.join(' · ')));
  li.appendChild(head);
  if (chunk !== undefined) {
    const snippet = el('div', 'snippet');
    const text = chunk.text.length > 240 ? chunk.text.slice(0, 240) + '...' : chunk.text;
    highlightInto(snippet, text, query);
    li.appendChild(snippet);
    const details = el('details');
    const summary = el('summary', undefined, `chunk ${chunk.id}${chunk.headings.length > 0 ? ' · ' + chunk.headings.join(' / ') : ''}`);
    details.appendChild(summary);
    const full = el('div', 'chunk-full');
    full.textContent = chunk.text;
    details.appendChild(full);
    let loaded = false;
    details.addEventListener('toggle', () => {
      if (details instanceof HTMLDetailsElement && details.open && !loaded) {
        loaded = true;
        void attachExplain(details, chunk.id, query);
      }
    });
    li.appendChild(details);
  }
  list.appendChild(li);
}

async function runSearch(): Promise<void> {
  const query = $('query') as HTMLInputElement;
  const text = query.value.trim();
  if (text === '') return;
  const mode = (document.querySelector('input[name="mode"]:checked') as HTMLInputElement).value;
  const rerank = ($('rerank') as HTMLInputElement).checked;
  $('run').setAttribute('disabled', '');
  const progress = showProgress('progress');
  try {
    const reply = await client.request<{ result: QueryResult & { answer?: ComposedAnswer }; modelsUsed: string[] }>(
      { kind: 'query', corpusId: activeCorpus, text, options: currentQueryOptions(mode, rerank) },
      progress,
    );
    if (reply.modelsUsed.includes('embed')) liveEmbedderLoaded = true;
    if (reply.modelsUsed.includes('rerank')) liveRerankerLoaded = true;
    setModelPill();
    renderResults(reply.result, text);
  } catch (err) {
    $('results-meta').textContent = err instanceof Error ? err.message : String(err);
  } finally {
    hideProgress('progress');
    $('run').removeAttribute('disabled');
  }
}

// ---------- compare ----------

async function runCompare(): Promise<void> {
  const input = $('compare-query') as HTMLInputElement;
  const text = input.value.trim();
  if (text === '') return;
  const grid = $('compare-grid');
  grid.replaceChildren();
  const configs: Array<{ label: string; options: Record<string, unknown> }> = [
    { label: 'BM25', options: { k: 10, dense: null } },
    { label: 'Dense (flat)', options: { k: 10, lexical: false, dense: 'flat' } },
    { label: 'Hybrid RRF', options: { k: 10, fusion: 'rrf' } },
    { label: 'Hybrid RRF + rerank', options: { k: 10, fusion: 'rrf', rerankTop: 25 } },
  ];
  const judgments = evalJudgmentsByText.get(text);
  for (const config of configs) {
    const col = el('div', 'compare-col');
    col.appendChild(el('h4', undefined, config.label));
    try {
      const reply = await client.request<{ result: QueryResult; modelsUsed: string[] }>(
        { kind: 'query', corpusId: activeCorpus, text, options: config.options },
      );
      if (reply.modelsUsed.includes('embed')) liveEmbedderLoaded = true;
      if (reply.modelsUsed.includes('rerank')) liveRerankerLoaded = true;
      setModelPill();
      const state = corpora.get(activeCorpus)!;
      const ol = el('ol');
      for (const doc of reply.result.docs) {
        const meta = state.docs.find((d) => d.id === doc.docId);
        const li = el('li', undefined, meta?.title ?? doc.docId);
        const grade = judgments?.[doc.docId];
        if (grade === 2) li.appendChild(el('span', 'mark2', ' ●'));
        else if (grade === 1) li.appendChild(el('span', 'mark1', ' ◐'));
        ol.appendChild(li);
      }
      col.appendChild(ol);
    } catch (err) {
      col.appendChild(el('p', 'hint', err instanceof Error ? err.message : String(err)));
    }
    grid.appendChild(col);
  }
}

// ---------- evaluate ----------

function renderEvalTable(reports: ConfigReport[]): void {
  const box = $('eval-table');
  box.replaceChildren();
  const table = el('table', 'eval');
  const head = el('tr');
  for (const h of ['configuration', 'R@1', 'R@5', 'R@10', 'MRR', 'nDCG@10', 'p50 ms', 'p95 ms']) {
    head.appendChild(el('th', undefined, h));
  }
  table.appendChild(head);
  const best = {
    r1: Math.max(...reports.map((r) => r.aggregate.recall[1]!)),
    r5: Math.max(...reports.map((r) => r.aggregate.recall[5]!)),
    r10: Math.max(...reports.map((r) => r.aggregate.recall[10]!)),
    mrr: Math.max(...reports.map((r) => r.aggregate.mrr)),
    ndcg: Math.max(...reports.map((r) => r.aggregate.ndcg[10]!)),
  };
  for (const r of reports) {
    const a = r.aggregate;
    const row = el('tr');
    const cells: Array<[string, number]> = [
      [r.name, NaN],
      [(a.recall[1]! * 100).toFixed(1), a.recall[1]!],
      [(a.recall[5]! * 100).toFixed(1), a.recall[5]!],
      [(a.recall[10]! * 100).toFixed(1), a.recall[10]!],
      [(a.mrr * 100).toFixed(1), a.mrr],
      [(a.ndcg[10]! * 100).toFixed(1), a.ndcg[10]!],
      [a.latencyP50.toFixed(2), NaN],
      [a.latencyP95.toFixed(2), NaN],
    ];
    const isBest = a.recall[1] === best.r1 || a.mrr === best.mrr;
    if (isBest) row.className = 'best';
    for (const [text] of cells) row.appendChild(el('td', undefined, text));
    table.appendChild(row);
  }
  box.appendChild(table);
  box.appendChild(el('p', 'hint',
    `${reports[0]?.aggregate.queries ?? 0} queries with relevance judgments. Green rows lead at least one metric. ` +
    'Percentages. The hybrid rows do not win every column on this corpus - that honesty is the point of showing all six.'));
}

async function runEval(): Promise<void> {
  $('eval-run').setAttribute('disabled', '');
  const progress = showProgress('eval-progress');
  try {
    const reply = await client.request<{ reports: ConfigReport[] }>({ kind: 'eval', corpusId: activeCorpus }, progress);
    renderEvalTable(reply.reports);
  } catch (err) {
    $('eval-table').replaceChildren(el('p', 'hint', err instanceof Error ? err.message : String(err)));
  } finally {
    hideProgress('eval-progress');
    $('eval-run').removeAttribute('disabled');
  }
}

function drawEfChart(points: Array<{ ef: number; recall: number; latencyMs: number }>): void {
  const canvas = $('ef-chart') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  const padL = 56;
  const padB = 40;
  const padT = 16;
  const padR = 16;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const maxLat = Math.max(...points.map((p) => p.latencyMs), 0.001);
  const x = (i: number) => padL + (points.length === 1 ? plotW / 2 : (i / (points.length - 1)) * plotW);
  const yRecall = (r: number) => padT + (1 - r) * plotH;
  ctx.strokeStyle = '#2d333f';
  ctx.fillStyle = '#9aa4b2';
  ctx.font = '11px ui-monospace, monospace';
  for (let g = 0; g <= 4; g++) {
    const gy = padT + (g / 4) * plotH;
    ctx.beginPath();
    ctx.moveTo(padL, gy);
    ctx.lineTo(width - padR, gy);
    ctx.stroke();
    ctx.fillText(`${(100 - g * 5)}%`, 8, gy + 4);
  }
  ctx.strokeStyle = '#4fa3ff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((p, i) => {
    if (i === 0) ctx.moveTo(x(i), yRecall(p.recall));
    else ctx.lineTo(x(i), yRecall(p.recall));
  });
  ctx.stroke();
  ctx.fillStyle = '#4fa3ff';
  points.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(x(i), yRecall(p.recall), 4, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.fillStyle = '#9aa4b2';
  points.forEach((p, i) => {
    ctx.fillText(`ef=${p.ef}`, x(i) - 16, height - 22);
    ctx.fillText(`${p.latencyMs.toFixed(2)}ms`, x(i) - 16, height - 8);
  });
  ctx.fillText(`max ${maxLat.toFixed(2)}ms per query`, padL, padT - 4);
}

async function runEfSweep(): Promise<void> {
  $('ef-run').setAttribute('disabled', '');
  try {
    const reply = await client.request<{ points: Array<{ ef: number; recall: number; latencyMs: number }> }>(
      { kind: 'efSweep', corpusId: activeCorpus },
    );
    drawEfChart(reply.points);
  } finally {
    $('ef-run').removeAttribute('disabled');
  }
}

// ---------- corpus ----------

function renderCorpusStats(): void {
  const state = corpora.get(activeCorpus)!;
  const box = $('corpus-stats');
  box.replaceChildren();
  for (const [key, label] of [['docs', 'documents'], ['chunks', 'chunks'], ['vocab', 'vocabulary'], ['avgChunkTokens', 'avg tokens/chunk'], ['embedder', 'embedder']] as const) {
    const card = el('div', 'card');
    card.appendChild(el('b', undefined, String(state.stats[key] ?? '-')));
    card.appendChild(el('span', undefined, label));
    box.appendChild(card);
  }
}

function renderDocList(): void {
  const state = corpora.get(activeCorpus)!;
  const list = $('doc-list');
  list.replaceChildren();
  for (const doc of state.docs) {
    const b = el('button', undefined, doc.title);
    b.addEventListener('click', () => {
      for (const other of list.querySelectorAll('button')) other.classList.remove('active');
      b.classList.add('active');
      renderDocView(doc.id);
    });
    list.appendChild(b);
  }
}

function renderDocView(docId: string): void {
  const state = corpora.get(activeCorpus)!;
  const doc = state.docs.find((d) => d.id === docId)!;
  const view = $('doc-view');
  view.replaceChildren();
  view.appendChild(el('h3', undefined, doc.title));
  const chunks = state.chunks.filter((c) => c.docId === docId);
  for (const chunk of chunks) {
    const div = el('div', 'chunk');
    div.appendChild(el('div', 'cid', `${chunk.id}${chunk.headings.length > 0 ? ' · ' + chunk.headings.join(' / ') : ''} · chars ${chunk.start}-${chunk.end}`));
    div.appendChild(el('div', undefined, chunk.text));
    view.appendChild(div);
  }
}

function slugify(name: string, taken: Set<string>): string {
  const base = name.replace(/\.[^.]*$/, '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'doc';
  let slug = base;
  let i = 2;
  while (taken.has(slug)) slug = `${base}-${i++}`;
  return slug;
}

async function handleUpload(files: FileList): Promise<void> {
  const progress = showProgress('progress');
  try {
    let corpus;
    if (files.length === 1 && files[0]!.name.endsWith('.json')) {
      corpus = validateCorpus(JSON.parse(await files[0]!.text()));
    } else {
      const docs = [];
      const taken = new Set<string>();
      let total = 0;
      for (const file of Array.from(files)) {
        const text = await file.text();
        total += text.length;
        if (total > IMPORT_LIMITS.maxTotalChars) throw new Error(`uploads exceed ${IMPORT_LIMITS.maxTotalChars} characters total`);
        const id = slugify(file.name, taken);
        taken.add(id);
        docs.push({ id, title: file.name, text });
      }
      corpus = validateCorpus({ version: 1, name: `upload-${Date.now().toString(36)}`, docs });
    }
    const reply = await client.request<{ corpusId: string; stats: Record<string, number | string>; chunks: Chunk[]; docs: CorpusDoc[] }>(
      { kind: 'indexCorpus', corpus }, progress,
    );
    liveEmbedderLoaded = true;
    setModelPill();
    corpora.set(reply.corpusId, { id: reply.corpusId, docs: reply.docs, chunks: reply.chunks, stats: reply.stats });
    addCorpusOption(reply.corpusId, `${corpus.name} (${reply.docs.length} docs)`);
    activeCorpus = reply.corpusId;
    ($('corpus-picker') as HTMLSelectElement).value = reply.corpusId;
    renderCorpusStats();
    renderDocList();
    saveUserCorpus(corpus);
  } catch (err) {
    alert(err instanceof Error ? err.message : String(err));
  } finally {
    hideProgress('progress');
  }
}

function exportCorpus(): void {
  const state = corpora.get(activeCorpus)!;
  const corpus = { version: 1, name: state.id, docs: state.docs };
  const blob = new Blob([JSON.stringify(corpus, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = el('a') as HTMLAnchorElement;
  a.href = url;
  a.download = `${state.id}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

// ---------- IndexedDB persistence (user corpora, documents only) ----------

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('recall', 1);
    req.onupgradeneeded = () => req.result.createObjectStore('corpora', { keyPath: 'name' });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function saveUserCorpus(corpus: { name: string; docs: CorpusDoc[] }): void {
  void openDb().then((db) => {
    const tx = db.transaction('corpora', 'readwrite');
    tx.objectStore('corpora').put(corpus);
  });
}

async function loadUserCorpora(): Promise<Array<{ name: string; docs: CorpusDoc[] }>> {
  try {
    const db = await openDb();
    return await new Promise((resolve, reject) => {
      const req = db.transaction('corpora').objectStore('corpora').getAll();
      req.onsuccess = () => resolve(req.result as Array<{ name: string; docs: CorpusDoc[] }>);
      req.onerror = () => reject(req.error);
    });
  } catch {
    return [];
  }
}

// ---------- boot ----------

function addCorpusOption(id: string, label: string): void {
  const picker = $('corpus-picker') as HTMLSelectElement;
  const option = document.createElement('option');
  option.value = id;
  option.textContent = label;
  picker.appendChild(option);
}

async function boot(): Promise<void> {
  const [corpusRaw, evalRaw, preRaw] = await Promise.all([
    fetch('./data/corpus/night-sky.json').then((r) => r.json()),
    fetch('./data/eval/night-sky-queries.json').then((r) => r.json()),
    fetch('./data/precomputed/night-sky.bge-small-en-v1.5.json').then((r) => r.json()),
  ]);
  const reply = await client.request<{ corpusId: string; stats: Record<string, number | string>; chunks: Chunk[]; docs: CorpusDoc[] }>(
    { kind: 'init', corpus: corpusRaw, evalSet: evalRaw, precomputed: preRaw },
  );
  corpora.set(reply.corpusId, { id: reply.corpusId, docs: reply.docs, chunks: reply.chunks, stats: reply.stats });
  activeCorpus = reply.corpusId;
  evalSetQueries = (evalRaw as { queries: Array<{ id: string; text: string; judgments: Record<string, number> }> }).queries;
  evalJudgmentsByText = new Map(evalSetQueries.map((q) => [q.text, q.judgments]));
  addCorpusOption(reply.corpusId, `night-sky (${reply.docs.length} docs, bundled)`);
  renderCorpusStats();
  renderDocList();
  setModelPill();

  for (const saved of await loadUserCorpora()) {
    if (corpora.has(saved.name)) continue;
    addCorpusOption(saved.name, `${saved.name} (${saved.docs.length} docs, saved - reindex on select)`);
  }
}

$('run').addEventListener('click', () => void runSearch());
($('query') as HTMLInputElement).addEventListener('keydown', (e) => { if (e.key === 'Enter') void runSearch(); });
$('compare-run').addEventListener('click', () => void runCompare());
($('compare-query') as HTMLInputElement).addEventListener('keydown', (e) => { if (e.key === 'Enter') void runCompare(); });
$('eval-run').addEventListener('click', () => void runEval());
$('ef-run').addEventListener('click', () => void runEfSweep());
$('upload-btn').addEventListener('click', () => ($('upload-input') as HTMLInputElement).click());
($('upload-input') as HTMLInputElement).addEventListener('change', (e) => {
  const files = (e.target as HTMLInputElement).files;
  if (files !== null && files.length > 0) void handleUpload(files);
  (e.target as HTMLInputElement).value = '';
});
$('export-btn').addEventListener('click', exportCorpus);
($('corpus-picker') as HTMLSelectElement).addEventListener('change', (e) => {
  activeCorpus = (e.target as HTMLSelectElement).value;
  if (corpora.has(activeCorpus)) {
    renderCorpusStats();
    renderDocList();
  }
});

void boot();

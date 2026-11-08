/**
 * Browser smoke test: serves docs/, drives headless Chrome over CDP, and
 * checks the demo actually works - boot, search, answer, evaluation table.
 * Runs locally and in CI (ubuntu-latest ships google-chrome). No npm deps.
 */
import { spawn } from 'node:child_process';

const HTTP_PORT = 8931;
const CDP_PORT = 9333;
const root = new URL('..', import.meta.url).pathname;

const server = spawn('python3', ['-m', 'http.server', String(HTTP_PORT), '--bind', '127.0.0.1', '--directory', `${root}docs`], { stdio: 'ignore' });
const chrome = spawn('google-chrome', [
  '--headless=new', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
  `--remote-debugging-port=${CDP_PORT}`, 'about:blank',
], { stdio: 'ignore' });

const cleanup = (code) => {
  try { chrome.kill('SIGKILL'); } catch {}
  try { server.kill('SIGKILL'); } catch {}
  process.exit(code);
};
setTimeout(() => { console.error('smoke: timeout'); cleanup(1); }, 120000);
process.on('SIGINT', () => cleanup(2));

async function waitFor(fn, label, timeoutMs = 45000) {
  const start = Date.now();
  for (;;) {
    try {
      const v = await fn();
      if (v) return v;
    } catch {}
    if (Date.now() - start > timeoutMs) throw new Error(`timed out waiting for ${label}`);
    await new Promise((r) => setTimeout(r, 250));
  }
}

const checks = [];
const check = (name, ok, detail = '') => {
  checks.push({ name, ok, detail });
  console.log(`${ok ? 'ok' : 'NOT OK'} - ${name}${detail ? ' - ' + detail : ''}`);
};

await waitFor(async () => (await fetch(`http://127.0.0.1:${CDP_PORT}/json/version`)).ok, 'chrome devtools');
const target = await (await fetch(`http://127.0.0.1:${CDP_PORT}/json/new`, { method: 'PUT' })).json();
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });

let mid = 0;
const rpcPending = new Map();
const consoleErrors = [];
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.id && rpcPending.has(msg.id)) {
    rpcPending.get(msg.id)(msg);
    rpcPending.delete(msg.id);
  } else if (msg.method === 'Runtime.exceptionThrown') {
    consoleErrors.push(msg.params.exceptionDetails?.exception?.description ?? 'page exception');
  }
};
const cdp = (method, params = {}) => new Promise((resolve) => {
  const id = ++mid;
  rpcPending.set(id, resolve);
  ws.send(JSON.stringify({ id, method, params }));
});
const evaluate = async (expression) => {
  const res = await cdp('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (res.result?.exceptionDetails) throw new Error(JSON.stringify(res.result.exceptionDetails));
  return res.result?.result?.value;
};

await cdp('Runtime.enable');
await cdp('Page.enable');
await cdp('Page.navigate', { url: `http://127.0.0.1:${HTTP_PORT}/` });

await waitFor(() => evaluate(`document.querySelectorAll('#corpus-stats .card').length > 0`), 'boot');
check('boot builds the bundled index', true);

// Semantic showcase: paraphrase query with no keyword overlap ("flicker"),
// covered by the bundled precomputed vectors. Hybrid (RRF) must surface the
// twinkling article first.
await evaluate(`(() => { const q = document.querySelector('#query'); q.value = 'why does starlight seem to flicker on and off'; document.querySelector('#run').click(); })()`);
await waitFor(() => evaluate(`document.querySelectorAll('#results > li').length > 0`), 'search results');
const top = await evaluate(`document.querySelector('#results > li .title')?.textContent ?? ''`);
check('hybrid search ranks the twinkling article first for a paraphrase query', top.includes('Why stars twinkle'), top);
const answer = await evaluate(`(() => { const a = document.querySelector('#answer'); return { hidden: a.classList.contains('hidden'), sentences: a.querySelectorAll('p').length }; })()`);
check('grounded answer renders with quoted sentences', !answer.hidden && answer.sentences > 0, `${answer.sentences} sentences`);
const meta = await evaluate(`document.querySelector('#results-meta').textContent`);
check('latency breakdown is shown', /total .*ms/.test(meta), meta);

// Lexical showcase: rare-term keyword query where BM25 alone wins.
await evaluate(`(() => { document.querySelector('#mode-controls input[value="bm25"]').click(); const q = document.querySelector('#query'); q.value = 'Bortle scale'; document.querySelector('#run').click(); })()`);
await waitFor(() => evaluate(`document.querySelector('#results > li .title')?.textContent.includes('Light pollution')`), 'lexical search');
check('BM25 mode ranks the light pollution doc first for "Bortle scale"', true);
await evaluate(`document.querySelector('#mode-controls input[value="hybrid"]').click()`);

await evaluate(`document.querySelector('[data-tab="evaluate"]').click()`);
await evaluate(`document.querySelector('#eval-run').click()`);
await waitFor(() => evaluate(`document.querySelectorAll('#eval-table table.eval tr').length >= 7`), 'evaluation table', 90000);
const rows = await evaluate(`[...document.querySelectorAll('#eval-table table.eval tr')].slice(1).map(tr => tr.cells[0].textContent + ' R@10=' + tr.cells[3].textContent)`);
check('evaluation table has all six configurations', rows.length === 6, rows.join(' | '));

const xssProbe = '<img src=x onerror=window.__xss=1>';
await evaluate(`(() => { const q = document.querySelector('#query'); q.value = ${JSON.stringify(xssProbe)}; document.querySelector('#run').click(); })()`);
await new Promise((r) => setTimeout(r, 1500));
const xss = await evaluate(`window.__xss === 1`);
check('query text cannot inject markup', !xss);

// Uploaded corpus with a hostile title must render as inert text too, and the
// whole upload -> index -> search path must work (live model not required for
// BM25-only search).
const hostile = JSON.stringify({
  version: 1, name: 'hostile-test',
  docs: [
    { id: 'a1', title: xssProbe + ' Zebra astronomy', text: 'Zebras are not astronomy, but this doc mentions quasars and pulsars for ranking.' },
    { id: 'b2', title: 'Plain doc', text: 'Pulsars spin rapidly and emit beams of radio waves.' },
  ],
});
await evaluate(`(() => {
  const input = document.querySelector('#upload-input');
  const dt = new DataTransfer();
  dt.items.add(new File([${JSON.stringify(hostile)}], 'hostile.json', { type: 'application/json' }));
  input.files = dt.files;
  input.dispatchEvent(new Event('change', { bubbles: true }));
})()`);
await waitFor(() => evaluate(`document.querySelector('#corpus-picker')?.textContent.includes('hostile-test')`), 'hostile corpus import');
await evaluate(`(() => { document.querySelector('#mode-controls input[value="bm25"]').click(); const q = document.querySelector('#query'); q.value = 'pulsars'; document.querySelector('#run').click(); })()`);
await waitFor(() => evaluate(`document.querySelector('#results > li .title')?.textContent.includes('Plain doc')`), 'search over uploaded corpus');
const xss2 = await evaluate(`window.__xss === 1`);
check('uploaded corpus works and its title cannot inject markup', !xss2);

check('no page exceptions during the whole run', consoleErrors.length === 0, consoleErrors[0] ?? '');

const failed = checks.filter((c) => !c.ok);
console.log(failed.length === 0 ? `smoke ok: ${checks.length} checks` : `smoke FAILED: ${failed.length}/${checks.length}`);
cleanup(failed.length === 0 ? 0 : 1);

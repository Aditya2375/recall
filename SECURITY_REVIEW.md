# Security review - recall

Item-by-item pass over the pre-ship checklist, with evidence. Statuses: PASS
(checked, evidence below), N/A (the thing the item protects does not exist in
this project - stated explicitly, not skipped silently), TODO (must happen
before or right after publication).

Architecture context for every item below: recall is a fully static site. No
backend, no database, no auth, no accounts, no cookies, no analytics, no API
keys, no forms. User-uploaded documents live only in the visitor's own
IndexedDB. The only network calls the page makes are same-origin (its own
files) and HTTPS model downloads from the pinned Hugging Face hosts named in
the CSP.

## Pre-ship routine

1. **Read the diff** - PASS. All source (src/, scripts/, tests/, docs/) was
   reviewed line by line; there is no generated or copied code outside
   docs/ort/ (vendored onnxruntime-web binaries) and the bundled worker.js,
   which is built from src/ and diff-checked in CI.
2. **gitleaks detect** - TODO in CI (workflow runs gitleaks on full history).
   Local evidence: no keys exist anywhere in the project by design (no
   backends to key), and a manual scan of src/, scripts/, tests/, README.md,
   data/ found no key/token/secret strings.
3. **Dependency audit** - PASS. `npm audit` and `npm audit --omit=dev`: 0
   vulnerabilities. Only two runtime-relevant packages:
   @huggingface/transformers (pinned 4.3.2) and its onnxruntime-web
   dependency; esbuild/typescript/tsc are dev-only. CI reruns npm audit.
4. **Static scan** - TODO in CI (semgrep community rules). Local evidence:
   manual review; the risky-pattern grep below is part of tests/check-docs.sh
   and CI.
5. **Logged-out test** - N/A. There is nothing to log in to and no storage to
   query. Every byte the site serves is public by design; each visitor's own
   IndexedDB is same-origin and unreachable by other origins by browser
   construction.
6. **MDN Observatory scan** - TODO after the URL is live. Known limit: GitHub
   Pages cannot send HTTP headers, so CSP is delivered via <meta> (see item
   17); the Observatory score will reflect that Pages limitation, accepted
   per the checklist's own Pages guidance.
7. **Push public only after checks** - PASS (process). Nothing was pushed
   before this review, check-docs, unit tests, browser smoke, and the
   precomputed-verify gate all passed locally.

## Secrets and keys

8. **(1) No secret in code, config, bundle or history** - PASS. The project
   has no secrets to leak: no API keys, no service accounts, no tokens. CI
   runs gitleaks over full git history. GitHub secret scanning + push
   protection to be enabled on the repo at creation (free on public repos).
9. **(2) Keys meant to be public** - N/A. No keys of any kind ship. DevTools
   search of the live bundle finds no key material (only model revision
   hashes, which are public identifiers).
10. **(3) Gitignore secrets before first commit** - PASS. .gitignore covers
    .env*, *.keystore, *.jks, local.properties, *service-account*.json since
    the first commit.
11. **(4) Rotate anything that ever leaked** - N/A. Nothing ever existed to
    leak.

## Access control and data

12. **(5) RLS on every table** - N/A. No database.
13. **(6) Storage buckets private by default** - N/A. No buckets/Firebase.
    Visitor uploads live in their own browser IndexedDB only.
14. **(7) Authorization on the server** - N/A. No server, no per-user
    resources.
15. **(8) Collect less** - PASS. The site collects nothing: no analytics, no
    telemetry, no cookies. The corpus and eval set are hand-written astronomy
    text with no personal data; demo content contains no names, emails,
    phones, or addresses (verified by grep).

## Injection and input handling

16. **(9) No untrusted text into innerHTML/eval/DOM HTML** - PASS.
    `grep -rn "innerHTML\|eval(\|dangerouslySetInnerHTML\|new Function" src/`
    matches only a comment stating the rule. All corpus, query, and model
    text renders through textContent/createElement. Browser smoke test
    includes a live XSS probe (`<img src=x onerror=...>` as a query and as an
    uploaded doc title) and asserts it cannot execute. CSP has no
    unsafe-inline and no unsafe-eval (only 'wasm-unsafe-eval' for ONNX).
17. **(10) Parameterized queries; no shell strings** - N/A. No SQL, no shell
    execution, no pickle/yaml. Corpus JSON is parsed with JSON.parse and
    validated field-by-field (src/core/corpus.ts).
18. **(11) Validate URLs and file paths** - PASS. The page fetches only three
    relative data paths plus Hugging Face model URLs allowed by CSP
    connect-src; no user input is ever turned into a fetch URL. Uploads are
    read with FileReader, never by path. Import limits (500 docs,
    2,000,000 chars, strict id/title rules) are enforced before any
    processing.

## Dependencies and supply chain

19. **(12) Every package exists and is the right one** - PASS. Dependencies
    are @huggingface/transformers, onnxruntime-web (transitive), esbuild,
    typescript - all long-established, verified on npmjs.com before install.
    package-lock.json pins exact versions and integrity hashes.
20. **(13) Commit lockfiles and audit** - PASS. package-lock.json is
    committed; npm audit is clean and runs in CI.
21. **(14) Treat AI config files as code** - PASS. The repo contains no
    AI-assistant instruction files of any kind. No code or README
    instructions were copied from external repos; everything here is written
    for this project.
22. **(15) Agents never get production access** - N/A. There is no production
    data or environment; the site is static and its data is public content.

## Config, transport and accounts

23. **(16) HTTPS only** - TODO at publish: tick "Enforce HTTPS" in the repo's
    Pages settings. The site itself makes no http:// requests (grep-verified;
    CSP upgrades nothing because nothing insecure is referenced).
24. **(17) Security headers** - PASS within Pages' limits. CSP via <meta>:
    default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; connect-src
    'self' plus the four pinned Hugging Face download hosts; img-src 'self'
    data:; object-src 'none'; base-uri 'none'; form-action 'none'.
    frame-ancestors cannot be set via meta (documented Pages limitation).
25. **(18) Debug off, errors quiet, source maps off** - PASS. esbuild runs
    with minify and sourcemap: false; no .map files ship in docs/
    (ls-verified). Errors surface to the user as short messages, not stack
    traces; nothing logs sensitive state (there is none).
26. **(19) Login provider, never homemade auth** - N/A. No auth anywhere.
27. **(20) Rate limits and abuse controls** - N/A. No public endpoint; GitHub
    Pages serves static files. Client-side cost is bounded by import limits
    and same-machine compute only.
28. **(21) Fail closed and log the failure** - PASS. Corpus validation
    rejects on the first violated rule (never imports partially). Unknown
    worker messages are answered with an explicit error, never executed.
    Model downloads that fail surface an error and leave the demo in its
    precomputed state. Nothing fails open.
29. **(22) Two-factor on accounts** - N/A to this repo's code; account-level
    setting on GitHub is the owner's own checklist item and stays with them.

## Static sites, GitHub Pages and Vercel

30. **(23) Everything in the published folder is public** - PASS. docs/
    contains only the built site, vendored wasm, and the public corpus/eval/
    precomputed data. Searched for phone, email, address, and personal names:
    none present. Git author identity is the owner's public GitHub noreply
    address only.
31. **(24) No client-side login or hidden admin page** - N/A. No login, no
    admin, no hidden routes.
32. **(25) Vercel secrets** - N/A. Hosted on GitHub Pages, not Vercel; and no
    secrets exist regardless.
33. **(26) Pin or self-host third-party scripts and fonts** - PASS. Zero CDN
    scripts, zero web fonts, zero external assets: app.js, worker.js and the
    ONNX wasm are self-hosted in docs/; model weights download only from the
    pinned Hugging Face hosts at exact content revisions. system-ui font
    stack only.
34. **(27) Forms go to a service with spam protection** - N/A. No forms.
35. **(28) Delete old deployments** - N/A to this new repo; owner's standing
    account hygiene item.

## Android apps (items 29-36, 37-39)

36. **(29-39) APK, manifest, keystore, WebView, relay channel, update
    hashes** - N/A. This project ships no Android app and no update channel.

## Python tools and the pension monitor (items 40-45)

37. **(40-43) Credentials, read-only monitors, log scrubbing, safe
    parsing** - N/A. No Python, no monitor, no credentials. (Item 44 applies
    to this repo's CI and is answered below.)
38. **(44) GitHub Actions least privilege** - PASS. The CI workflow sets
    `permissions: contents: read` at the top level, uses no secrets, and pins
    every third-party action to a full commit SHA.
39. **(45) Alerts to a private channel** - N/A. The project sends no alerts.

## Residual risks accepted on purpose

- **Model weight supply chain**: weights download from Hugging Face at pinned
  revisions; a compromised upstream revision would still be sandboxed by the
  CSP (no exfil host is whitelisted beyond the HF hosts themselves).
- **IndexedDB quota exhaustion via crafted upload**: bounded by the 2 MB
  import cap and browser quota; failure mode is a shown error, not a crash.
- **CI minutes**: the workflow runs on push/PR only; no scheduled runs.

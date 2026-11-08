#!/bin/sh
# check-docs.sh - the README's claims must match the code and data. When a
# pinned number changes on purpose, change it here and in README.md together.
set -eu
cd "$(dirname "$0")/.."
fail() { echo "check-docs: $1" >&2; exit 1; }

UNIT=$(npm test 2>&1 | grep -o '^# pass [0-9]*' | grep -o '[0-9]*')
[ "$UNIT" = "73" ] || fail "unit tests: README says 73, runner says $UNIT"
grep -q "73 unit tests" README.md || fail "README no longer says 73 unit tests"
grep -q "9 end-to-end checks" README.md || fail "README lost the smoke check count"
grep -q "aditya2375.github.io/recall" README.md || fail "README lost the live demo link"

# eval table pins (regenerate with node scripts/evaluate.mjs)
for row in "bm25|55.8|75.0|81.7|86.5|81.9" \
           "dense (flat)|53.7|88.3|95.0|89.5|87.3" \
           "dense (hnsw ef=50)|53.7|88.3|95.0|89.5|87.3" \
           "hybrid rrf|54.6|85.4|90.8|91.3|88.1" \
           "hybrid linear a=0.5|60.8|86.7|93.3|94.9|91.4" \
           "hybrid rrf + rerank|64.2|88.8|92.9|97.8|93.8"; do
  pat=$(echo "$row" | sed 's/\./\\./g; s/|/ *| */g')
  grep -qE "^\\| $pat" README.md || fail "README eval row missing: $row"
done
grep -q 'ef_search' README.md || fail "README lost the ef sweep header"
grep -qE '^\| recall *\| 80.7 *\| 94.3 *\| 99.1 *\| 100 *\|' README.md || fail "README HNSW recall row missing" 

# model pins agree between the browser adapters, the precompute script and README
for pin in "ea104dacec62c0de699686887e3f920caeb4f3e3" "a09144355adeed5f58c8ed011d209bf8ee5a1fec"; do
  grep -q "$pin" src/web/models.ts || fail "models.ts lost revision pin $pin"
  grep -q "$pin" scripts/precompute-core.mjs || fail "precompute-core.mjs lost revision pin $pin"
done
grep -q 'ea104dac' README.md || fail "README lost the embed model revision"
grep -q 'a0914435' README.md || fail "README lost the reranker revision"

# data pins
DOCS=$(node -e "console.log(JSON.parse(require('fs').readFileSync('data/corpus/night-sky.json','utf8')).docs.length)")
[ "$DOCS" = "50" ] || fail "corpus doc count changed: $DOCS"
grep -q "50-document" README.md || fail "README lost the corpus size"
grep -q "40-query" README.md || fail "README lost the eval set size"
grep -q "71 graded" README.md || fail "README lost the judgment count"

# built site is present and self-contained
for f in docs/index.html docs/app.js docs/worker.js docs/style.css \
         docs/ort/ort-wasm-simd-threaded.mjs docs/ort/ort-wasm-simd-threaded.wasm \
         docs/data/precomputed/night-sky.bge-small-en-v1.5.json; do
  [ -s "$f" ] || fail "$f missing - run node scripts/build-web.mjs"
done
grep -q "Content-Security-Policy" docs/index.html || fail "docs/index.html lost its CSP"

# import-limit pins
grep -q "maxDocs: 500" src/core/corpus.ts || fail "corpus doc limit changed"
grep -q "maxTotalChars: 2_000_000" src/core/corpus.ts || fail "corpus size limit changed"

# no stray AI/tooling mentions anywhere public-facing. Authored text only:
# minified bundles, base64 artifacts and wasm can contain look-alike bytes.
! grep -rli --exclude=check-docs.sh "claude\|anthropic\|openai\|chatgpt\|copilot\|gpt-4\|llm" \
  README.md SECURITY_REVIEW.md src/ scripts/ tests/ data/corpus data/eval \
  .github/ package.json docs/index.html docs/style.css \
  || fail "tooling mention found"
echo "check-docs ok"

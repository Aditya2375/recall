import { pipeline, AutoTokenizer, env } from '@huggingface/transformers';
env.cacheDir = new URL('../.cache/hf', import.meta.url).pathname;
await pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5', { dtype: 'q8', revision: 'ea104dacec62c0de699686887e3f920caeb4f3e3' });
console.log('bge cached');
await AutoTokenizer.from_pretrained('Xenova/ms-marco-MiniLM-L-6-v2', { revision: 'a09144355adeed5f58c8ed011d209bf8ee5a1fec' });
console.log('msmarco tokenizer cached');

/**
 * Chunkers. Every chunk keeps character offsets into its source document,
 * so the UI can highlight exactly what was indexed. Chunk ids are stable:
 * `${docId}#c${position}` with position assigned in document order.
 *
 * - fixed: sliding token window with overlap.
 * - structure: split on Markdown headings and blank-line paragraph breaks,
 *   merge small neighbours up to the token budget, split oversized blocks
 *   with the fixed chunker. Heading context travels with each chunk.
 */

import { tokenizeWithSpans } from './tokenize.js';

export interface Chunk {
  id: string;
  docId: string;
  index: number;
  text: string;
  start: number;
  end: number;
  headings: string[];
}

export interface ChunkOptions {
  maxTokens?: number;
  overlap?: number;
}

const DEFAULT_MAX_TOKENS = 180;
const DEFAULT_OVERLAP = 36;

function resolve(opts: ChunkOptions): { maxTokens: number; overlap: number } {
  const maxTokens = opts.maxTokens ?? DEFAULT_MAX_TOKENS;
  const overlap = opts.overlap ?? DEFAULT_OVERLAP;
  if (!Number.isInteger(maxTokens) || maxTokens <= 0) throw new Error('chunk: maxTokens must be a positive integer');
  if (!Number.isInteger(overlap) || overlap < 0) throw new Error('chunk: overlap must be a non-negative integer');
  if (overlap >= maxTokens) throw new Error('chunk: overlap must be smaller than maxTokens');
  return { maxTokens, overlap };
}

interface Block {
  start: number;
  end: number;
  headings: string[];
}

export function chunkFixed(docId: string, text: string, opts: ChunkOptions = {}): Chunk[] {
  const { maxTokens, overlap } = resolve(opts);
  const tokens = tokenizeWithSpans(text);
  if (tokens.length === 0) return [];
  const out: Chunk[] = [];
  const step = maxTokens - overlap;
  for (let begin = 0; begin < tokens.length; begin += step) {
    const endTok = Math.min(begin + maxTokens, tokens.length);
    const start = tokens[begin]!.start;
    const end = tokens[endTok - 1]!.end;
    out.push({
      id: `${docId}#c${out.length}`,
      docId,
      index: out.length,
      text: text.slice(start, end),
      start,
      end,
      headings: [],
    });
    if (endTok === tokens.length) break;
  }
  return out;
}

/** Split raw text into heading-aware blocks. Internal; tested via chunkStructure. */
function splitBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  const headingStack: Array<{ level: number; title: string }> = [];
  const lines = text.split('\n');
  let offset = 0;
  let blockStart: number | null = null;
  let blockEnd = 0;

  const flush = () => {
    if (blockStart !== null && blockEnd > blockStart) {
      blocks.push({
        start: blockStart,
        end: blockEnd,
        headings: headingStack.map((h) => h.title),
      });
    }
    blockStart = null;
  };

  for (const line of lines) {
    const lineStart = offset;
    const lineEnd = offset + line.length;
    offset = lineEnd + 1; // account for the '\n'
    const heading = /^(#{1,6})\s+(.+?)\s*$/.exec(line);
    if (heading) {
      flush();
      const level = heading[1]!.length;
      const title = heading[2]!.replace(/#+\s*$/, '').trim();
      while (headingStack.length > 0 && headingStack[headingStack.length - 1]!.level >= level) {
        headingStack.pop();
      }
      headingStack.push({ level, title });
      continue;
    }
    if (line.trim() === '') {
      flush();
      continue;
    }
    if (blockStart === null) blockStart = lineStart;
    blockEnd = lineEnd;
  }
  flush();
  return blocks;
}

export function chunkStructure(docId: string, text: string, opts: ChunkOptions = {}): Chunk[] {
  const { maxTokens, overlap } = resolve(opts);
  const blocks = splitBlocks(text);
  const out: Chunk[] = [];
  const push = (start: number, end: number, headings: string[]) => {
    out.push({
      id: `${docId}#c${out.length}`,
      docId,
      index: out.length,
      text: text.slice(start, end),
      start,
      end,
      headings,
    });
  };

  let pending: Block | null = null;
  const pendingTokens = (b: Block) => tokenizeWithSpans(text.slice(b.start, b.end)).length;

  for (const block of blocks) {
    const count = tokenizeWithSpans(text.slice(block.start, block.end)).length;
    if (count > maxTokens) {
      if (pending !== null) {
        push(pending.start, pending.end, pending.headings);
        pending = null;
      }
      // Oversized block: fall back to fixed windows inside it, keeping headings.
      const inner = chunkFixed(docId, text.slice(block.start, block.end), { maxTokens, overlap });
      for (const c of inner) {
        push(block.start + c.start, block.start + c.end, block.headings);
      }
      continue;
    }
    if (pending !== null && pendingTokens(pending) + count > maxTokens) {
      push(pending.start, pending.end, pending.headings);
      pending = null;
    }
    if (pending === null) {
      pending = { ...block };
    } else {
      // Merge: headings of the newer block win when they introduce new context.
      const merged: string[] = pending.headings.slice();
      for (const h of block.headings) {
        if (!merged.includes(h)) merged.push(h);
      }
      pending = { start: pending.start, end: block.end, headings: merged };
    }
  }
  if (pending !== null) push(pending.start, pending.end, pending.headings);
  return out;
}

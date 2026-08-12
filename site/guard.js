/* =============================================================================
 * JudgeGuard — the distilled guardrail, running in the browser.
 *
 * This is a faithful port of judgeguard/distill/{features,train}.py, not a
 * lookalike. The student is a logistic model over hashed n-grams plus an
 * isotonic calibration layer, so the entire inference path is a sparse dot
 * product and a step function — nothing about it needs a server.
 *
 * Two things bite here. The hashing must match exactly, and the calibration
 * layer operates on the decision function rather than a squashed probability.
 * The fiddly part is the hashing. scikit-learn's HashingVectorizer indexes with
 * MurmurHash3 (x86_32, signed, seed 0) over UTF-8 bytes, and any disagreement
 * in tokenisation, n-gram construction or the hash itself silently produces a
 * different model rather than an error. site/parity.test.js therefore checks
 * this port against probabilities computed in Python.
 *
 * Note on Unicode: Python's `(?u)\w` and JS's `\w` differ outside ASCII. The
 * corpus is ASCII, so the ports agree; a non-ASCII corpus would need
 * \p{L}-class patterns on this side.
 * ========================================================================== */

/* ------------------------------------------------------------------ murmur3 */
function murmur3_32(bytes, seed = 0) {
  const c1 = 0xcc9e2d51, c2 = 0x1b873593;
  let h1 = seed | 0;
  const nblocks = bytes.length >> 2;

  for (let i = 0; i < nblocks; i++) {
    const j = i << 2;
    let k1 = (bytes[j] | (bytes[j + 1] << 8) | (bytes[j + 2] << 16) | (bytes[j + 3] << 24)) | 0;
    k1 = Math.imul(k1, c1);
    k1 = (k1 << 15) | (k1 >>> 17);
    k1 = Math.imul(k1, c2);
    h1 ^= k1;
    h1 = (h1 << 13) | (h1 >>> 19);
    h1 = (Math.imul(h1, 5) + 0xe6546b64) | 0;
  }

  let k1 = 0;
  const tail = nblocks << 2;
  switch (bytes.length & 3) {
    case 3: k1 ^= bytes[tail + 2] << 16;  // falls through
    case 2: k1 ^= bytes[tail + 1] << 8;   // falls through
    case 1:
      k1 ^= bytes[tail];
      k1 = Math.imul(k1, c1);
      k1 = (k1 << 15) | (k1 >>> 17);
      k1 = Math.imul(k1, c2);
      h1 ^= k1;
  }

  h1 ^= bytes.length;
  h1 ^= h1 >>> 16;
  h1 = Math.imul(h1, 0x85ebca6b);
  h1 ^= h1 >>> 13;
  h1 = Math.imul(h1, 0xc2b2ae35);
  h1 ^= h1 >>> 16;
  return h1 | 0;                       // signed, exactly as murmurhash3_bytes_s32
}

const _enc = typeof TextEncoder !== "undefined" ? new TextEncoder() : null;
function hashIndex(token, nFeatures) {
  const h = murmur3_32(_enc.encode(token), 0);
  return Math.abs(h) % nFeatures;      // sklearn: abs(h) % n_features
}

/* -------------------------------------------------------------- tokenisation */
const TOKEN_RE = /\b\w\w+\b/g;         // sklearn default token_pattern
const WORD_RE = /[a-zA-Z']+/g;         // judgeguard.distill.features._WORD
const DIGIT_RE = /\d/g;
const WHITESPACE_RUN = /\s\s+/g;       // sklearn CountVectorizer._white_spaces
const PVALUE_RE = /\bp\s*=\s*0/;

function wordNgrams(text, [minN, maxN]) {
  const toks = (text.toLowerCase().match(TOKEN_RE) || []);
  let out = [];
  if (maxN === 1) return toks;
  let lo = minN;
  if (minN === 1) { out = toks.slice(); lo = 2; }
  const n0 = toks.length;
  for (let n = lo; n < Math.min(maxN + 1, n0 + 1); n++) {
    for (let i = 0; i < n0 - n + 1; i++) out.push(toks.slice(i, i + n).join(" "));
  }
  return out;
}

function charWbNgrams(text, [minN, maxN]) {
  // Mirrors sklearn's _char_wb_ngrams: pad each word, slide, and emit a short
  // word exactly once rather than once per n.
  const doc = text.toLowerCase().replace(WHITESPACE_RUN, " ");
  const out = [];
  for (const raw of doc.split(/\s+/)) {
    if (!raw) continue;
    const w = " " + raw + " ";
    const wLen = w.length;
    for (let n = minN; n <= maxN; n++) {
      let offset = 0;
      out.push(w.slice(offset, offset + n));
      while (offset + n < wLen) { offset += 1; out.push(w.slice(offset, offset + n)); }
      if (offset === 0) break;         // word shorter than n: counted once
    }
  }
  return out;
}

/** Hash a token list into an L2-normalised sparse vector (alternate_sign=False). */
function hashVector(tokens, nFeatures, offset) {
  const counts = new Map();
  for (const t of tokens) {
    const idx = hashIndex(t, nFeatures);
    counts.set(idx, (counts.get(idx) || 0) + 1);
  }
  let norm = 0;
  for (const v of counts.values()) norm += v * v;
  norm = Math.sqrt(norm) || 1;
  const out = new Map();
  for (const [i, v] of counts) out.set(i + offset, v / norm);
  return out;
}

/* ------------------------------------------------------------------ features */
function structuralFeatures(text, vocab) {
  const lower = text.toLowerCase();
  const words = lower.match(WORD_RE) || [];
  const n = Math.max(words.length, 1);
  const chars = Math.max(text.length, 1);
  const uniq = new Set(words).size;
  const sentences = Math.max(
    (text.split(".").length - 1) + (text.split("!").length - 1) + (text.split("?").length - 1), 1
  );
  const hedgeSet = new Set(vocab.hedge_words);
  const hedges = words.reduce((a, w) => a + (hedgeSet.has(w) ? 1 : 0), 0);
  let filler = 0;
  for (const p of vocab.filler_phrases) filler += lower.split(p).length - 1;
  const digits = (text.match(DIGIT_RE) || []).length;
  const commas = text.split(",").length - 1;
  const firstUpper = text.length > 0 && text[0] !== text[0].toLowerCase() ? 1 : 0;
  const long = words.reduce((a, w) => a + (w.length > 9 ? 1 : 0), 0);
  return [
    Math.log1p(text.length),
    Math.log1p(words.length),
    uniq / n,
    hedges / n,
    filler,
    digits / chars,
    commas / sentences,
    n / sentences,
    firstUpper,
    Math.log1p(sentences),
    long / n,
    PVALUE_RE.test(text) ? 1 : 0,
  ];
}

function coverageFeatures(context, question, answer, vocab) {
  const stop = new Set(vocab.stop_words);
  const setOf = (s) => {
    const out = new Set();
    for (const w of (s.toLowerCase().match(WORD_RE) || [])) if (!stop.has(w)) out.add(w);
    return out;
  };
  const ctx = setOf(context), ans = setOf(answer), q = setOf(question);
  if (ctx.size === 0) return [0, 0, 0, 0, 0, 0];
  let inter = 0; for (const w of ctx) if (ans.has(w)) inter++;
  let qa = 0; for (const w of q) if (ans.has(w)) qa++;
  let unsupported = 0; for (const w of ans) if (!ctx.has(w)) unsupported++;
  const nAns = Math.max(ans.size, 1);
  const digits = (answer.match(DIGIT_RE) || []).length;
  return [
    inter / Math.max(ctx.size, 1),
    inter / nAns,
    unsupported / nAns,
    qa / Math.max(q.size, 1),
    ans.size / Math.max(ctx.size, 1),
    digits / nAns,
  ];
}

/* ------------------------------------------------------------------ predict */
function isotonic(cal, p) {
  // sklearn IsotonicRegression with out_of_bounds="clip": piecewise linear
  // interpolation between knots, clamped at the ends.
  const { x, y } = cal;
  if (!x.length) return p;
  if (p <= x[0]) return y[0];
  if (p >= x[x.length - 1]) return y[y.length - 1];
  let lo = 0, hi = x.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (x[mid] <= p) lo = mid; else hi = mid; }
  const dx = x[hi] - x[lo];
  if (dx === 0) return y[hi];
  return y[lo] + ((p - x[lo]) * (y[hi] - y[lo])) / dx;
}

export class BrowserGuard {
  constructor(model) {
    this.m = model;
    this.threshold = model.threshold;
    // Sparse coefficient lookup per ensemble member.
    this.maps = model.members.map((mem) => {
      const map = new Map();
      const { i, v } = mem.coef;
      for (let k = 0; k < i.length; k++) map.set(i[k], v[k]);
      return map;
    });
  }

  featurize(context, question, answer) {
    const { word: nWord, char: nChar } = this.m.dims;
    const vec = new Map();
    for (const [i, v] of hashVector(wordNgrams(answer, this.m.ngrams.word), nWord, 0)) vec.set(i, v);
    for (const [i, v] of hashVector(charWbNgrams(answer, this.m.ngrams.char), nChar, nWord)) vec.set(i, v);
    const extra = [
      ...structuralFeatures(answer, this.m.vocab),
      ...coverageFeatures(context, question, answer, this.m.vocab),
    ];
    const base = nWord + nChar;
    extra.forEach((v, k) => { if (v !== 0) vec.set(base + k, v); });
    return vec;
  }

  probability(context, question, answer) {
    const x = this.featurize(context, question, answer);
    let acc = 0;
    for (let m = 0; m < this.m.members.length; m++) {
      const map = this.maps[m];
      let z = this.m.members[m].b;
      for (const [i, v] of x) { const w = map.get(i); if (w !== undefined) z += w * v; }
      // NOT sigmoid(z). sklearn's CalibratedClassifierCV fits the isotonic
      // calibrator on the estimator's *decision function*, so its knots live in
      // z-space (here roughly -11 to +2.5), not in [0, 1]. Squashing first
      // silently produces a different model -- it cost 12 of 30 parity cases.
      acc += isotonic(this.m.members[m].iso, z);
    }
    let p = acc / this.m.members.length;
    if (this.m.recalibrator) p = isotonic(this.m.recalibrator, p);
    return Math.min(1, Math.max(0, p));
  }

  /** Same shape as the FastAPI /guard response, so the two demos agree. */
  guard(context, question, answer) {
    const t0 = performance.now();
    const p = this.probability(context, question, answer);
    const dt = performance.now() - t0;
    const s = structuralFeatures(answer, this.m.vocab);
    const cov = coverageFeatures(context, question, answer, this.m.vocab);
    const signals = {
      length_chars: answer.length,
      type_token_ratio: s[2],
      hedge_density: s[3],
      filler_phrase_count: s[4],
      numeric_density: s[5],
      source_coverage: cov[0],
      unsupported_vocabulary: cov[2],
    };
    const decision = p >= this.threshold ? "allow" : "block";
    const drivers = [];
    if (signals.hedge_density > 0.02) drivers.push("high hedge-word density");
    if (signals.filler_phrase_count >= 2) drivers.push("content-free filler phrases");
    if (signals.numeric_density < 0.005) drivers.push("few committed numeric claims");
    if (signals.source_coverage < 0.35) drivers.push("low coverage of the source record");
    return {
      decision,
      probability_good: p,
      threshold: this.threshold,
      latency_ms: dt,
      reason:
        decision === "block"
          ? `p(acceptable)=${p.toFixed(3)} below threshold ${this.threshold.toFixed(2)}` +
            (drivers.length ? `; contributing signals: ${drivers.join(", ")}` : "")
          : `p(acceptable)=${p.toFixed(3)} at or above threshold ${this.threshold.toFixed(2)}`,
      signals,
      path: "inline-guardrail (browser)",
    };
  }
}

export const _internals = { murmur3_32, hashIndex, wordNgrams, charWbNgrams, structuralFeatures, coverageFeatures };

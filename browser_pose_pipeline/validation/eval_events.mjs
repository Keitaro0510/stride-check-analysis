// Contact / toe-off detection error against Fukuchi 2017 force-plate events.
// Input: outputs/fukuchi_events/*.json from prepare_fukuchi_events.py.
// Usage: node eval_events.mjs [fps,...] [noise,...]
//   noise = SD of Gaussian jitter added to every 2-D point, as a fraction of
//   leg length (a stand-in for pose-estimator error).
import { readdirSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
globalThis.self = globalThis;
createRequire(import.meta.url)(resolve(here, "../extract/extract.js"));
const X = self.StrideExtract._internal;

const FPS = (process.argv[2] || "33,60,120").split(",").map(Number);
const NOISE = (process.argv[3] || "0,0.01,0.02").split(",").map(Number);
const DIR = resolve(here, "outputs/fukuchi_events");
export const trials = readdirSync(DIR).filter(f => f.endsWith(".json")).sort().map(f => JSON.parse(readFileSync(resolve(DIR, f), "utf8")));

// Deterministic Gaussian noise so runs are comparable.
function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function gauss(u) { return Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u()); }

// Resample the 150 Hz marker series to `fps` video frames (frame i shows time i/fps).
export function toFrames(trial, fps, noise, seed) {
  const n = trial.series.LHip.length, dur = (n - 1) / trial.fs, count = Math.floor(dur * fps);
  const hip = trial.series.LHip, ank = trial.series.LAnkle;
  const legLen = Math.hypot(hip[0][0] - ank[0][0], hip[0][1] - ank[0][1]);
  const u = rng(seed), frames = [];
  for (let i = 0; i < count; i++) {
    const t = i / fps * trial.fs, a = Math.floor(t), w = t - a, b = Math.min(n - 1, a + 1);
    const frame = [];
    for (const [name, idx] of Object.entries(trial.landmarks)) {
      const p = trial.series[name];
      frame[idx] = { x: p[a][0] * (1 - w) + p[b][0] * w + gauss(u) * noise * legLen,
        y: p[a][1] * (1 - w) + p[b][1] * w + gauss(u) * noise * legLen, visibility: 1 };
    }
    frames.push(frame);
  }
  return frames;
}

const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const sd = a => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / Math.max(1, a.length - 1)); };
const median = a => { const b = [...a].sort((x, y) => x - y), m = b.length >> 1; return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2; };
function corr(a, b) { const ma = mean(a), mb = mean(b); let n = 0, da = 0, db = 0; a.forEach((v, i) => { n += (v - ma) * (b[i] - mb); da += (v - ma) ** 2; db += (b[i] - mb) ** 2; }); return n / Math.sqrt(da * db); }

export function evaluate(detect, fps, noise) {
  const ic = [], to = [], contactErr = [], trialRows = [];
  let truthSteps = 0, matched = 0, detected = 0;
  trials.forEach((trial, k) => {
    const frames = toFrames(trial, fps, noise, 1000 + k);
    const steps = detect(frames, fps);
    const tStart = 0.3, tEnd = frames.length / fps - 0.3;  // skip edge steps
    const truth = trial.events.filter(e => e.on_s > tStart && e.off_s < tEnd);
    truthSteps += truth.length; detected += steps.length;
    const est = [], ref = [];
    for (const s of steps) {
      const c = s.contact / fps, o = s.off / fps;
      const cand = truth.filter(e => e.leg === s.leg).sort((a, b) => Math.abs(a.on_s - c) - Math.abs(b.on_s - c))[0];
      if (!cand || Math.abs(cand.on_s - c) > 0.15) continue;
      matched++;
      ic.push((c - cand.on_s) * 1000); to.push((o - cand.off_s) * 1000);
      contactErr.push(((o - c) - (cand.off_s - cand.on_s)) * 1000);
      est.push(o - c); ref.push(cand.off_s - cand.on_s);
    }
    // What the app reports: the per-trial median.
    if (est.length >= 5) trialRows.push({ est: median(est), ref: median(ref) });
  });
  const tr = trialRows.map(r => (r.est - r.ref) * 1000);
  return { fps, noise, trials: trialRows.length, detectRate: matched / truthSteps, falseRate: (detected - matched) / Math.max(1, detected),
    ic: [mean(ic), sd(ic)], to: [mean(to), sd(to)], contact: [mean(contactErr), sd(contactErr)],
    trialContact: [mean(tr), sd(tr), trialRows.length > 2 ? corr(trialRows.map(r => r.est), trialRows.map(r => r.ref)) : NaN] };
}

export const detectors = {
  current: (frames, fps) => X.detectSteps(frames, fps, X.progressionSign(frames)),
  // Same, after the side-view leg tracking (markers are labelled correctly,
  // so this checks that the tracking does not break good labels).
  tracked: (frames, fps) => { const c = X.cleanLegs(frames, fps).frames; return X.detectSteps(c, fps, X.progressionSign(c)); },
};

function fmt(r) {
  const f = (v, d = 0) => Number.isFinite(v) ? v.toFixed(d) : "—";
  return `${String(r.fps).padStart(3)} fps  noise ${(r.noise * 100).toFixed(0)}%  found ${f(r.detectRate * 100)}% (false ${f(r.falseRate * 100)}%)  ` +
    `IC ${f(r.ic[0])}±${f(r.ic[1])} ms  TO ${f(r.to[0])}±${f(r.to[1])} ms  contact ${f(r.contact[0])}±${f(r.contact[1])} ms  ` +
    `trial-median contact ${f(r.trialContact[0])}±${f(r.trialContact[1])} ms r=${f(r.trialContact[2], 2)} (n=${r.trials})`;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const names = (process.env.DETECTORS || Object.keys(detectors).join(",")).split(",");
  for (const name of names) {
    console.log(`== ${name}`);
    for (const fps of FPS) for (const noise of NOISE) console.log(fmt(evaluate(detectors[name], fps, noise)));
  }
}

// Per-step measures from the pose points of the Twente videos, computed by
// extract.js exactly as the browser does after pose estimation (side-view leg
// tracking, naming from the front video, event detection, sideValues).
// Input: outputs/mp/SubjXX_run_YY_{side,front}.json from run_pose_py.py
// Output: outputs/twente/ours/SubjXX_run_YY.json (times in video seconds)
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
globalThis.self = globalThis;
createRequire(import.meta.url)(resolve(here, "../extract/extract.js"));
const X = self.StrideExtract._internal;
// MPDIR: pose points to use (default outputs/mp = Full model); TAG is added to
// the output names, e.g. MPDIR=outputs/mp_heavy TAG=heavy -> SubjXX_run_YY.heavy.json
const MP = resolve(here, process.env.MPDIR || "outputs/mp"), OUT = resolve(here, "outputs/twente/ours"), TAG = process.env.TAG ? "." + process.env.TAG : "";
mkdirSync(OUT, { recursive: true });

const obj = fr => fr.map(f => f ? f.map(p => p ? { x: p[0], y: p[1], visibility: p[2], z: p[3] } : null) : null);
// "tracked" (default, as the page), "tracked-only" (always track), "raw" (never)
const variant = process.env.VARIANT || "tracked";
// KF candidates over the stance phase: frames whose hip, knee and ankle were
// seen (visibility >= 0.5; points filled in by the leg tracking have 0.3).
function kfVariants(frames, s, dir) {
  const K = s.leg === "L" ? [23, 25, 27] : [24, 26, 28], a = Math.round(s.contact), b = Math.round(s.off), mid = (s.contact + s.off) / 2;
  const seen = i => frames[i] && K.every(k => frames[i][k] && frames[i][k].visibility >= .5);
  const kf = i => X.sideValues(frames[i], s.leg, dir).KF;
  const all = [], meas = [];
  for (let i = a; i <= b && i < frames.length; i++) { const v = kf(i); if (!Number.isFinite(v)) continue; all.push(v); if (seen(i)) meas.push(v); }
  let near = null;
  for (let d = 0; d <= b - a; d++) for (const i of [Math.round(mid) - d, Math.round(mid) + d]) if (near == null && i >= a && i <= b && seen(i)) near = kf(i);
  const mx = v => v.length ? Math.max(...v) : null;
  return { KF_max_seen: mx(meas), KF_max_all: mx(all), KF_mid_seen: Number.isFinite(near) ? near : null };
}
for (const f of readdirSync(MP).filter(n => n.endsWith("_side.json")).sort()) {
  const id = f.replace("_side.json", "");
  const side = JSON.parse(readFileSync(resolve(MP, f), "utf8")), front = JSON.parse(readFileSync(resolve(MP, id + "_front.json"), "utf8"));
  const fps = side.fps;
  let frames = obj(side.frames), naming = null;
  let method = "raw";
  if (variant === "tracked") {   // what the page does: tracked or labels, whichever matches the front video
    const r = X.resolveSideLegs(frames, fps, { frames: obj(front.frames), fps: front.fps });
    frames = r.frames; naming = r.naming; method = r.method;
  } else if (variant === "tracked-only") {
    frames = X.cleanLegs(frames, fps).frames; method = "tracked";
    naming = X.nameLegsByFrontal(frames, fps, { frames: obj(front.frames), fps: front.fps });
    if (naming && naming.swap) X.swapLegs(frames);
  }
  const dir = X.progressionSign(frames), steps = [];
  const at = i => Math.min(frames.length - 1, Math.max(0, Math.round(i)));
  for (const s of X.detectSteps(frames, fps, dir)) {
    const ic = X.sideValues(frames[at(s.contact)], s.leg, dir), mid = X.sideValues(frames[at((s.contact + s.off) / 2)], s.leg, dir);
    const num = v => Number.isFinite(v) ? v : null;
    // stride_s: this leg's cycle; back_s: the toe's most rearward moment (off + K * stride)
    steps.push({ leg: s.leg, on: s.contact / fps, off: s.off / fps, opposite: s.opposite / fps,
      stride_s: (s.next - s.contact) / fps, back_s: (s.off + X.TO_BEFORE_TOE_BACK_STRIDE * (s.next - s.contact)) / fps,
      KF: num(mid.KF), knee_flex_contact: num(ic.knee_flex_contact), shank_angle_contact: num(ic.shank_angle_contact),
      foot_angle_contact: num(ic.foot_angle_contact), overstride: num(ic.overstride), ...kfVariants(frames, s, dir) });
  }
  steps.sort((a, b) => a.on - b.on);
  writeFileSync(resolve(OUT, `${id}${variant === "tracked" ? "" : "." + variant}${TAG}.json`),
    JSON.stringify({ id, fps, frames: frames.length, direction: dir, method, naming, steps }));
  console.log(id, variant, method, "steps", steps.length, "naming", JSON.stringify(naming));
}

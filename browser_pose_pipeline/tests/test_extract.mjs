// Node checks for the parts of extract.js that do not need MediaPipe.
// Run: node browser_pose_pipeline/tests/test_extract.mjs
import assert from "node:assert/strict";
import { existsSync, openAsBlob } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
globalThis.self = globalThis;
createRequire(import.meta.url)(resolve(here, "../extract/extract.js"));
const X = self.StrideExtract;

// MediaPipe landmark indices used by rearValues.
const L_HIP = 23, R_HIP = 24, L_KNEE = 25, L_ANKLE = 27;
function pose(points) {
  const frame = [];
  for (const [i, [x, y]] of Object.entries(points)) frame[i] = { x, y, visibility: 1 };
  return frame;
}
// Seen from behind: the runner's left is on the image left.  The left knee
// sits outside (image left of) the hip-ankle line, and the right pelvis drops.
const rear = { [L_HIP]: [.40, .50], [R_HIP]: [.60, .55], [L_KNEE]: [.37, .70], [L_ANKLE]: [.40, .90] };
// The same body seen from the front is the mirror image.
const front = Object.fromEntries(Object.entries(rear).map(([i, [x, y]]) => [i, [1 - x, y]]));

const r = X._internal.rearValues(pose(rear), "L", "rear");
const f = X._internal.rearValues(pose(front), "L", "front");
assert.ok(r.KA > 0, "rear: knee outside the hip-ankle line gives positive KA");
assert.ok(r.CPD > 0, "rear: contralateral pelvis drop gives positive CPD");
for (const k of ["CPD", "HADD", "KA"]) assert.ok(Math.abs(r[k] - f[k]) < 1e-9, `front view reproduces rear ${k}: ${r[k]} vs ${f[k]}`);
// Without the front flag the mirrored image would report the opposite KA.
assert.ok(X._internal.rearValues(pose(front), "L", "rear").KA < 0);

// Frame-rate probe on a real clip, when the (git-ignored) test data exists.
const clip = resolve(here, "../../rtmpose_pipeline/test_data/subj04_run81/videos/cam02_side_h264.mp4");
if (existsSync(clip)) {
  const info = await X.probeVideo(await openAsBlob(clip));
  assert.equal(info.fps, 33); assert.equal(info.frames, 165); assert.equal(info.codec, "avc1"); assert.equal(info.variable, false);
} else console.log("skip probeVideo: test clip not found");
assert.equal(await X.probeVideo(new Blob(["not a video"])), null);
console.log("ok");

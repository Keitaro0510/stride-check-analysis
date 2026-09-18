# Validation on the Twente videos (same videos as `Running_mp4`)

Ground truth: Zenodo 6457662 (University of Twente; the videos are Zenodo
6644593), treadmill running at 6.3 / 8.1 / 9.9 km/h, Subj04-12.  Qualisys
markers (128 Hz) and a split-belt force treadmill (2048 Hz) recorded with the
videos (33 fps; front camera and a camera on the runner's **left** side, so
L = near leg, R = far leg).  Data: `data/twente_locomotion_6sensors_2022/`
(`Processed_data.rar`, MD5 checked; only running trials extracted).

```bash
.venv/bin/python run_pose_py.py VIDEO OUT.json   # or ./run_all_pose.sh for all videos
python3 twente_truth.py                           # force-plate events, marker angles
node twente_ours.mjs                              # extract.js on the pose points
python3 twente_compare.py                         # VARIANT=raw node twente_ours.mjs; python3 twente_compare.py .raw
```

Pose estimation for validation runs in Python (`run_pose_py.py`: mediapipe
1.0.1, the same model file and thresholds as the page) because the headless
test browser is too slow; everything after it is extract.js in Node.  On one
clip the browser saw each image 1-2 frames later than OpenCV (an MP4 display
offset; constant, so intervals are unaffected) and points differed by ~20 px
even when aligned, so browser numbers may differ somewhat.

Sync: the Qualisys recording has exactly the length of the videos.  Matching
our contacts to force-plate contacts (search within ±0.17 s, half a step)
gives an offset of 6 ms median (−13 to +21 ms), so the videos start with the
Qualisys recording.  A wider search locks onto contacts one step away and
swaps left and right; that is what first made the labels look wrong.

### Result (2026-09-18, extractor 0.4.0, 24 of 26 trials)

Subj06 run_81 has no ground truth (missing marker); Subj11 run_99 was skipped
(side-view tracking found only 10 steps).

| | Raw MediaPipe labels | With side-view leg tracking |
|---|---|---|
| Our steps matched to a force-plate contact | 78% | 99% |
| Left/right label agrees with the force plate | 80% mean, 14/24 trials ≥ 95% | 93% mean, 19/24 trials ≥ 95% |
| Contact-time error per step, near / far leg | −41 ± 82 / −44 ± 277 ms | −41 ± 40 / −48 ± 44 ms |

The tracking fixes runners whose legs MediaPipe labels by position (Subj09,
Subj10 run_99, Subj11 run_63, Subj12 run_99), but made four trials worse
whose raw labels were already right (Subj04 run_99 99 → 73%, Subj07 run_81
83 → 49%, Subj07 run_99 89 → 56%, Subj12 run_63 98 → 85%; in Subj07 the
front video could not name the legs).

With tracking, per step (same leg), ms, mean ± SD:

| | Initial contact | Toe-off | Contact time |
|---|---|---|---|
| Near leg (L) | 0 ± 16 | −41 ± 37 | −41 ± 40 |
| Far leg (R) | 0 ± 13 | −48 ± 43 | −48 ± 44 |

Per trial and leg (median over steps, as the app reports), ours − truth
across 24 trials:

| Measure | Near (L): bias, SD, r | Far (R): bias, SD, r |
|---|---|---|
| Contact time | −42 ms, 30 ms, 0.79 | −47 ms, 27 ms, 0.81 |
| `KF` (knee flexion, mid-stance) | −18.9°, 6.4°, 0.64 | −20.6°, 5.5°, 0.42 |
| `knee_flex_contact` | −3.6°, 10.0°, 0.07 | −5.9°, 5.8°, 0.21 |
| `shank_angle_contact` | +0.4°, 8.0°, 0.25 | +2.3°, 3.2°, 0.80 |
| `foot_angle_contact` | −9.0°, 8.3°, 0.22 | +5.1°, 10.8°, 0.39 |
| `overstride` | −0.02, 0.03, 0.86 | −0.03, 0.02, 0.93 |

Truth angles use the same 2-D formulas on sagittal-plane markers (hip joint
centre, epicondyles, malleoli, calcaneus, metatarsal heads), so the landmark
definitions differ from MediaPipe's (its hip is not the joint centre, its
toe is the toe tip).

What this says:

- Initial contact is unbiased on both legs.  Toe-off is 40-50 ms early on
  MediaPipe points: the Fukuchi-fitted offset (MT heads) does not carry over
  to MediaPipe's toe tip.  Refit on these trials (by subject, cross-validated).
- `KF` is 19-21° too small: mid-stance is when the legs cross, where the leg
  tracking replaces points with an interpolated path.  Needs a different
  definition or frame choice (e.g. peak flexion over stance on measured
  frames only).
- `knee_flex_contact` and `foot_angle_contact` barely track the truth
  (r ≤ 0.4); `overstride` and, on the far leg, `shank_angle_contact` do.
- The far leg is not worse than the near leg after tracking.
- Spec targets (section 4) are not met for contact time (0.010 s) or the
  angles yet.

### Changes after this validation (extractor 0.5.0, same 24 trials)

1. **Leg source chosen per trial** (`resolveSideLegs()`): the tracked legs or
   MediaPipe's own labels, whichever matches the front video's stance pattern
   better (|r|).  Labels right: 93% → 97% of steps (20/24 trials ≥ 95%;
   Subj07 run_81 76%, run_99 87%, Subj06 run_63 88%, Subj12 run_63 92%).
2. **Toe-off refitted on MediaPipe points**: toe furthest back − 11.3% of the
   stride (was 16.6%, fitted on Fukuchi's metatarsal-head markers;
   `twente_fit_events.py`).  Leaving one subject out at a time: contact time
   per trial and leg −6.5 ± 29.4 ms, MAE 21 ms, r 0.77 (was −42/−47 ms bias).
   On Fukuchi markers the same constant now overestimates contact time by
   37 ms: the offset belongs to the pose model's toe point, so it is fitted
   on MediaPipe (spec 7-3).
3. **KF could not be fixed by choosing other frames.**  Against the truth at
   mid-stance (near / far leg, bias and r across trials):

   | KF computed as | Near (L) | Far (R) |
   |---|---|---|
   | mid-stance frame (spec, current) | −16.6°, r 0.25 | −18.4°, r 0.15 |
   | nearest frame with seen points | −16.5°, r 0.27 | −19.6°, r 0.16 |
   | peak flexion in stance, seen frames | −8.7°, r 0.29 | −10.8°, r 0.02 |
   | peak flexion in stance, all frames | −7.6°, r 0.33 | −8.6°, r 0.03 |
   | 3-D world landmarks, front video (`twente_kf3d.py`) | −6.0°, r −0.09 | −17.0°, r 0.23 |

   The interpolated crossing frames are not the cause (seen frames give the
   same bias).  The truth varies little between runners (SD 6.8° for KF,
   7.8° for knee flexion at contact), and our error SD is 6-12°, so the
   2-D side view cannot rank runners by knee flexion.  The same holds for
   `knee_flex_contact` and `foot_angle_contact` (r ≤ 0.4).  `overstride`
   (r 0.74 / 0.93) and far-leg `shank_angle_contact` (r 0.79) do track.

Decision (2026-09-18): `KF` is reported as `null` with a note in
`quality.notes`.  The load estimate then uses the study median (model.js
imputes missing inputs); the form type and the form-based prediction need
`KF` and are not shown.

### Heavy model and further fixes (extractor 0.6.0)

Pose estimation re-run with MediaPipe Pose Landmarker **Heavy**
(`run_pose_py.py --model heavy`, outputs/mp_heavy; `MPDIR=outputs/mp_heavy
TAG=heavy node twente_ours.mjs; python3 twente_compare.py .heavy`).  Heavy
is now the page's default model (30.7 MB, about 3x slower than Full).
Changes found while checking it:

- Running direction from heel → toe instead of nose vs hips (the head is
  cropped in these videos; the nose gave the wrong direction in one trial).
- Contacts from the heel's forward peaks **relative to the pelvis**, with a
  5-95 percentile threshold: runners drift back on the treadmill, and with
  an absolute threshold later steps were missed (Subj10 run_63: 92 → 147
  steps).  Fukuchi results unchanged.
- **Step-wise correction from the front video** (`fixLegsByFrontal()`): a
  missed or invented crossing swaps the legs for every later step; each
  crossing-to-crossing step is checked against the front video ("which foot
  is lower") over 5 steps and swapped when the views clearly disagree (mean
  agreement < −0.3; 0.2-0.3 were best for Heavy on these same trials, so the
  labels figure below is slightly optimistic).
- Toe-off refitted on Heavy: 12.0% of the stride.

Final, Heavy, 24 trials (Subj11 run_99 still fails: 5 steps):

| | Near (L) | Far (R) |
|---|---|---|
| Left/right labels right | 98.2% of steps, 22/24 trials ≥ 95% (Subj06 run_99 92%, Subj10 run_63 91%) | |
| Initial contact per step | −1 ± 16 ms | +2 ± 12 ms |
| Contact time per trial | −6 ms, SD 29 ms, MAE 21 ms, r 0.83 | −4 ms, SD 20 ms, MAE 14 ms, r 0.91 |
| Contact time, leave-one-subject-out toe-off fit | −8 ms, SD 25 ms, MAE 19 ms, r 0.85 (both legs) | |
| `foot_angle_contact` | +5.3°, SD 4.1°, r 0.85 | +16.1°, SD 9.1°, r 0.61 |
| `knee_flex_contact` | −7.0°, SD 8.8°, r 0.39 | −3.9°, SD 4.6°, r 0.52 |
| `shank_angle_contact` | +0.7°, SD 7.7°, r 0.34 | +2.6°, SD 3.0°, r 0.79 |
| `overstride` | −0.04, SD 0.02, r 0.93 | −0.04, SD 0.02, r 0.94 |
| `KF` (not reported) | −16.6°, r 0.48 | −14.7°, r −0.01 |

Compared with Full, Heavy improves contact time (far leg SD 26 → 20 ms)
and the near-leg foot angle at contact (r 0.15 → 0.85).  The far-leg foot
angle reads 16° too high (a heel-strike bias); a per-leg offset could be
fitted, but it has not been.  On Fukuchi markers the toe-off offset now gives
+31 ms (it belongs to MediaPipe's toe point).

# Event-detection validation (Fukuchi 2017)

Checks the contact / toe-off detection in `extract/extract.js` against
force-plate events, without any pose-estimation error.

- Reference: Fukuchi 2017 treadmill running (force 300 Hz, markers 150 Hz),
  2.5 / 3.5 / 4.5 m/s, first 15 s of each trial.  Contact = vertical force
  > 50 N; left/right from the centre of pressure.
- Input to the detector: 2-D side-view points built from the markers and laid
  out as MediaPipe landmarks (hip = hip joint centre, knee/ankle = mid-point of
  the medial and lateral markers, heel = `Heel.Bottom`, toe = mid-point of
  MT1 and MT5).  Resampled to 33 / 60 / 120 fps, with optional Gaussian jitter
  on every point (1% or 2% of leg length) as a stand-in for pose error.

```bash
python3 prepare_fukuchi_events.py      # ~6 min, writes outputs/ (git-ignored)
node eval_events.mjs [fps,...] [noise,...]
```

## Result (2026-09-18)

95-97 trials from 39 runners.  "Trial" = the per-trial median, which is what
the app reports.

| Detector | Initial contact | Toe-off | Contact time per trial | Duty per trial |
|---|---|---|---|---|
| v0.2 (heel forward peak; toe lifted 32% of its range) | −29 ± 13 ms | +59…+69 ms | **+88…+99 ms** (r 0.62–0.92) | +0.20…+0.28 |
| v0.3 (same cycles; timing corrections below) | 0 ± 10–13 ms | 0 ± 14–16 ms | 0 ± 15 ms in-sample; **about ±15–20 ms, MAE ≈ 17 ms, r ≈ 0.84** when the corrections are fitted on half the runners and tested on the other half | 0 ± 0.06 (r ≈ 0.6) |

Step time (hence cadence) error per step: ±15 ms at 33 fps, ±8–10 ms at
60–120 fps, no bias.

v0.2 overestimated contact time by about 0.1 s, which is why duty factor came
out near 1 and flight time near 0 on real videos.  The Python reference
(`rtmpose_pipeline/scripts/extract_running_measures.py`) uses the same v0.2
logic and shows the same symptom (contact 0.39 s, flight 0 on subj04_run81).

### v0.3 corrections

- Initial contact = heel's most forward frame + 29 ms (constant across
  speeds; not proportional to stride time).
- Toe-off = frame at which the toe is furthest behind the pelvis (Zeni-style,
  sub-frame parabolic fit) − 16.6% of the stride time.  This offset scales
  with stride time (r = 0.54) and, unlike v0.2's toe-height threshold, does not
  drift with frame rate or jitter.

Both were fitted to all trials.  Fitted separately on odd and even subject
numbers they agree to 1 ms and 0.01 stride, but the per-half mean error is
+7.5 / −7.7 ms, so expect a runner-dependent bias of about ±8 ms.

## What this does not cover

- Real pose-estimator error.  MediaPipe's heel and foot-index points are not
  the `Heel.Bottom` and MT markers, so the offsets may shift on video; i.i.d.
  jitter also ignores left/right swaps and occlusion of the far leg.
- Overground running, and treadmill speeds outside 2.5–4.5 m/s.
- The spec target for contact time (0.010 s) is not met even with perfect
  points: the kinematic proxies themselves vary by ±15 ms between runners.
  A per-runner correction model (spec section 7-2, stage 3) is the next step.

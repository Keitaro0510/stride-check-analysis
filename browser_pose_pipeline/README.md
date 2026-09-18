# Browser pose pipeline

This is a browser-only replacement for `analysis/step6_app/app/extract/`.
It uses MediaPipe Pose Landmarker in the user's browser, never uploads video,
and implements the `window.StrideExtract.extractMeasures()` contract expected
by Stride Check.

## Install in the public site

Copy `extract/extract.js` over the public site's `extract/extract.js`, then
publish with the normal `63_build_site.py` workflow.  The file dynamically
loads the pinned MediaPipe JavaScript runtime and model from jsDelivr and
Google's public model bucket.  Those are the only network requests; the video
itself is never sent anywhere.

For a fully self-hosted release, download the pinned assets named in the top
of `extract.js` into `extract/vendor/` and set `assetBaseUrl` when calling
`StrideExtract.configure()`.

## What is implemented

- one or two local video `File`s; a side view is required for timing
- MediaPipe Pose Landmarker inference, one person per frame
- frame sampling, quality checks, contact/toe-off event detection, and all
  `measures.json` fields consumed by the existing site
- frontal-plane metrics (CPD, HADD, KA) from a video filmed from behind or,
  with `{frontalView: "front"}`, from the front.  The file is still passed as
  `files.rear`; `video.frontal_view` records `"rear"` or `"front"`.  A front
  view is the mirror image of a rear view, which changes only the sign of KA
  (CPD uses the vertical offset and HADD is unsigned), so KA's lateral
  direction is flipped, as in `rtmpose_pipeline --rear-view front`.  This
  assumes MediaPipe labels the runner's own left and right in both views; it is
  not yet validated against rear-view recordings.
- useful partial results (`null` for measurements which cannot be made)
- progress updates and cancellation through `AbortSignal`

## Pose model

MediaPipe Pose Landmarker **Heavy** (30.7 MB, float16), chosen over Full after
validation against motion capture (`validation/README.md`): better contact
time and foot angle, about 3x slower.  Knee flexion at mid-stance (`KF`) is
reported as `null`: it could not be measured from video (r ≤ 0.5).

## Side-view leg tracking

In side views MediaPipe often labels the legs by position (front = one label,
back = the other), so the labels change legs at every step and the far leg's
points are pulled onto the near leg while the legs cross.  `cleanLegs()`:

1. finds the leg crossings (the feet closest along the running direction),
   made regular with the median interval, and alternates which leg is in front
   at each one (the legs cross once per step);
2. through each crossing, predicts each leg's path from the frames on either
   side (cubic Hermite) and replaces points that are off that path
   (visibility 0.3, drawn hollow in the check view);
3. names the two tracks left/right from the front or rear video: the stance
   foot is the lower one in both views (`nameLegsByFrontal()`), and corrects
   single steps where the views clearly disagree (`fixLegsByFrontal()`);
4. keeps MediaPipe's own labels instead when they match the front/rear video
   better (`resolveSideLegs()`: some runners are labelled correctly, and
   tracking them can only add errors).  Without a
   front/rear video the names may be swapped, and a note says so.

On Subj09 run 99 (8 s, 33 fps) this changed 6/3 detected steps and cadence
104 to 10/10 steps and cadence 165; the tracked legs match the front video's
stance pattern with r = 0.76 (the raw labels: |r| < 0.2).  On correctly
labelled marker data (Fukuchi) it changes nothing.  The page's check view has
a switch to show the uncorrected points.

## Important limits before a public accuracy claim

Browser media APIs do not expose the source frame rate, so the extractor reads
it from the MP4/QuickTime header of each file (`mdhd` timescale and `stts`
frame durations; only the `moov` box is read).  `StrideExtract.probeVideo(file)`
returns `{fps, frames, duration, variable, codec}` or `null`, and the page uses
it to show the detected rate as soon as a file is chosen.

- Header readable: every frame is analysed once, seeking to the middle of each
  frame.  `video.fps_source` is `"metadata"`.  A different `{fps}` passed in
  the options is ignored and noted in `quality.notes`.
- Header unreadable (for example WebM): the `{fps}` option, chosen by the user
  on the page, is used and `fps_source` is `"user"`.
- Neither: frames are sampled at 60 Hz, `fps_source` is `null`, and
  `contact_s`, `flight_s`, `duty`, and `duty_asym_pct` are `null` because
  their time resolution is unknown.  Cadence is still reported.

Below 60 fps a note says that contact and flight times are resolved to one
frame only (30 ms at 33 fps).  The spec's 0.010 s target needs 120 fps or more.
Slow-motion phone files (whose header rate may be the playback rate, not the
capture rate) have not been checked yet.

Only the first 20 seconds of each file are analysed by default
(`{maxSeconds}` option).  At a normal cadence that is still about 25 steps per
leg.  The rear frame used for each side frame is chosen by time, assuming both
recordings start at the same instant; a length mismatch is noted.

Contact and toe-off timing was checked against Fukuchi 2017 force-plate events
using marker-derived points (`validation/README.md`): contact time per trial is
unbiased with about ±15–20 ms error, but this excludes pose-estimator error.
The 2-D angle calculations must still be validated against the marker
reference data.  In particular, rear/side start synchronization,
camera alignment, occlusion of the far leg, and MediaPipe hip-point bias are
not calibrated by this implementation.

## Checks

`node --check extract/extract.js` validates syntax, and
`node tests/test_extract.mjs` checks the front/rear sign convention and the
frame-rate probe.  `probeVideo` also runs in
Node (pass a `Blob` from `fs.openAsBlob`).  Browser testing must be
performed over HTTP(S), not by opening `index.html` as a `file:` URL, because
WebAssembly and dynamic module imports may be blocked for local files.

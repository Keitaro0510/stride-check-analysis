"""Run MediaPipe Pose Landmarker (Python) on a video with the settings extract.js uses.

The browser version is too slow in a headless test browser for whole trials,
so validation runs pose estimation here and the rest of the pipeline
(extract.js: leg tracking, events, angles) in Node on the saved points.
Same package version (mediapipe 1.0.1), model file, running mode and
confidence thresholds as extract.js; JS and Python outputs can still differ
slightly (spec section 7-3), so compare them on one clip before relying on it.

Usage: .venv/bin/python run_pose_py.py VIDEO OUT.json [--seconds S] [--start S] [--model lite|full|heavy]
Output: {"fps", "start_frame", "frames": [[[x, y, vis, z] * 33] | null],
         "world": [[[x, y, z, vis] * 33] | null]}
"""

import argparse
import json
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

HERE = Path(__file__).resolve().parent
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--model", choices=("lite", "full", "heavy"), default="full")
    a = ap.parse_args()
    model = HERE / "outputs" / "models" / f"pose_landmarker_{a.model}.task"
    if not model.exists():
        model.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL.format(v=a.model), model)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    first = int(round(a.start * fps))
    last = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if a.seconds is None else first + int(a.seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model), delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=.55, min_pose_presence_confidence=.55, min_tracking_confidence=.55)
    frames, world = [], []
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        for i in range(first, last):
            ok, bgr = cap.read()
            if not ok:
                break
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            r = lm.detect_for_video(img, int(round((i - first) * 1000 / fps)))
            if r.pose_landmarks:
                frames.append([[p.x, p.y, p.visibility, p.z] for p in r.pose_landmarks[0]])
                world.append([[p.x, p.y, p.z, p.visibility] for p in r.pose_world_landmarks[0]])
            else:
                frames.append(None); world.append(None)
    Path(a.out).write_text(json.dumps({"fps": fps, "start_frame": first, "frames": frames, "world": world}, separators=(",", ":")))
    print(f"{a.out}: {len(frames)} frames at {fps:g} fps, {sum(f is None for f in frames)} without a person")


if __name__ == "__main__":
    main()

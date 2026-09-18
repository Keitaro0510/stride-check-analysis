# StrideCheck analysis

Analysis code behind **StrideCheck**, a prototype built at the
[DBCLS BioHackathon 2026](https://2026.biohackathon.org/): it turns running
videos into gait measures, joint-load estimates and a form-based injury
prediction, entirely in the browser.

- Web page: <https://keitaro0510.github.io/stride-check/>
- Web page source: <https://github.com/Keitaro0510/stride-check>
- Report: <https://github.com/biohackathon-japan/BH26-StrideCheck>

StrideCheck is a research prototype. Its predictions are weak and are not
medical advice.

## Contents

| Folder | What it does |
|---|---|
| `analysis/step1_data/` | Reads the four source datasets into common tables |
| `analysis/step2_video/` | Measurement spec: definitions, directions and target errors of the values measured from video |
| `analysis/step3_types/` | Rhythm and form types, and whether they differ in injury rate |
| `analysis/step4_load/` | Joint-load models from video-equivalent values (Fukuchi markers projected to 2-D) |
| `analysis/step5_evidence/` | Injury prediction with person-wise cross-validation; replication of a published AUC |
| `analysis/step6_app/` | The page's calculations (`app/model.js`, with a Python reference and golden tests) and the site build |
| `browser_pose_pipeline/extract/` | Browser video analysis (`extract.js`): MediaPipe Pose Landmarker, side-view leg tracking, contact and toe-off detection |
| `browser_pose_pipeline/validation/` | Validation against force plates and motion capture (Fukuchi 2017; Wang et al., same videos) |

Each folder has a README with its purpose, inputs, outputs and results.
Most of them are in Japanese; `browser_pose_pipeline/` is in English.
`outputs/` folders hold the summary results (reports, figures, model
files); intermediate tables with one row per runner are not included and are
recreated by `run.sh`.

## Data

**No third-party data is included.**  Download the datasets from their
sources and place them as described in `analysis_data/README.md`
(`analysis_data/SHA256SUMS` lets you check that you have the same files).

| Dataset | Source | License |
|---|---|---|
| Wu et al. 2026, *npj Digital Medicine* | Supplementary files, <https://doi.org/10.1038/s41746-026-02413-y> | CC BY 4.0 |
| Loh et al. 2025, *Int J Sports Med* | NIE Data Repository, <https://doi.org/10.1055/a-2706-5516> | CC BY-NC |
| Loh & Kong 2026, *J Med Biol Eng* | NIE Data Repository, <https://doi.org/10.1007/s40846-026-01008-y> | CC BY-NC |
| Fukuchi et al. 2017, *PeerJ* | <https://doi.org/10.6084/m9.figshare.4543435> | CC BY 4.0 |
| Wang et al. (validation) | Motion capture and forces <https://doi.org/10.5281/zenodo.6457662> → `data/twente_locomotion_6sensors_2022/`; videos <https://doi.org/10.5281/zenodo.6644593>, converted to H.264 MP4 → `Running_mp4/SubjXX/` | see the records |

## Running

Python 3 with numpy, pandas, scipy, scikit-learn, statsmodels, lifelines,
joblib, pyarrow and matplotlib; Node.js for the page's model and the
extractor.  Run the steps in order:

```bash
bash analysis/step1_data/run.sh
bash analysis/step3_types/run.sh
bash analysis/step5_evidence/run.sh
bash analysis/step4_load/run.sh
bash analysis/step6_app/run.sh      # includes the golden tests (node tests/test_model.mjs)
node browser_pose_pipeline/tests/test_extract.mjs
```

The video validation needs MediaPipe in Python (3.10 or later); see
`browser_pose_pipeline/validation/README.md`.

## License

- Code: MIT License (`LICENSE`).
- Results derived from the datasets (files in `outputs/`, and
  `analysis/step6_app/app/data/`) keep the licenses of the datasets they come
  from.  Everything derived from Loh et al. 2025 and Loh & Kong 2026 is
  **CC BY-NC: non-commercial use only**.  Cite the original datasets when you
  use these results.

## Authors

Keitaro Takaoki, Haruma Abe, Teppei Okazaki (Tokyo City University).

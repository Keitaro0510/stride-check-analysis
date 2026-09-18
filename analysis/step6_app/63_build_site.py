"""ステップ6：公開版の Web サイト（GitHub Pages）用のファイル一式を作る。

使い方: python3 63_build_site.py [出力先]   （既定は outputs/site。公開用リポジトリに出すときはそのフォルダを指定）

出力先の .git 以外の中身を作り直す。アーティファクトと違い、スクリプトは埋め込まずに別ファイルのまま置く。
  index.html          英語が先に出る版（EN / 日本語 で切り替え）
  model.js, i18n.js   計算と文言
  data/               app_data.js、サンプル3名の measures.json
  extract/            動画 → measures.json（チームメイトが差し替える。いまは「開発中」を表示する仮のもの）
  docs/               動画計測の仕様書
  README.md, .nojekyll
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE / "app"
SPEC = HERE.parent / "step2_video" / "video_measurement_spec.md"
BROWSER_EXTRACTOR = HERE.parents[1] / "browser_pose_pipeline" / "extract" / "extract.js"

README = """# Stride Check

Prototype from BioHackathon 2026. It shows running-style types, a form-based injury prediction and estimated load on four body parts from running measurements, and sketches how runners could donate data to grow an open prospective dataset.

**Live page:** https://keitaro0510.github.io/{repo}/

- Everything runs in the browser. Nothing you load is sent anywhere.
- Video analysis (`extract/`) runs MediaPipe Pose Landmarker locally in the browser. Videos are not uploaded. It is an experimental measurement feature and is not medical advice.
- The injury prediction is weak (AUC 0.64, 95% CI 0.52–0.76) and is not medical advice.

## Files
| Path | Contents |
|---|---|
| `index.html` | The page (English first, Japanese toggle) |
| `model.js` | All calculations: `StrideModel.evaluate(measures, APP_DATA)` |
| `i18n.js` | Text in English and Japanese |
| `data/app_data.js` | Type cut-offs, prediction and load model coefficients, reference distributions, samples |
| `data/samples/` | Sample `measures.json` files |
| `extract/` | Browser-only video → `measures.json` (MediaPipe Pose Landmarker) |
| `docs/video_measurement_spec.md` | Measurement definitions and requirements (Japanese) |

## Data sources
- Wu et al. 2026, *npj Digital Medicine*: 142 endurance runners, 12-month prospective follow-up (CC BY 4.0). Rhythm types and distributions.
- Loh et al. 2025, *Int J Sports Med*: 81 runners, 12-month prospective follow-up (CC BY-NC). Form types and injury prediction. Only summary statistics and model coefficients are included; no individual records. Non-commercial use only.
- Fukuchi et al. 2017, *PeerJ*: 3D motion and ground reaction forces of 39 runners (CC BY 4.0). Load models and the three sample runners.

The analysis code that produced these models is kept in a separate repository.

## License
- **Code** (`index.html`, `model.js`, `i18n.js`, `extract/`): MIT License, see `LICENSE`. Anyone may use, change and share it, including commercially, as long as the copyright notice is kept.
- **Data** (`data/`): values derived from the datasets above keep those datasets' licenses. The parts derived from Loh et al. 2025 (form types, form-based prediction, form distributions) are **CC BY-NC: non-commercial use only**; the rest is CC BY 4.0. Cite the original datasets when you use them.
- MediaPipe (loaded from its CDN at run time) is Apache 2.0.
"""

LICENSE = """MIT License

Copyright (c) 2026 Keitaro Takaoki, Haruma Abe, Teppei Okazaki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This license covers the code only.  The files in data/ are derived from
third-party datasets and keep their licenses; see README.md.
"""


def main():
    dest = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else HERE / "outputs" / "site"
    repo = dest.name if len(sys.argv) > 1 else "stride-check"
    dest.mkdir(parents=True, exist_ok=True)
    for p in dest.iterdir():
        if p.name == ".git":
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()

    html = (APP / "index.html").read_text(encoding="utf-8")
    assert html.count('data-default-lang="ja"') == 1 and html.count('<html lang="ja"') == 1
    html = html.replace('data-default-lang="ja"', 'data-default-lang="en"').replace('<html lang="ja"', '<html lang="en"')
    (dest / "index.html").write_text(html, encoding="utf-8")
    for f in ("model.js", "i18n.js"):
        shutil.copy2(APP / f, dest / f)
    shutil.copytree(APP / "data", dest / "data")
    shutil.copytree(APP / "extract", dest / "extract")
    if not BROWSER_EXTRACTOR.is_file():
        raise FileNotFoundError(f"Browser extractor not found: {BROWSER_EXTRACTOR}")
    shutil.copy2(BROWSER_EXTRACTOR, dest / "extract" / "extract.js")
    (dest / "docs").mkdir()
    shutil.copy2(SPEC, dest / "docs" / SPEC.name)
    (dest / "README.md").write_text(README.format(repo=repo), encoding="utf-8")
    (dest / "LICENSE").write_text(LICENSE, encoding="utf-8")
    (dest / ".nojekyll").write_text("", encoding="utf-8")

    files = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file() and ".git/" not in p.as_posix() + "/" and not p.relative_to(dest).as_posix().startswith(".git/"))
    total = sum((dest / f).stat().st_size for f in files)
    print(f"{dest}: {len(files)} ファイル、{total / 1024:.0f} KB")
    for f in files:
        print("  " + f)


if __name__ == "__main__":
    main()

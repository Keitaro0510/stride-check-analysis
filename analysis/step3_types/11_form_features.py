"""ステップ3-2：フォームのタイプ分けに使う、脚ごとの特徴量と重みの表を作り、角度の尺度を確認する。

入力: step1_data/outputs/loh2025_limbs.csv, lk2026_limbs.csv
出力: outputs/form_features.csv, outputs/form_scale_check.csv, outputs/form_data_check.txt

脚を単位にする理由:
  ケガをしなかった55名は両脚、ケガをした26名は1名を除き片脚だけに角度がある。1人1行で「両脚の平均」と「片脚の値」を
  混ぜると、平均した値は真ん中に寄るので、ケガをした人が端のタイプに入りやすくなる見かけの差が出る。
  そこで脚を1件とし、重み = 1 ÷（その人の角度がある脚の数）にする（ケガをしなかった人の脚は0.5、片脚の人は1）。
  これは「各人からランダムに片脚を選ぶ」ことの期待値と同じ。信頼区間と安定性は人単位のブートストラップで出す。
角度の向き（出力の注意書きを参照）:
  股関節の内転（HADD）は「値が大きいほど内転が小さい」、膝の外反（KA）は「正のとき膝が外に開く」と解釈する（未確定）。
  Loh & Kong 2026 の健常群は、膝の外反の符号の向きが他と逆の可能性が高い。
速度の補正:
  トレッドミルの速度が 8〜13.5 km/h とばらばらなので、角度ごとに速度への重み付き回帰の傾きを求め、時速10km相当の値も作る（感度分析用）。
"""
import numpy as np
import pandas as pd

from common import FORM_LABELS, FORM_VARIANTS, OUT, STEP1

ANGLES = FORM_VARIANTS["A"]


def wmean(x, w):
    return float(np.average(x, weights=w))


def wsd(x, w):
    m = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - m) ** 2, weights=w) * w.sum() / (w.sum() - 1)))


def main():
    limbs = pd.read_csv(STEP1 / "loh2025_limbs.csv")
    d = limbs[limbs.has_angles].copy()
    d["n_limbs"] = d.groupby("pid").pid.transform("size")
    d["weight"] = 1.0 / d.n_limbs
    d = d[["pid", "side", "injured", "weight", "n_limbs", "male", "age", "height_cm", "mass_kg", "bmi",
           "km_per_week", "run_years", "treadmill_kmh"] + ANGLES].reset_index(drop=True)

    slopes = {}
    X = np.column_stack([np.ones(len(d)), d.treadmill_kmh - 10])
    W = np.diag(d.weight)
    for a in ANGLES:
        beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ d[a].to_numpy())
        slopes[a] = float(beta[1])
        d[f"{a}_adj10"] = d[a] - beta[1] * (d.treadmill_kmh - 10)
    d.to_csv(OUT / "form_features.csv", index=False)

    # 角度の尺度: Loh 2025（ケガをしなかった人の脚、ケガをした人の脚）と Loh & Kong 2026（健常群、受傷群）
    lk = pd.read_csv(STEP1 / "lk2026_limbs.csv")
    rows = []
    groups = {
        "Loh 2025 ケガなし（参加時）": (d[d.injured == 0], True),
        "Loh 2025 ケガあり（参加時、ケガの前）": (d[d.injured == 1], True),
        "Loh & Kong 2026 健常群": (lk[lk.group == "control"], False),
        "Loh & Kong 2026 受傷群（ケガの後）": (lk[lk.group == "injured"], False),
    }
    for name, (g, weighted) in groups.items():
        w = g.weight.to_numpy() if weighted else np.ones(len(g))
        for a in ANGLES:
            x = g[a].to_numpy(float)
            ok = ~np.isnan(x)
            rows.append({"group": name, "angle": a, "label": FORM_LABELS[a], "n_limbs": int(ok.sum()),
                         "mean": wmean(x[ok], w[ok]), "sd": wsd(x[ok], w[ok])})
    scale = pd.DataFrame(rows)
    ref = scale[scale.group == "Loh & Kong 2026 健常群"].set_index("angle")
    scale["smd_vs_lk_control"] = [(r["mean"] - ref.loc[r.angle, "mean"]) / ref.loc[r.angle, "sd"] for _, r in scale.iterrows()]
    scale.to_csv(OUT / "form_scale_check.csv", index=False)

    # 角度どうしの相関（符号の向きの確認）
    corr_rows = []
    for name, (g, _) in groups.items():
        for s in ("L", "R"):
            gg = g[g.side == s].dropna(subset=ANGLES)
            corr_rows.append({"group": name, "side": s, "n_limbs": len(gg),
                              "corr_CPD_HADD": gg.CPD.corr(gg.HADD), "corr_HADD_KA": gg.HADD.corr(gg.KA), "corr_CPD_KA": gg.CPD.corr(gg.KA)})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT / "form_angle_correlations.csv", index=False)

    lines = [
        "== ステップ3-2 データの確認 ==",
        f"角度がある脚: {len(d)} 脚（{d.pid.nunique()} 名）",
        f"ケガなし: {d[d.injured == 0].pid.nunique()} 名 {int((d.injured == 0).sum())} 脚 / ケガあり: {d[d.injured == 1].pid.nunique()} 名 {int((d.injured == 1).sum())} 脚",
        f"重みの合計（人数に相当）: {d.weight.sum():.1f}、ケガあり {d[d.injured == 1].weight.sum():.1f}",
        f"ケガの割合（重み付き）: {wmean(d.injured, d.weight):.1%}",
        f"トレッドミルの速度: {d.treadmill_kmh.min()}〜{d.treadmill_kmh.max()} km/h",
        "速度 1 km/h あたりの角度の変化（重み付き回帰）: " + ", ".join(f"{FORM_LABELS[a]} {s:+.2f}°" for a, s in slopes.items()),
        "",
        "角度の尺度（平均 ± SD、Loh & Kong 2026 健常群との標準化平均差）:",
        scale.pivot(index="group", columns="label", values="mean").round(1).to_string(),
        "",
        scale.pivot(index="group", columns="label", values="smd_vs_lk_control").round(2).to_string(),
        "",
        "角度どうしの相関（脚の左右別）:",
        corr.round(2).to_string(index=False),
        "",
        "注意: Loh & Kong 2026 の健常群だけ、股関節の内転と膝の外反の相関が負（他は正）。健常群のシートは膝の外反の符号の向きが逆の可能性が高い。",
        "      上の標準化平均差のうち、膝の外反は健常群との比較として使えない。",
        "解釈: 骨盤の傾きと股関節の内転の値は、すべてのデータで負の相関。反対側の骨盤が落ちるほど立脚側の内転は大きくなるはずなので、",
        "      股関節の内転の値は「大きいほど内転が小さい（約90°−内転）」、膝の外反の値は「正のとき膝が外に開く」と解釈する（未確定）。",
    ]
    text = "\n".join(lines)
    (OUT / "form_data_check.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

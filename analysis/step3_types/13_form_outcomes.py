"""ステップ3-2：フォームのタイプごとの、12か月でケガをした人の割合を出す。

入力: outputs/form_features.csv, form_assignments.csv, form_models.json
出力:
  outputs/form_outcomes.csv     タイプごとのケガの割合（重み付き）、オッズ比（調整の段階ごと）
  outputs/form_tests.csv        タイプ間の差の検定（人単位の並べ替え検定）
  outputs/form_continuous.csv   角度を連続値のまま入れたロジスティック回帰（1SDあたり）
  outputs/form_sensitivity.csv  感度分析
  outputs/form_power.csv        検出できる差の大きさ
  outputs/form_forest.png, form_map.png, form_scale.png

重み: 脚ごとに 1 ÷ その人の脚の数（ケガをしなかった人の脚は0.5）。信頼区間は人単位のブートストラップ（2000回）。
"""
import importlib
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from common import FORM_LABELS, FORM_SELECTED_VARIANT, FORM_VARIANTS, OUT, SEED, plt, setup_font

warnings.filterwarnings("ignore")
fc = importlib.import_module("12_form_cluster")
N_BOOT_CI = 2000
N_PERM = 2000
ADJUST = {"調整なし": [], "＋性別・年齢": ["male", "age"], "＋週の走行距離": ["male", "age", "km_per_week"]}
COLORS = ["#1F5F8B", "#C47A12", "#2E7D5B", "#8A4F9E"]


def person_boot_indices(pids, rng):
    persons = np.unique(pids)
    rows_of = {p: np.flatnonzero(pids == p) for p in persons}
    pick = rng.choice(persons, len(persons), replace=True)
    return np.concatenate([rows_of[p] for p in pick])


def wrisk(df, col):
    g = df.groupby(col)
    return (g.apply(lambda x: np.average(x.injured, weights=x.weight)), g.weight.sum())


def chi2_stat(df, col):
    """重み付きのタイプ×ケガの表のカイ二乗統計量（並べ替え検定の統計量）。"""
    tab = df.pivot_table(index=col, columns="injured", values="weight", aggfunc="sum", fill_value=0)
    exp = np.outer(tab.sum(1), tab.sum(0)) / tab.values.sum()
    return float(((tab.values - exp) ** 2 / np.where(exp > 0, exp, 1)).sum())


def perm_test(df, col, rng):
    """人単位でケガのラベルを並べ替える（その人の全部の脚が同じラベルを持つ）。"""
    obs = chi2_stat(df, col)
    persons = df.groupby("pid").injured.first()
    count = 0
    for _ in range(N_PERM):
        shuffled = pd.Series(rng.permutation(persons.to_numpy()), index=persons.index)
        dd = df.assign(injured=df.pid.map(shuffled))
        count += chi2_stat(dd, col) >= obs - 1e-12
    return (count + 1) / (N_PERM + 1)


def logit_or(df, col, covars, ref):
    X = pd.get_dummies(df[col]).drop(columns=[ref]).astype(float)
    if covars:
        cov = df[covars].astype(float)
        cov = (cov - cov.mean()) / cov.std()
        X = pd.concat([X, cov], axis=1)
    X = sm.add_constant(X, has_constant="add")
    res = sm.GLM(df.injured.astype(float), X, family=sm.families.Binomial(), freq_weights=df.weight).fit()
    return res.params


def summarize(df, col, variant, analysis, rng, boot=True):
    rows, tests = [], []
    risk, size = wrisk(df, col)
    ref = size.idxmax()
    pids = df.pid.to_numpy()
    boot_risk = {t: [] for t in risk.index}
    boot_or = {adj: {t: [] for t in risk.index if t != ref} for adj in ADJUST}
    point_or = {adj: logit_or(df, col, cov, ref) for adj, cov in ADJUST.items()}
    if boot:
        for _ in range(N_BOOT_CI):
            bd = df.iloc[person_boot_indices(pids, rng)]
            r, _ = wrisk(bd, col)
            for t in boot_risk:
                if t in r.index:
                    boot_risk[t].append(r[t])
            for adj, cov in ADJUST.items():
                try:
                    if bd[col].nunique() < len(risk) or bd.injured.nunique() < 2:
                        continue
                    p = logit_or(bd, col, cov, ref)
                    for t in boot_or[adj]:
                        if t in p.index and abs(p[t]) < 20:
                            boot_or[adj][t].append(p[t])
                except Exception:
                    pass
    for t in risk.index:
        n_eff = size[t]
        k = risk[t] * n_eff
        wl, wh = sm.stats.proportion_confint(k, n_eff, method="wilson")
        lo, hi = (np.percentile(boot_risk[t], [2.5, 97.5]) if boot_risk[t] else (np.nan, np.nan))
        rows.append({"variant": variant, "analysis": analysis, "type": t, "metric": "injury_12m", "est": risk[t], "lo": lo, "hi": hi,
                     "wilson_lo": wl, "wilson_hi": wh, "n_person_equiv": n_eff, "n_limbs": int((df[col] == t).sum()),
                     "injured_person_equiv": k})
    for adj in ADJUST:
        for t, vals in boot_or[adj].items():
            est = np.exp(point_or[adj][t])
            lo, hi = (np.exp(np.percentile(vals, [2.5, 97.5])) if len(vals) > 100 else (np.nan, np.nan))
            rows.append({"variant": variant, "analysis": analysis, "type": t, "metric": f"OR_vs_{ref}（{adj}）", "est": est, "lo": lo, "hi": hi})
    tests.append({"variant": variant, "analysis": analysis, "test": "人単位の並べ替え検定（タイプ×ケガ）", "p": perm_test(df, col, rng)})
    return rows, tests, ref


def refit(df, variant, method, k, cols=None):
    cols = cols or FORM_VARIANTS[variant]
    base = FORM_VARIANTS[variant]
    tmp = df.copy()
    for b, c in zip(base, cols):
        tmp[b] = df[c]
    X = tmp[base].to_numpy(float)
    labels, _, _ = fc.fit_method(method, k, X, tmp.weight.to_numpy(), tmp.pid.to_numpy(), base)
    names = fc.type_names(tmp, labels, tmp.weight.to_numpy())
    return np.array([names[int(t)]["id"] for t in labels])


def main():
    setup_font()
    rng = np.random.default_rng(SEED)
    feats = pd.read_csv(OUT / "form_features.csv")
    assign = pd.read_csv(OUT / "form_assignments.csv")
    models = json.loads((OUT / "form_models.json").read_text(encoding="utf-8"))
    d = feats.merge(assign[["pid", "side"] + [f"type_{v}" for v in FORM_VARIANTS]], on=["pid", "side"])

    rows, tests = [], []
    for v in FORM_VARIANTS:
        r, t, _ = summarize(d, f"type_{v}", v, "主な分析", rng)
        rows += r
        tests += t

    # 連続値（1SDあたり、重み付きロジスティック回帰、人単位ブートストラップ）
    cont = []
    z = d.copy()
    for a in FORM_VARIANTS["A"]:
        m = np.average(z[a], weights=z.weight)
        s = np.sqrt(np.average((z[a] - m) ** 2, weights=z.weight))
        z[a + "_z"] = (z[a] - m) / s
    models_cont = {f"{FORM_LABELS[a]}だけ": [a + "_z"] for a in FORM_VARIANTS["A"]}
    models_cont["4つの角度を一緒に"] = [a + "_z" for a in FORM_VARIANTS["A"]]
    pids = z.pid.to_numpy()
    for label, cols in models_cont.items():
        X = sm.add_constant(z[cols])
        est = sm.GLM(z.injured.astype(float), X, family=sm.families.Binomial(), freq_weights=z.weight).fit().params
        boots = []
        for _ in range(N_BOOT_CI):
            bz = z.iloc[person_boot_indices(pids, rng)]
            try:
                boots.append(sm.GLM(bz.injured.astype(float), sm.add_constant(bz[cols]), family=sm.families.Binomial(),
                                    freq_weights=bz.weight).fit().params)
            except Exception:
                pass
        B = pd.DataFrame(boots)
        for c in cols:
            lo, hi = np.exp(np.percentile(B[c], [2.5, 97.5]))
            p = 2 * min((B[c] > 0).mean(), (B[c] < 0).mean())
            cont.append({"model": label, "term": FORM_LABELS[c[:-2]], "OR_per_SD": np.exp(est[c]), "lo": lo, "hi": hi, "p_boot": p})
    pd.DataFrame(cont).to_csv(OUT / "form_continuous.csv", index=False)

    # 感度分析（画面に使う組み合わせ）
    SV = FORM_SELECTED_VARIANT
    mS = models[SV]
    col = f"type_{SV}"
    sens = []

    def add_sens(label, dd, c):
        r, t, _ = summarize(dd, c, SV, label, rng, boot=True)
        inc = [x for x in r if x["metric"] == "injury_12m"]
        sens.append({"analysis": label, "n_person_equiv": round(dd.weight.sum(), 1),
                     "injury_12m_by_type": "; ".join(f"{x['type']} {x['est']:.0%}（{x['lo']:.0%}〜{x['hi']:.0%}, {x['n_person_equiv']:.1f}人分）" for x in inc),
                     "p_perm": t[0]["p"]})

    # 1人1行・重みなし（偏りの確認）: 角度のある側の平均
    person = d.groupby("pid").agg({**{a: "mean" for a in FORM_VARIANTS["A"]}, "injured": "first", "male": "first", "age": "first",
                                   "km_per_week": "first", "treadmill_kmh": "first"}).reset_index()
    person["weight"] = 1.0
    person["side"] = "mean"
    person["type_p"] = refit(person, SV, mS["method"], mS["k"])
    add_sens("重み付けせず、1人1行（角度のある脚の平均）で分け直す［偏りの確認］", person, "type_p")
    for side, name in (("L", "左脚だけ"), ("R", "右脚だけ")):
        ds = d[d.side == side].copy()
        ds["weight"] = 1.0
        ds["type_s"] = refit(ds, SV, mS["method"], mS["k"])
        add_sens(f"{name}で分け直す", ds, "type_s")
    da = d.copy()
    da["type_adj"] = refit(da, SV, mS["method"], mS["k"], [f"{a}_adj10" for a in FORM_VARIANTS[SV]])
    add_sens("時速10km相当に速度補正した角度で分け直す", da, "type_adj")
    for k_alt in sorted({max(2, mS["k"] - 1), mS["k"] + 1} - {mS["k"]}):
        dk = d.copy()
        dk["type_k"] = refit(dk, SV, mS["method"], k_alt)
        add_sens(f"参考：同じ方法で {k_alt} タイプ", dk, "type_k")
    for other in [v for v in FORM_VARIANTS if v != SV]:
        add_sens(f"参考：組み合わせ{other}（{models[other]['method']}、{models[other]['k']}タイプ）", d, f"type_{other}")
    pd.DataFrame(sens).to_csv(OUT / "form_sensitivity.csv", index=False)

    # 検出できる差（有意水準5%・検出力80%、基準のタイプのケガの割合を固定して、もう一方の割合がいくつなら検出できるか）
    power = []
    out = pd.DataFrame(rows)
    for v in FORM_VARIANTS:
        sub = out[(out.variant == v) & (out.metric == "injury_12m")].set_index("type")
        ref = sub.n_person_equiv.idxmax()
        p0, n0 = sub.loc[ref, "est"], sub.loc[ref, "n_person_equiv"]
        for t in sub.index:
            if t == ref:
                continue
            n1 = sub.loc[t, "n_person_equiv"]
            za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
            det = []
            for direction in (1, -1):
                grid = np.arange(0.001, 0.999, 0.001)
                grid = grid[grid > p0] if direction == 1 else grid[grid < p0][::-1]
                for p1 in grid:
                    pbar = (p0 * n0 + p1 * n1) / (n0 + n1)
                    se0 = np.sqrt(pbar * (1 - pbar) * (1 / n0 + 1 / n1))
                    se1 = np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
                    if abs(p1 - p0) >= za * se0 + zb * se1:
                        det.append(p1)
                        break
            power.append({"variant": v, "comparison": f"{t} vs {ref}", "ref_risk": p0,
                          "detectable_risk_lower": min(det) if det else np.nan, "detectable_risk_upper": max(det) if det else np.nan})
    pd.DataFrame(power).to_csv(OUT / "form_power.csv", index=False)
    out.to_csv(OUT / "form_outcomes.csv", index=False)
    pd.DataFrame(tests).to_csv(OUT / "form_tests.csv", index=False)

    # 図
    sub = out[(out.analysis == "主な分析") & (out.metric == "injury_12m")].sort_values(["variant", "type"])
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(sub) + 1.4))
    for i, r in enumerate(sub.itertuples()):
        y = -i - (0.5 if r.variant == "B" else 0)
        c = COLORS[int(r.type[1:]) - 1]
        ax.plot([r.lo * 100, r.hi * 100], [y, y], color=c, lw=2)
        ax.plot(r.est * 100, y, "o", color=c)
    ax.set_yticks([-i - (0.5 if r.variant == "B" else 0) for i, r in enumerate(sub.itertuples())])
    ax.set_yticklabels([f"{r.variant}-{r.type} {models[r.variant]['types'][r.type]['name']}（{r.n_person_equiv:.0f}人分）" for r in sub.itertuples()], fontsize=9)
    ax.axvline(np.average(d.injured, weights=d.weight) * 100, color="#888", ls="--", lw=1)
    ax.set_xlim(0, 100)
    ax.set_xlabel("12か月でケガをした人の割合（%）と95%信頼区間（人単位のブートストラップ）。破線＝全体")
    ax.set_title("フォームのタイプごとのケガの割合（Loh 2025、81名）")
    ax.grid(axis="x", color="#ddd")
    fig.savefig(OUT / "form_forest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, t in enumerate(sorted(d[col].unique())):
        for inj, mk in ((0, "o"), (1, "X")):
            g = d[(d[col] == t) & (d.injured == inj)]
            ax.scatter(g.CPD, g.HADD, s=np.where(inj, 60, 22), marker=mk, color=COLORS[i], alpha=0.75,
                       label=f"{t} {models[SV]['types'][t]['name']}（{'後でケガ' if inj else 'ケガなし'}）")
    ax.set_xlabel("骨盤の傾き CPD（°）")
    ax.set_ylabel("股関節の内転 HADD の値（°、大きいほど内転が小さいと解釈）")
    ax.set_title(f"フォームのタイプ（組み合わせ{SV}）。×＝後でケガをした人の脚")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "form_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    scale = pd.read_csv(OUT / "form_scale_check.csv")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, a in zip(axes, FORM_VARIANTS["A"]):
        s = scale[scale.angle == a]
        ax.errorbar(s["mean"], range(len(s)), xerr=s["sd"], fmt="o", color="#1F5F8B", capsize=3)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s.group if a == "CPD" else [""] * len(s), fontsize=8)
        ax.set_title(f"{FORM_LABELS[a]}（{a}）")
        ax.invert_yaxis()
    fig.suptitle("角度の平均±SD（膝の外反は Loh & Kong 2026 健常群だけ符号の向きが逆の可能性）", fontsize=10)
    fig.savefig(OUT / "form_scale.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.set_option("display.width", 220)
    print(out[out.analysis == "主な分析"].round(3).to_string(index=False))
    print(pd.DataFrame(tests).round(3).to_string(index=False))
    print(pd.DataFrame(cont).round(3).to_string(index=False))
    print(pd.DataFrame(sens).to_string(index=False))
    print(pd.DataFrame(power).round(2).to_string(index=False))


if __name__ == "__main__":
    main()

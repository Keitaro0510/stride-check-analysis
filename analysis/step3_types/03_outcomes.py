"""ステップ3-1：採用した分け方ごとに、タイプごとのケガの多さを出す。

入力: outputs/rhythm_features.csv, rhythm_assignments.csv, rhythm_models.json
出力:
  outputs/rhythm_outcomes.csv     タイプごとの52週以内のケガの割合、100週あたりのケガの週、ハザード比（調整の段階ごと）
  outputs/rhythm_tests.csv        タイプ間の差の検定（log-rank、Cox の尤度比検定）
  outputs/rhythm_continuous.csv   ピッチと Duty factor を連続値のまま入れた Cox 回帰
  outputs/rhythm_sensitivity.csv  感度分析
  outputs/rhythm_power.csv        検出できるハザード比の大きさ
  outputs/rhythm_km.png, rhythm_forest.png, rhythm_map.png

主な分析の対象: 走り方の値を埋めたと思われる2名を除く140名
"""
import importlib
import json
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy import stats
from statsmodels.discrete.discrete_model import NegativeBinomial

from common import HORIZON_WEEKS, OUT, SELECTED_VARIANT, VARIANTS, VARIANTS_12, plt, setup_font

warnings.filterwarnings("ignore")
cluster = importlib.import_module("02_cluster")

ADJUST = {
    "調整なし": [],
    "＋性別": ["sex"],
    "＋ケガ歴": ["sex", "injury_days_prev_year", "past_stress_injury"],
    "＋練習量": ["sex", "injury_days_prev_year", "past_stress_injury", "run_hours_norm", "past_month_distance_norm"],
}


# ------------------------------------------------------------------ 集計の部品
def km_incidence(d: pd.DataFrame, horizon=HORIZON_WEEKS):
    kmf = KaplanMeierFitter().fit(d.time, d.event)
    s = float(kmf.survival_function_at_times(horizon).iloc[0])
    ci = kmf.confidence_interval_survival_function_
    row = ci[ci.index <= horizon].iloc[-1]
    return 1 - s, 1 - float(row.iloc[1]), 1 - float(row.iloc[0]), kmf


def nb_rates(d: pd.DataFrame, col: str) -> pd.DataFrame:
    X = pd.get_dummies(d[col]).astype(float)
    res = NegativeBinomial(d.n_injury_weeks.to_numpy(float), X, exposure=d.n_weeks.to_numpy(float)).fit(disp=0, maxiter=500)
    ci = res.conf_int()
    out = pd.DataFrame({"type": X.columns, "est": np.exp(res.params[X.columns]) * 100,
                        "lo": np.exp(ci.loc[X.columns, 0]) * 100, "hi": np.exp(ci.loc[X.columns, 1]) * 100})
    # 率がすべて等しいかの尤度比検定（切片だけのモデルと比べる）
    null = NegativeBinomial(d.n_injury_weeks.to_numpy(float), np.ones((len(d), 1)), exposure=d.n_weeks.to_numpy(float)).fit(disp=0, maxiter=500)
    lr = 2 * (res.llf - null.llf)
    return out, float(stats.chi2.sf(lr, X.shape[1] - 1))


def cox_types(d: pd.DataFrame, col: str, covars: list, ref: str):
    dummies = pd.get_dummies(d[col]).drop(columns=[ref]).astype(float)
    base = d[["time", "event"] + covars].copy()
    full = pd.concat([base, dummies], axis=1)
    cph = CoxPHFitter().fit(full, "time", "event")
    if covars:
        red = CoxPHFitter().fit(base, "time", "event")
        lr = 2 * (cph.log_likelihood_ - red.log_likelihood_)
        p = float(stats.chi2.sf(lr, dummies.shape[1]))
    else:
        p = float(cph.log_likelihood_ratio_test().p_value)
    s = cph.summary.loc[dummies.columns]
    hr = pd.DataFrame({"type": dummies.columns, "est": s["exp(coef)"].to_numpy(),
                       "lo": s["exp(coef) lower 95%"].to_numpy(), "hi": s["exp(coef) upper 95%"].to_numpy(), "p": s["p"].to_numpy()})
    return hr, p


def summarize(d: pd.DataFrame, col: str, variant: str, analysis: str):
    """1つの分け方について、タイプごとの結果と検定をまとめる。"""
    rows, tests = [], []
    ref = d[col].value_counts().idxmax()
    for t, g in d.groupby(col):
        inc, lo, hi, _ = km_incidence(g)
        rows.append({"variant": variant, "analysis": analysis, "type": t, "metric": "injury_52w", "est": inc, "lo": lo, "hi": hi,
                     "n": len(g), "events": int(g.event.sum()), "events_52w": int(((g.event == 1) & (g.time <= HORIZON_WEEKS)).sum())})
    rates, p_nb = nb_rates(d, col)
    for _, r in rates.iterrows():
        rows.append({"variant": variant, "analysis": analysis, "type": r.type, "metric": "injury_weeks_per_100", "est": r.est, "lo": r.lo, "hi": r.hi})
    p_lr = multivariate_logrank_test(d.time, d[col], d.event).p_value
    tests.append({"variant": variant, "analysis": analysis, "test": "log-rank（52週以内に限らない、全期間）", "p": p_lr})
    tests.append({"variant": variant, "analysis": analysis, "test": "負の二項回帰（100週あたりのケガの週が等しいか）", "p": p_nb})
    for adj, covars in ADJUST.items():
        hr, p = cox_types(d, col, covars, ref)
        for _, r in hr.iterrows():
            rows.append({"variant": variant, "analysis": analysis, "type": r.type, "metric": f"HR_vs_{ref}（{adj}）",
                         "est": r.est, "lo": r.lo, "hi": r.hi, "p": r.p})
        tests.append({"variant": variant, "analysis": analysis, "test": f"Cox 尤度比検定（{adj}）", "p": p})
    return rows, tests, ref


def refit_labels(df: pd.DataFrame, variant: str, method: str, k: int, cols_override=None):
    """同じ方法・タイプの数で分け直し、ピッチの低い順に R1… と名前を付け直す（感度分析用）。"""
    cols = cols_override or VARIANTS[variant]
    base = VARIANTS[variant]
    tmp = df.copy()
    for b, c in zip(base, cols):  # 12km/h などの列を、名前付けのために基準の列名に置き換える
        tmp[b] = df[c]
    labels, _, _ = cluster.fit_method(method, k, tmp[base].to_numpy(float), base, n_init=50)
    names = cluster.type_names(tmp, labels, variant)
    return np.array([names[int(t)]["id"] for t in labels])


# ------------------------------------------------------------------ 図
COLORS = ["#1F5F8B", "#C47A12", "#2E7D5B", "#8A4F9E", "#B4443A"]


def plot_km(d: pd.DataFrame, col: str, names: dict, path, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (t, g) in enumerate(sorted(d.groupby(col))):
        kmf = KaplanMeierFitter().fit(g.time, g.event, label=f"{t} {names.get(t, '')}（{len(g)}名）")
        kmf.plot_cumulative_density(ax=ax, ci_show=True, color=COLORS[i % len(COLORS)])
    ax.axvline(HORIZON_WEEKS, color="#888", ls="--", lw=1)
    ax.set_xlabel("追跡した週（質問票に回答した週）")
    ax.set_ylabel("初めてケガをした人の割合（累積）")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_forest(out: pd.DataFrame, models: dict, path):
    sub = out[(out.analysis == "主な分析") & (out.metric == "injury_52w")]
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(sub) + 1.6))
    y = 0
    labels = []
    for variant in ["A", "B", "C"]:
        v = sub[sub.variant == variant].sort_values("type")
        for i, r in enumerate(v.itertuples()):
            ax.plot([r.lo * 100, r.hi * 100], [y, y], color=COLORS[i % len(COLORS)], lw=2)
            ax.plot(r.est * 100, y, "o", color=COLORS[i % len(COLORS)])
            labels.append(f"{variant}-{r.type} {models[variant]['types'][r.type]['name']}（{int(r.n)}名）")
            y -= 1
        y -= 0.5
        labels.append("")
    ticks = []
    yy = 0
    for variant in ["A", "B", "C"]:
        for _ in range((sub.variant == variant).sum()):
            ticks.append(yy)
            yy -= 1
        yy -= 0.5
    ax.set_yticks(ticks)
    ax.set_yticklabels([lab for lab in labels if lab], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("52週以内に初めてケガをした割合（%）と95%信頼区間")
    ax.set_title("リズムのタイプごとのケガの割合（npj 2026、140名）")
    ax.grid(axis="x", color="#ddd")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_map(feats: pd.DataFrame, assign: pd.DataFrame, models: dict, path):
    d = feats.merge(assign, on=["person_id", "imputed_running"])
    SV = SELECTED_VARIANT
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for i, t in enumerate(sorted(d[f"type_{SV}"].unique())):
        for rf, mk in ((1, "o"), (0, "^")):
            g = d[(d[f"type_{SV}"] == t) & (d.imputed_running == 0) & (d.rearfoot_10 == rf)]
            ax.scatter(g.cadence_10, g.duty_10, s=np.where(g.event == 1, 34, 16), alpha=0.75, color=COLORS[i], marker=mk,
                       edgecolors=np.where(g.alt_strike == 1, "black", "none"),
                       label=f"{t} {models[SV]['types'][t]['name']}" if rf == 1 else None)
    imp = d[d.imputed_running == 1]
    ax.scatter(imp.cadence_10, imp.duty_10, marker="x", color="#555", label="値を埋めたと思われる2名（除外）")
    ax.set_xlabel("ピッチ（歩/分、時速10km）")
    ax.set_ylabel("Duty factor（接地時間 ÷ 1歩の時間）")
    ax.set_title(f"リズムのタイプ（組み合わせ{SV}）。○踵接地 △それ以外、黒枠＝左右で接地が違う、大きい点＝ケガ", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 本体
def main():
    setup_font()
    feats = pd.read_csv(OUT / "rhythm_features.csv")
    assign = pd.read_csv(OUT / "rhythm_assignments.csv")
    models = json.loads((OUT / "rhythm_models.json").read_text(encoding="utf-8"))
    d_all = feats.merge(assign, on=["person_id", "imputed_running"])
    d = d_all[d_all.imputed_running == 0].reset_index(drop=True)

    rows, tests, refs = [], [], {}
    for variant in VARIANTS:
        r, t, ref = summarize(d, f"type_{variant}", variant, "主な分析")
        rows += r
        tests += t
        refs[variant] = ref

    # 連続値のまま（タイプに分けない）
    cont = []
    z = d.copy()
    for c in ["cadence_10", "duty_10"]:
        z[c + "_z"] = (z[c] - z[c].mean()) / z[c].std()
        z[c + "_z2"] = z[c + "_z"] ** 2
    for label, cols in [("直線（ピッチ・Duty factor、1SDあたり）", ["cadence_10_z", "duty_10_z"]),
                        ("直線＋性別", ["cadence_10_z", "duty_10_z", "sex"]),
                        ("直線＋ケガ歴＋練習量", ["cadence_10_z", "duty_10_z", *ADJUST["＋練習量"]])]:
        cph = CoxPHFitter().fit(z[["time", "event"] + cols], "time", "event")
        for c in ["cadence_10_z", "duty_10_z"]:
            s = cph.summary.loc[c]
            cont.append({"model": label, "term": c, "HR_per_SD": s["exp(coef)"], "lo": s["exp(coef) lower 95%"],
                         "hi": s["exp(coef) upper 95%"], "p": s["p"]})
    lin = CoxPHFitter().fit(z[["time", "event", "cadence_10_z", "duty_10_z"]], "time", "event")
    quad = CoxPHFitter().fit(z[["time", "event", "cadence_10_z", "duty_10_z", "cadence_10_z2", "duty_10_z2"]], "time", "event")
    p_quad = stats.chi2.sf(2 * (quad.log_likelihood_ - lin.log_likelihood_), 2)
    cont.append({"model": "曲線（2乗の項を足したときの尤度比検定）", "term": "cadence_z2 + duty_z2", "p": p_quad})
    pd.DataFrame(cont).to_csv(OUT / "rhythm_continuous.csv", index=False)

    # 感度分析（画面に使う組み合わせ、採用した方法とタイプの数のまま）
    SV = SELECTED_VARIANT
    mS = models[SV]
    col = f"type_{SV}"
    sens = []

    def add_sens(label, dd, c):
        r, t, _ = summarize(dd, c, SV, label)
        inc = [x for x in r if x["metric"] == "injury_52w"]
        sens.append({"analysis": label, "n": len(dd),
                     "injury_52w_by_type": "; ".join(f"{x['type']} {x['est']:.0%}（{x['lo']:.0%}〜{x['hi']:.0%}, {x['n']}名）" for x in inc),
                     "p_logrank": next(x["p"] for x in t if x["test"].startswith("log-rank")),
                     "p_cox_unadjusted": next(x["p"] for x in t if x["test"] == "Cox 尤度比検定（調整なし）")})

    add_sens("追跡4週未満の人を除く", d[d.n_weeks >= 4], col)
    add_sens("追跡の最初の週にケガがあった人を除く", d[d.injured_week1 == 0], col)
    add_sens("値を埋めたと思われる2名も含める", d_all, col)
    d12 = d.copy()
    d12["type_12"] = refit_labels(d12, SV, mS["method"], mS["k"], VARIANTS_12[SV])
    add_sens("時速12kmの値で分け直す", d12, "type_12")
    dh = d.copy()
    X = np.column_stack([np.ones(len(dh)), dh.height_norm, dh.sex])
    beta = np.linalg.lstsq(X, dh.cadence_10, rcond=None)[0]
    dh["cadence_10"] = dh.cadence_10 - X @ beta + dh.cadence_10.mean()
    dh["type_h"] = refit_labels(dh, SV, mS["method"], mS["k"])
    add_sens("身長と性別で補正したピッチで分け直す", dh, "type_h")
    for k_alt in sorted({max(2, mS["k"] - 1), mS["k"] + 1} - {mS["k"]}):
        alt = d.copy()
        alt["type_alt"] = refit_labels(alt, SV, mS["method"], k_alt)
        add_sens(f"参考：同じ方法で {k_alt} タイプ", alt, "type_alt")
    for other in [v for v in VARIANTS if v != SV]:
        add_sens(f"参考：組み合わせ{other}（{models[other]['method']}、{models[other]['k']}タイプ）", d, f"type_{other}")
    pd.DataFrame(sens).to_csv(OUT / "rhythm_sensitivity.csv", index=False)

    # 検出できるハザード比（有意水準5%・検出力80%、基準のタイプとの比較）
    power = []
    for variant in VARIANTS:
        col = f"type_{variant}"
        ref = refs[variant]
        g_ref = d[d[col] == ref]
        for t, g in d.groupby(col):
            if t == ref:
                continue
            D = g.event.sum() + g_ref.event.sum()
            p = len(g) / (len(g) + len(g_ref))
            hr = np.exp((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) / np.sqrt(D * p * (1 - p)))
            power.append({"variant": variant, "comparison": f"{t} vs {ref}", "events": int(D), "detectable_HR": hr})
    pd.DataFrame(power).to_csv(OUT / "rhythm_power.csv", index=False)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "rhythm_outcomes.csv", index=False)
    pd.DataFrame(tests).to_csv(OUT / "rhythm_tests.csv", index=False)

    names_S = {t: v["name"] for t, v in models[SELECTED_VARIANT]["types"].items()}
    plot_km(d, f"type_{SELECTED_VARIANT}", names_S, OUT / "rhythm_km.png", f"リズムのタイプ（組み合わせ{SELECTED_VARIANT}）ごとの、初めてのケガまでの期間")
    plot_forest(out, models, OUT / "rhythm_forest.png")
    plot_map(feats, assign, models, OUT / "rhythm_map.png")

    pd.set_option("display.width", 200)
    print(out[(out.analysis == "主な分析") & out.metric.isin(["injury_52w", "injury_weeks_per_100"])]
          [["variant", "type", "metric", "est", "lo", "hi", "n", "events", "events_52w"]].round(3).to_string(index=False))
    print(pd.DataFrame(tests).round(3).to_string(index=False))
    print(out[out.metric.str.startswith("HR")][["variant", "type", "metric", "est", "lo", "hi", "p"]].round(3).to_string(index=False))
    print(pd.DataFrame(cont).round(3).to_string(index=False))
    print(pd.DataFrame(sens).to_string(index=False))
    print(pd.DataFrame(power).round(2).to_string(index=False))


if __name__ == "__main__":
    main()

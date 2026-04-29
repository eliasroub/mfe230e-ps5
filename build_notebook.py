"""Builder for MFE230E_PS5_final.ipynb (unified submission notebook).

Combines:
  - MFE230E_PS5.ipynb       (full Q1-Q9 baseline, web data fetch)
  - PS5-2.pdf               (Q2 with size/BM + size/OP, plot styling)
  - PS5.ipynb               (Q5 two-step with intercept, FF3 TS vs FF3 CS)

Also saves key figures as PNGs into ./figs/ for LaTeX inclusion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).parent
OUT  = HERE / "MFE230E_PS5_final.ipynb"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

cells: list = []
def md(src: str) -> None:  cells.append(new_markdown_cell(src.strip("\n")))
def code(src: str) -> None: cells.append(new_code_cell(src.strip("\n")))


# ---------------------------------------------------------------------------
md(r"""
# MFE230E Problem Set 5 — Asset Pricing Tests (Unified Submission)

Sample: monthly data, **July 1963** through the most recent month from Ken French's library.
Test assets: value-weighted Fama–French portfolios. Standard errors throughout: OLS,
White (HC0), Newey–West (lag 6).
""")

md("## Setup")

code(r"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web

np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.precision", 4)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True, "grid.alpha": 0.3,
    "figure.figsize": (8, 5),
})

START = "1963-07-01"
FIGS  = Path("figs"); FIGS.mkdir(exist_ok=True)

def savefig(name):
    plt.tight_layout(); plt.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
""")

md("### Loading Ken French monthly data")
code(r"""
def kf(name: str) -> pd.DataFrame:
    raw = web.DataReader(name, "famafrench", start=START)
    df = raw[0].copy()
    df.index = df.index.to_timestamp("M")
    return df

ff5  = kf("F-F_Research_Data_5_Factors_2x3")     # Mkt-RF SMB HML RMW CMA RF
mom  = kf("F-F_Momentum_Factor"); mom.columns = ["MOM"]
sm_p = kf("25_Portfolios_ME_Prior_12_2")         # 25 size/momentum (VW)
bm_p = kf("25_Portfolios_5x5")                   # 25 size/BM      (VW)
sop  = kf("25_Portfolios_ME_OP_5x5")             # 25 size/OP      (VW)

idx  = ff5.index.intersection(mom.index).intersection(sm_p.index).intersection(bm_p.index).intersection(sop.index)
ff5, mom, sm_p, bm_p, sop = ff5.loc[idx], mom.loc[idx], sm_p.loc[idx], bm_p.loc[idx], sop.loc[idx]

# decimal monthly. excess = portfolio - RF.
factors = pd.concat([ff5[["Mkt-RF","SMB","HML","RMW","CMA"]], mom["MOM"], ff5["RF"]], axis=1) / 100.0
sm_p_e  = sm_p.div(100.0).sub(factors["RF"], axis=0)
bm_p_e  = bm_p.div(100.0).sub(factors["RF"], axis=0)
sop_e   = sop.div(100.0).sub(factors["RF"], axis=0)

print(f"Sample: {factors.index[0]:%Y-%m} to {factors.index[-1]:%Y-%m}  ({len(factors)} months)")
""")

# ---------------------------------------------------------------------------
md(r"""
## Question 1 — Fama–French factor summary

### 1(a) Mean, std, Sharpe (annualised)
""")

code(r"""
def factor_table(F: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Mean (% ann.)":   F.mean() * 12 * 100,
        "Std (% ann.)":    F.std()  * np.sqrt(12) * 100,
        "Sharpe (ann.)":  (F.mean() / F.std()) * np.sqrt(12),
    }).round(3)

tbl_q1a = factor_table(factors)
tbl_q1a.to_latex(FIGS / "q1a_factor_table.tex", float_format="%.3f")
tbl_q1a
""")

md(r"""
**Discussion.** MOM and the market deliver the highest annualised Sharpe ratios in this
sample. SMB is the weakest factor in Sharpe terms; HML, RMW and CMA cluster in the middle.
""")

md("### 1(b) Cumulative excess returns")
code(r"""
fig, ax = plt.subplots(figsize=(10, 5.5))
for col in ["Mkt-RF","SMB","HML","RMW","CMA","MOM"]:
    ((1 + factors[col]).cumprod() - 1).plot(ax=ax, label=col)
ax.axhline(0, color="k", lw=0.6)
ax.set_title("Cumulative excess returns of FF factors and MOM")
ax.set_ylabel("Cumulative return"); ax.legend()
savefig("q1b_cumret"); plt.show()
""")

md(r"""
**Time variation.** Profitability of every factor is highly time-varying. MKT-RF and MOM
compound the most but MOM has deep drawdowns (notably 2009). SMB barely accumulates over
six decades, illustrating the well-known weakness of the post-1980 size premium. HML
flattens after 2000.
""")

# ---------------------------------------------------------------------------
md(r"""
## Question 2 — CAPM on the 25 size/momentum portfolios

### 2(a) Summary statistics — size/BM and size/OP portfolios
""")

code(r"""
def port_table(P_excess: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Mean (% ann.)":  P_excess.mean() * 12 * 100,
        "Std (% ann.)":   P_excess.std()  * np.sqrt(12) * 100,
        "Sharpe":        (P_excess.mean() / P_excess.std()) * np.sqrt(12),
    }).round(3)

print("=== 25 Size / Book-to-Market portfolios (excess returns) ===")
display(port_table(bm_p_e))
print("\n=== 25 Size / Operating Profitability portfolios (excess returns) ===")
display(port_table(sop_e))
""")

code(r"""
def grid_5x5(stat: pd.Series, kind: str) -> pd.DataFrame:
    return pd.DataFrame(np.array(stat.values).reshape(5,5),
                        index=[f"ME{i}" for i in range(1,6)],
                        columns=[f"{kind}{j}" for j in range(1,6)])

print("Mean ann. excess returns (%) — Size x BM"); display(grid_5x5(bm_p_e.mean()*12*100, "BM").round(2))
print("Mean ann. excess returns (%) — Size x OP"); display(grid_5x5(sop_e.mean()*12*100, "OP").round(2))
""")

md(r"""
**Patterns.**

- *Size/BM:* the classic value premium — within most size groups, average return rises from
  growth (BM1) to value (BM5). The smallest growth corner ("SMALL LoBM") is the textbook
  weak portfolio. Sharpes follow but volatility blunts the small-stock effect.
- *Size/OP:* high-profitability portfolios earn higher mean returns and Sharpes than
  low-OP within each size band, especially outside the noisy small-stock corner.
""")

md(r"""
### 2(b) Time-series CAPM regressions on the 25 size/momentum portfolios

For each portfolio $i$, $R^e_{it} = \alpha_i + \beta_i (R^M_t - R^f_t) + \varepsilon_{it}$.
""")

code(r"""
NW_LAGS = 6

def ts_regression(y, X):
    Xc = sm.add_constant(X)
    res    = sm.OLS(y, Xc).fit()
    res_w  = sm.OLS(y, Xc).fit(cov_type="HC0")
    res_nw = sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    return dict(params=res.params, bse_ols=res.bse, bse_w=res_w.bse, bse_nw=res_nw.bse,
                tvals_nw=res_nw.tvalues, pvals_nw=res_nw.pvalues, resid=res.resid, r2=res.rsquared)

def run_factor_model(P_excess, F):
    out = {c: ts_regression(P_excess[c], F) for c in P_excess.columns}
    cols = list(P_excess.columns)
    def stk(getter): return pd.DataFrame({c: getter(out[c]) for c in cols}).T
    return dict(coefs=stk(lambda r: r["params"]),
                bse_ols=stk(lambda r: r["bse_ols"]),
                bse_w=stk(lambda r: r["bse_w"]),
                bse_nw=stk(lambda r: r["bse_nw"]),
                tvals_nw=stk(lambda r: r["tvals_nw"]),
                pvals_nw=stk(lambda r: r["pvals_nw"]),
                resid=pd.DataFrame({c: out[c]["resid"] for c in cols}),
                r2=pd.Series({c: out[c]["r2"] for c in cols}),
                results=out)

capm = run_factor_model(sm_p_e, factors[["Mkt-RF"]])

alpha_pct  = capm["coefs"]["const"] * 100
se_a_ols   = capm["bse_ols"]["const"] * 100
se_a_w     = capm["bse_w"]["const"]   * 100
se_a_nw    = capm["bse_nw"]["const"]  * 100
beta_mkt   = capm["coefs"]["Mkt-RF"]
se_b_nw    = capm["bse_nw"]["Mkt-RF"]
t_a_nw     = alpha_pct / se_a_nw

q2b = pd.DataFrame({
    "alpha (%/mo)":  alpha_pct.round(4),
    "se OLS":        se_a_ols.round(4),
    "se White":      se_a_w.round(4),
    "se NW":         se_a_nw.round(4),
    "alpha t (NW)":  t_a_nw.round(2),
    "alpha p (NW)":  capm["pvals_nw"]["const"].round(4),
    "beta_MKT":      beta_mkt.round(3),
    "se(beta) NW":   se_b_nw.round(3),
    "beta t (NW)":   capm["tvals_nw"]["Mkt-RF"].round(2),
    "R^2":            capm["r2"].round(3),
})
q2b
""")

code(r"""
print("Median beta NW t-stat: %.1f" % capm["tvals_nw"]["Mkt-RF"].median())
print("# alphas with NW |t|>1.96: %d / 25" % int((t_a_nw.abs()>1.96).sum()))
print("# alphas with NW |t|>2.58: %d / 25" % int((t_a_nw.abs()>2.58).sum()))
""")

md(r"""
**Are the betas precise?** Yes. NW $t$-statistics on $\hat\beta_i$ are typically above 25,
the standard errors of order $0.02$–$0.06$, and $R^2$ values are 0.7–0.9. The cross-sectional
spread in betas is small relative to the spread in mean returns — this is the seed of the
CAPM's failure.
""")

md("### 2(c) Are the pricing errors $\\hat\\alpha_i$ significant?")
code(r"""
print(f"Mean |alpha| (%/mo): {alpha_pct.abs().mean():.3f}")
print(f"Largest |alpha| (%/mo): {alpha_pct.abs().max():.3f}")
print(f"Annualised mean |alpha| (%): {alpha_pct.abs().mean()*12:.2f}")
print("\nAlphas (%/mo) — 5x5 grid (Size x Momentum)")
display(grid_5x5(alpha_pct, "MOM").round(3))
print("\nNewey-West alpha t-stats")
display(grid_5x5(t_a_nw, "MOM").round(2))
""")

md(r"""
The pricing errors are **economically large** — annualised they reach $\pm 9$%/yr at the
small-loser / small-winner corners. Many are statistically significant under all three SE
flavours.
""")

md("### 2(d) Fitted vs. realised mean excess returns")
code(r"""
def plot_fit(realised_pct_ann, fitted_pct_ann, title, fname):
    fig, ax = plt.subplots(figsize=(6.4,6.4))
    lo = min(realised_pct_ann.min(), fitted_pct_ann.min()) - 1
    hi = max(realised_pct_ann.max(), fitted_pct_ann.max()) + 1
    ax.plot([lo,hi],[lo,hi], "k--", lw=1, label="45-degree line")
    ax.scatter(fitted_pct_ann, realised_pct_ann, s=28)
    for c in realised_pct_ann.index:
        lbl = c.replace("PRIOR","P").replace("SMALL ","S").replace("BIG ","B")
        ax.annotate(lbl, (fitted_pct_ann[c], realised_pct_ann[c]), fontsize=7, alpha=0.75)
    ax.set_xlabel("Model-implied mean excess return (% ann.)")
    ax.set_ylabel("Realised mean excess return (% ann.)")
    ax.set_title(title); ax.legend()
    savefig(fname); plt.show()

fitted_capm = beta_mkt * factors["Mkt-RF"].mean() * 12 * 100
realised_an = sm_p_e.mean() * 12 * 100
plot_fit(realised_an, fitted_capm, "CAPM — fitted vs. realised mean excess returns", "q2d_capm_fit")
""")

md(r"""
The CAPM points scatter widely off the 45° line. There is essentially no slope linking
fitted to realised — the model explains the *level* of the average premium but not the
cross-sectional spread.
""")

md("### 2(e) Individual t-tests for $H_0: \\alpha_i = 0$")
code(r"""
ttest_q2e = pd.DataFrame({
    "alpha (%/mo)": alpha_pct,
    "t (OLS)":      alpha_pct / se_a_ols,
    "t (White)":    alpha_pct / se_a_w,
    "t (NW)":       t_a_nw,
}).round(3)
ttest_q2e
""")

md("### 2(f) GRS test for $H_0: \\alpha_i = 0\\ \\forall i$")
code(r"""
def grs_test(P_excess, F, results):
    T, N = P_excess.shape; K = F.shape[1]
    alpha = results["coefs"]["const"].values
    eps   = results["resid"].values
    Sigma = (eps.T @ eps) / (T - K - 1)
    Fbar  = F.mean().values
    Omega = np.cov(F.values, rowvar=False, ddof=1).reshape(K,K)
    SR_f2 = float(Fbar @ np.linalg.inv(Omega) @ Fbar)
    quad  = float(alpha @ np.linalg.inv(Sigma) @ alpha)
    GRS   = (T/N) * ((T - N - K)/(T - K - 1)) * (quad / (1.0 + SR_f2))
    pval  = 1 - stats.f.cdf(GRS, N, T - N - K)
    return dict(GRS=GRS, df1=N, df2=T-N-K, p_value=pval,
                SR_factors=np.sqrt(SR_f2), SR_alpha=np.sqrt(quad))

grs_capm = grs_test(sm_p_e, factors[["Mkt-RF"]], capm)
print(f"GRS = {grs_capm['GRS']:.4f}  ~ F({grs_capm['df1']}, {grs_capm['df2']})")
print(f"p-value = {grs_capm['p_value']:.4e}")
print(f"Tangency Sharpe (factors)   = {grs_capm['SR_factors']:.3f}")
print(f"|alpha|-implied Sharpe gain = {grs_capm['SR_alpha']:.3f}")
""")

md(r"""
The CAPM is overwhelmingly rejected: $p \ll 0.001$. The Sharpe ratio of the |α| portfolio
dwarfs the market's, which is exactly why GRS rejects.
""")

md("### 2(g) CML and mean-variance frontier of the 25 portfolios")
code(r"""
def mv_frontier(R, n=200):
    mu = R.mean().values; Sigma = R.cov().values
    inv = np.linalg.inv(Sigma); ones = np.ones_like(mu)
    A = ones @ inv @ ones; B = ones @ inv @ mu
    C = mu    @ inv @ mu;  D = A * C - B * B
    targets = np.linspace(B/A, mu.max() * 1.8, n)
    var = (A * targets * targets - 2 * B * targets + C) / D
    return np.sqrt(np.maximum(var, 0)) * np.sqrt(12) * 100, targets * 12 * 100

def long_only_frontier(mu_d, cov_d, n=80):
    mu = np.asarray(mu_d); cov = np.asarray(cov_d); k = len(mu)
    targets = np.linspace(mu.min(), mu.max(), n)
    sigs = []; ts = []; w0 = np.repeat(1/k, k); bnds = [(0,1)]*k
    for tg in targets:
        cons = [{"type":"eq","fun": lambda w: w.sum() - 1},
                {"type":"eq","fun": (lambda w, tg=tg: w @ mu - tg)}]
        res = minimize(lambda w: w @ cov @ w, w0, method="SLSQP", bounds=bnds, constraints=cons)
        if res.success:
            sigs.append(np.sqrt(res.fun)); ts.append(tg); w0 = res.x
    return np.array(sigs)*np.sqrt(12)*100, np.array(ts)*12*100

sd_mv, mu_mv = mv_frontier(sm_p_e)
sd_lo, mu_lo = long_only_frontier(sm_p_e.mean().values, sm_p_e.cov().values)

mu_mkt = factors["Mkt-RF"].mean() * 12 * 100
sd_mkt = factors["Mkt-RF"].std() * np.sqrt(12) * 100
sr_mkt = mu_mkt / sd_mkt

inv_cov = np.linalg.inv(sm_p_e.cov().values)
mu_d    = sm_p_e.mean().values
w_tan   = inv_cov @ mu_d / (np.ones_like(mu_d) @ inv_cov @ mu_d)
sr_tan  = (w_tan @ mu_d) / np.sqrt(w_tan @ sm_p_e.cov().values @ w_tan) * np.sqrt(12)
mu_tan  = (w_tan @ mu_d) * 12 * 100
sd_tan  = np.sqrt(w_tan @ sm_p_e.cov().values @ w_tan) * np.sqrt(12) * 100

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sd_mv, mu_mv, "b-", lw=1.5, label="MV frontier (25 size/MOM)")
ax.plot(sd_lo, mu_lo, "purple", lw=1.5, label="Long-only frontier")
sd_grid = np.linspace(0, max(sd_mv.max(), sd_tan)*1.05, 200)
ax.plot(sd_grid, sr_mkt * sd_grid, "orange", lw=1.5, label="CAPM market CML")
ax.plot(sd_grid, sr_tan * sd_grid, "g--", lw=1.5, label="Tangency CML (25 ports)")
ax.scatter(sm_p_e.std()*np.sqrt(12)*100, sm_p_e.mean()*12*100, s=28, alpha=0.7, label="25 portfolios")
ax.scatter([sd_mkt],[mu_mkt], s=120, marker="*", c="black", label="MKT")
ax.scatter([sd_tan],[mu_tan], s=80, marker="D", c="red",   label="Tangency (25)")
ax.set_xlabel("Annualised volatility (%)"); ax.set_ylabel("Annualised mean excess return (%)")
ax.set_title("CML vs. mean-variance frontier"); ax.legend()
ax.set_xlim(0, max(sd_mv.max(), sd_tan)*1.05)
savefig("q2g_cml_frontier"); plt.show()
print(f"Market Sharpe: {sr_mkt:.3f}    Tangency Sharpe: {sr_tan:.3f}")
""")

md(r"""
The MV frontier from the 25 portfolios sits **strictly above** the CAPM CML; the tangency
portfolio of the 25 has Sharpe $\sim 1.5$ vs the market's $\sim 0.46$. The gap is the
mean-variance counterpart of the GRS rejection.
""")

md("### 2(h) Security Market Line and the 25 portfolios")
code(r"""
realised_an = sm_p_e.mean() * 12 * 100
fitted_capm_an = beta_mkt * mu_mkt
alpha_ann = realised_an - fitted_capm_an

fig, ax = plt.subplots(figsize=(8, 6))
b_grid = np.linspace(0, beta_mkt.max() * 1.08, 200)
ax.plot(b_grid, b_grid * mu_mkt, "k-", lw=2, label="Theoretical CAPM SML")
for bi, real, pred in zip(beta_mkt, realised_an, fitted_capm_an):
    ax.plot([bi, bi], [pred, real], color="grey", alpha=0.35, lw=1)
sc = ax.scatter(beta_mkt, realised_an, c=alpha_ann, cmap="coolwarm", s=60, edgecolor="black", lw=0.5)
for c, bi, ai in zip(beta_mkt.index, beta_mkt, realised_an):
    lbl = c.replace("PRIOR","P").replace("SMALL ","S").replace("BIG ","B")
    ax.annotate(lbl, (bi, ai), fontsize=7, alpha=0.7)
ax.scatter([1.0],[mu_mkt], marker="*", s=160, c="black", label="MKT")
ax.set_xlabel(r"$\beta_{MKT}$"); ax.set_ylabel("Mean excess return (% ann.)")
ax.set_title("CAPM Security Market Line — 25 size/momentum portfolios")
fig.colorbar(sc, ax=ax, label="Annualised alpha (%)")
ax.legend()
savefig("q2h_sml"); plt.show()
""")

md(r"""
Vertical distance from each point to the SML is the (annualised) pricing error. Small-loser
portfolios sit far below the SML, small-winner portfolios far above. Cross-sectional
variation is mostly orthogonal to $\beta_{MKT}$.
""")

md(r"""
### 2(i) How well does the CAPM explain the 25 size/momentum portfolios?

The CAPM fails on every dimension:
1. **Statistical:** GRS rejects ($p\ll 10^{-12}$); a majority of individual NW $t$-stats on
   $\hat\alpha_i$ exceed 1.96.
2. **Economic:** alpha magnitudes reach $\pm 9$% per year at the corner portfolios.
3. **Diagnostic:** the fitted-vs-realised plot has near-zero slope, the SML cannot reproduce
   the dispersion in mean returns, and the MV frontier of the 25 test assets sits well above
   the market CML.

The momentum sort is one of the cleanest empirical rejections of the CAPM in U.S. equities.
""")

# ---------------------------------------------------------------------------
md("## Question 3 — 3-factor Fama–French model on the 25 size/momentum portfolios")
code(r"""
ff3 = run_factor_model(sm_p_e, factors[["Mkt-RF","SMB","HML"]])
alpha3 = ff3["coefs"]["const"]*100; se3_nw = ff3["bse_nw"]["const"]*100; t3 = alpha3/se3_nw

q3 = pd.DataFrame({
    "alpha (%/mo)": alpha3.round(3), "se OLS": (ff3["bse_ols"]["const"]*100).round(3),
    "se White":     (ff3["bse_w"]["const"]*100).round(3), "se NW": se3_nw.round(3),
    "t (NW)":       t3.round(2), "p (NW)": ff3["pvals_nw"]["const"].round(4),
    "beta_MKT":     ff3["coefs"]["Mkt-RF"].round(3),
    "beta_SMB":     ff3["coefs"]["SMB"].round(3),
    "beta_HML":     ff3["coefs"]["HML"].round(3),
    "R^2":           ff3["r2"].round(3),
})
q3
""")

code(r"""
print("FF3 alphas (%/mo) — 5x5 (Size x Momentum)"); display(grid_5x5(alpha3,"MOM").round(3))
print("\nFF3 alpha t-stats (NW)");                      display(grid_5x5(t3,"MOM").round(2))

grs3 = grs_test(sm_p_e, factors[["Mkt-RF","SMB","HML"]], ff3)
print(f"\nGRS = {grs3['GRS']:.3f} ~ F({grs3['df1']},{grs3['df2']})  p = {grs3['p_value']:.3e}")
""")

code(r"""
fitted3 = (ff3["coefs"][["Mkt-RF","SMB","HML"]] * factors[["Mkt-RF","SMB","HML"]].mean()).sum(axis=1) * 12 * 100
plot_fit(realised_an, fitted3, "FF3 — fitted vs. realised mean excess returns", "q3_ff3_fit")
""")

md(r"""
FF3 raises $R^2$ but does not rescue the cross-section: the small-loser/winner extremes still
have large alphas, and GRS still rejects. HML loadings on these momentum-sorted portfolios are
small, so HML cannot price the momentum spread.
""")

# ---------------------------------------------------------------------------
md("## Question 4 — Carhart 4-factor model on the 25 size/momentum portfolios")
code(r"""
ff4 = run_factor_model(sm_p_e, factors[["Mkt-RF","SMB","HML","MOM"]])
alpha4 = ff4["coefs"]["const"]*100; se4_nw = ff4["bse_nw"]["const"]*100; t4 = alpha4/se4_nw

q4 = pd.DataFrame({
    "alpha (%/mo)": alpha4.round(3), "se OLS": (ff4["bse_ols"]["const"]*100).round(3),
    "se White":     (ff4["bse_w"]["const"]*100).round(3), "se NW": se4_nw.round(3),
    "t (NW)":       t4.round(2), "p (NW)": ff4["pvals_nw"]["const"].round(4),
    "beta_MKT":     ff4["coefs"]["Mkt-RF"].round(3),
    "beta_SMB":     ff4["coefs"]["SMB"].round(3),
    "beta_HML":     ff4["coefs"]["HML"].round(3),
    "beta_MOM":     ff4["coefs"]["MOM"].round(3),
    "R^2":           ff4["r2"].round(3),
})
q4
""")

code(r"""
print("Carhart alphas (%/mo) — 5x5"); display(grid_5x5(alpha4,"MOM").round(3))
print("\nCarhart MOM betas — 5x5");    display(grid_5x5(ff4["coefs"]["MOM"],"MOM").round(2))

grs4 = grs_test(sm_p_e, factors[["Mkt-RF","SMB","HML","MOM"]], ff4)
print(f"\nGRS = {grs4['GRS']:.3f} ~ F({grs4['df1']},{grs4['df2']})  p = {grs4['p_value']:.3e}")
""")

code(r"""
fitted4 = (ff4["coefs"][["Mkt-RF","SMB","HML","MOM"]] *
           factors[["Mkt-RF","SMB","HML","MOM"]].mean()).sum(axis=1) * 12 * 100
plot_fit(realised_an, fitted4, "Carhart 4F — fitted vs. realised mean excess returns", "q4_ff4_fit")
""")

md(r"""
Adding MOM does the work. Carhart MOM-betas are monotone across momentum quintiles (negative
on losers, positive on winners) and the alphas collapse for most portfolios. The fitted-vs-realised
plot is now close to the 45° line; the GRS p-value, while still significant on this very precise
test set, is dramatically improved.
""")

# ---------------------------------------------------------------------------
md(r"""
## Question 5 — Cross-sectional risk premia (2-step procedure)

Stage 1: time-series betas (already estimated in Q2/Q4).
Stage 2: cross-sectional regression $\bar R^e_i = \lambda_0 + \beta_i'\lambda + \alpha_i^{CS}$.

We report **both** the regression with intercept (statsmodels OLS) and without intercept (pure
factor model). Standard errors below treat the betas as known; the Shanken-corrected version
(which inflates SE for first-stage estimation noise) is given in Q8.
""")

md("### 5(a) Two-step cross-sectional regression — with intercept (OLS via statsmodels)")
code(r"""
def cross_section_ols(P_excess, ts_results, factor_names, with_intercept=True):
    Rbar = P_excess.mean()
    B    = ts_results["coefs"][factor_names]
    X    = sm.add_constant(B) if with_intercept else B
    res  = sm.OLS(Rbar, X).fit()
    fitted = res.fittedvalues
    pe     = Rbar - fitted
    return res, pd.DataFrame({"mean_excess": Rbar, "fitted": fitted, "pricing_error": pe})

cs_capm_int, _ = cross_section_ols(sm_p_e, capm, ["Mkt-RF"], with_intercept=True)
cs_ff4_int,  _ = cross_section_ols(sm_p_e, ff4, ["Mkt-RF","SMB","HML","MOM"], with_intercept=True)

print("=== CAPM CS regression (with intercept) ===")
print(cs_capm_int.summary().tables[1])
print("\n=== Carhart CS regression (with intercept) ===")
print(cs_ff4_int.summary().tables[1])
""")

md(r"""
**With intercept**, the CAPM is already rejected at this stage: the estimated market risk
premium $\hat\lambda_{MKT}$ is **negative and statistically significant**, the opposite sign
of the realised market premium. Carhart $\hat\lambda$'s are all close to their factor means
and the four-factor cross-sectional fit is greatly improved.
""")

md("### Two-step cross-sectional regression — no intercept (factor-model implied)")
code(r"""
def cross_section_no_intercept(P_excess, ts_results, factor_names):
    Rbar = P_excess.mean().values
    B    = ts_results["coefs"][factor_names].values
    eps  = ts_results["resid"].values; T, N = eps.shape
    BtB_inv = np.linalg.inv(B.T @ B)
    lam     = BtB_inv @ B.T @ Rbar
    Sigma_e = (eps.T @ eps) / T
    var_lam = BtB_inv @ B.T @ Sigma_e @ B @ BtB_inv / T
    se      = np.sqrt(np.diag(var_lam))
    fitted  = B @ lam; pe = Rbar - fitted
    return dict(lam=lam, se=se, t=lam/se, fitted=fitted, pricing_error=pe,
                Rbar=Rbar, B=B, Sigma_e=Sigma_e, factor_names=factor_names)

cs_capm = cross_section_no_intercept(sm_p_e, capm, ["Mkt-RF"])
cs_ff4  = cross_section_no_intercept(sm_p_e, ff4,  ["Mkt-RF","SMB","HML","MOM"])

def lam_df(cs, name):
    d = pd.DataFrame({"lambda (%/mo)": cs["lam"]*100, "se (%/mo)": cs["se"]*100, "t": cs["t"]},
                     index=cs["factor_names"]).round(3)
    d.index.name = name
    return d

display(lam_df(cs_capm, "CAPM"))
display(lam_df(cs_ff4,  "Carhart"))
""")

md("### 5(b) Estimated lambdas vs. model-implied premia (factor means)")
code(r"""
implied_capm = factors[["Mkt-RF"]].mean()*100
implied_ff4  = factors[["Mkt-RF","SMB","HML","MOM"]].mean()*100

cmp = pd.DataFrame({
    "lambda_CAPM_int (%/mo)":  np.r_[cs_capm_int.params.values*100, [np.nan]*3],
    "lambda_CAPM_no_int (%/mo)": np.r_[[np.nan, cs_capm["lam"][0]*100], [np.nan]*3],
    "lambda_4F_int (%/mo)":    np.r_[cs_ff4_int.params.values*100],
    "lambda_4F_no_int (%/mo)": np.r_[[np.nan, *cs_ff4["lam"]*100]],
    "factor mean (%/mo)":      np.r_[[np.nan, implied_capm.values[0]], implied_ff4.values[1:]],
}, index=["intercept","Mkt-RF","SMB","HML","MOM"]).round(3)
cmp
""")

md(r"""
For tradable factors, the model implies $\lambda_k = E[f_k]$. The CAPM with intercept gives
a *negative* $\hat\lambda_{MKT}$ — the cross-section "wants" a negative price of market risk
to fit the data, the textbook CAPM rejection (Black critique). The four-factor lambdas are
close to the factor means, especially for MKT and MOM.
""")

md(r"""
### 5(c) Time-series alphas (Q3, FF3) vs. cross-sectional pricing errors

Compare the FF3 *time-series* alphas to (i) the FF3 *cross-sectional* pricing errors and
(ii) the Carhart *cross-sectional* pricing errors.
""")

code(r"""
cs_ff3 = cross_section_no_intercept(sm_p_e, ff3, ["Mkt-RF","SMB","HML"])
ts_alpha_ff3 = ff3["coefs"]["const"].values * 100
cs_pe_ff3    = cs_ff3["pricing_error"] * 100
cs_pe_ff4    = cs_ff4["pricing_error"] * 100

cmp_pe = pd.DataFrame({
    "TS alpha FF3 (%/mo)": ts_alpha_ff3,
    "CS pe FF3 (%/mo)":    cs_pe_ff3,
    "CS pe FF4 (%/mo)":    cs_pe_ff4,
}, index=sm_p_e.columns).round(3)
cmp_pe.head(10)
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, y, name in zip(axes, [cs_pe_ff3, cs_pe_ff4], ["FF3", "FF4 (Carhart)"]):
    ax.scatter(ts_alpha_ff3, y, s=24)
    lo = min(ts_alpha_ff3.min(), y.min()) - 0.1
    hi = max(ts_alpha_ff3.max(), y.max()) + 0.1
    ax.plot([lo,hi],[lo,hi], "k--", lw=1)
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("FF3 time-series alpha (%/mo)")
    ax.set_title(f"vs. CS pricing error ({name})")
axes[0].set_ylabel("Cross-sectional pricing error (%/mo)")
savefig("q5c_alpha_vs_pe"); plt.show()

print(f"Mean |TS alpha FF3| (%/mo): {np.abs(ts_alpha_ff3).mean():.3f}")
print(f"Mean |CS pe FF3|   (%/mo): {np.abs(cs_pe_ff3).mean():.3f}")
print(f"Mean |CS pe FF4|   (%/mo): {np.abs(cs_pe_ff4).mean():.3f}")
""")

md(r"""
The FF3 time-series alphas remain large because FF3 cannot price the momentum spread. The
**Carhart** cross-sectional pricing errors are about an order of magnitude smaller — the MOM
factor absorbs most of the cross-sectional dispersion in mean returns.
""")

md("### 5(d) 60-month rolling betas of the four corner portfolios (Carhart)")
code(r"""
corners = ["SMALL LoPRIOR", "SMALL HiPRIOR", "BIG LoPRIOR", "BIG HiPRIOR"]
F4 = factors[["Mkt-RF","SMB","HML","MOM"]]; F4c = sm.add_constant(F4)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, c in zip(axes.flat, corners):
    rr = RollingOLS(sm_p_e[c], F4c, window=60, min_nobs=60).fit()
    rb = rr.params.dropna()
    rb[["Mkt-RF","SMB","HML","MOM"]].plot(ax=ax)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title(f"60-month rolling Carhart betas: {c}"); ax.legend(loc="upper left", fontsize=8)
savefig("q5d_rolling_corners"); plt.show()
""")

md(r"""
**Time-varying?** Yes, decisively. SMB loadings in particular swing by more than 1 across
the rolling windows; MOM loadings of the corner portfolios drift substantially (small-loser
$\beta_{MOM}$ moves between $-0.7$ and below $-1.6$). Time variation in betas is one reason
the unconditional Carhart model still leaves residual GRS rejections.
""")

# ---------------------------------------------------------------------------
md("## Question 6 — Carhart 4-factor on the combined 50 test assets (size/MOM + size/OP)")
code(r"""
combined = pd.concat([sm_p_e.add_prefix("MOM_"), sop_e.add_prefix("OP_")], axis=1).dropna()
print("Shape:", combined.shape)

ff4_50 = run_factor_model(combined, factors[["Mkt-RF","SMB","HML","MOM"]])
alpha50 = ff4_50["coefs"]["const"]*100; se50_nw = ff4_50["bse_nw"]["const"]*100; t50 = alpha50/se50_nw

q6 = pd.DataFrame({
    "alpha (%/mo)": alpha50.round(3), "se NW": se50_nw.round(3), "t (NW)": t50.round(2),
    "beta_MKT": ff4_50["coefs"]["Mkt-RF"].round(3),
    "beta_SMB": ff4_50["coefs"]["SMB"].round(3),
    "beta_HML": ff4_50["coefs"]["HML"].round(3),
    "beta_MOM": ff4_50["coefs"]["MOM"].round(3),
    "R^2":       ff4_50["r2"].round(3),
})
q6.head(8)
""")

code(r"""
grs4_50 = grs_test(combined, factors[["Mkt-RF","SMB","HML","MOM"]], ff4_50)
print(f"GRS (50 assets, Carhart) = {grs4_50['GRS']:.3f}  ~ F({grs4_50['df1']},{grs4_50['df2']})")
print(f"p-value = {grs4_50['p_value']:.3e}")
print(f"# alphas with NW |t|>1.96: {int((t50.abs()>1.96).sum())} / 50")

fitted50_an = (ff4_50["coefs"][["Mkt-RF","SMB","HML","MOM"]] *
              factors[["Mkt-RF","SMB","HML","MOM"]].mean()).sum(axis=1) * 12 * 100
realised50_an = combined.mean() * 12 * 100

fig, ax = plt.subplots(figsize=(7, 7))
lo = min(realised50_an.min(), fitted50_an.min()) - 1
hi = max(realised50_an.max(), fitted50_an.max()) + 1
ax.plot([lo,hi],[lo,hi], "k--", lw=1)
mask = realised50_an.index.str.startswith("MOM_")
ax.scatter(fitted50_an[mask],  realised50_an[mask],  s=24, label="size/MOM", alpha=0.75)
ax.scatter(fitted50_an[~mask], realised50_an[~mask], s=24, label="size/OP",  alpha=0.75)
ax.set_xlabel("Model-implied (% ann.)"); ax.set_ylabel("Realised (% ann.)")
ax.set_title("Carhart 4F — 50 test assets"); ax.legend()
savefig("q6_50_assets"); plt.show()
""")

md(r"""
Adding the 25 size/OP portfolios shifts the burden to the profitability dimension. Carhart
has no factor capturing the OP premium, so pricing errors accumulate along the small-low-OP
corner; GRS rejects strongly. Pricing the combined cross-section requires RMW (i.e. FF5 or
FF5+MOM).
""")

# ---------------------------------------------------------------------------
md("## Question 7 (optional) — Fama–MacBeth")
code(r"""
def fama_macbeth(P_excess, ts_results, factor_names):
    B = ts_results["coefs"][factor_names].values
    R = P_excess[ts_results["coefs"].index].values; T, N = R.shape
    K = B.shape[1]
    BtB_inv = np.linalg.inv(B.T @ B)
    lam_t = np.zeros((T, K))
    for t in range(T): lam_t[t] = BtB_inv @ B.T @ R[t]
    lam = lam_t.mean(axis=0); se = lam_t.std(axis=0, ddof=1) / np.sqrt(T)
    return pd.DataFrame({"lambda_FM (%/mo)": lam*100, "se_FM (%/mo)": se*100, "t_FM": lam/se},
                       index=factor_names).round(3)

print("Fama-MacBeth — CAPM");    display(fama_macbeth(sm_p_e, capm, ["Mkt-RF"]))
print("Fama-MacBeth — Carhart"); display(fama_macbeth(sm_p_e, ff4, ["Mkt-RF","SMB","HML","MOM"]))
""")

md(r"""
The FM point estimates equal the no-intercept OLS 2-step (algebraic identity) but FM standard
errors reflect time-series variation in the realised slopes — for Carhart, MKT and MOM remain
significant, SMB and HML are closer to zero.
""")

# ---------------------------------------------------------------------------
md(r"""
## Question 8 (optional) — Shanken correction
$$
\widehat{\mathrm{var}}_{\rm Shanken}(\hat\lambda) =
(\hat B'\hat B)^{-1}\hat B'\hat\Sigma_e\hat B(\hat B'\hat B)^{-1}
\bigl(1+\hat\lambda'\hat\Sigma_F^{-1}\hat\lambda\bigr)/T + \hat\Sigma_F/T.
$$
""")
code(r"""
def shanken_se(cs, F):
    SigmaF = np.cov(F.values, rowvar=False, ddof=1).reshape(len(cs["factor_names"]), -1)
    SigmaFinv = np.linalg.inv(SigmaF); T = F.shape[0]
    BtB_inv = np.linalg.inv(cs["B"].T @ cs["B"])
    base = BtB_inv @ cs["B"].T @ cs["Sigma_e"] @ cs["B"] @ BtB_inv / T
    inflate = 1.0 + cs["lam"] @ SigmaFinv @ cs["lam"]
    var_sh = base * inflate + SigmaF / T
    se_sh = np.sqrt(np.diag(var_sh))
    return pd.DataFrame({"lambda (%/mo)": cs["lam"]*100,
                         "se naive (%/mo)": cs["se"]*100,
                         "se Shanken (%/mo)": se_sh*100,
                         "t (Shanken)": cs["lam"]/se_sh},
                        index=cs["factor_names"]).round(3)

print("CAPM — Shanken-corrected"); display(shanken_se(cs_capm, factors[["Mkt-RF"]]))
print("Carhart — Shanken-corrected"); display(shanken_se(cs_ff4, factors[["Mkt-RF","SMB","HML","MOM"]]))
""")

md(r"""
With monthly factors the inflation factor $1+\hat\lambda'\hat\Sigma_F^{-1}\hat\lambda$ is
small here, so naive and Shanken-corrected SEs are close — qualitative inference is
unchanged.
""")

# ---------------------------------------------------------------------------
md(r"""
## Question 9 — Three individual stocks: CAPM and Carhart

We pick:
- **AAPL** (Apple) — mega-cap.
- **HSY**  (Hershey) — mid-cap consumer staples.
- **CRMT** (America's Car-Mart) — small-cap auto retailer.
""")

code(r"""
import yfinance as yf
tickers = ["AAPL","HSY","CRMT"]

# Monthly data
prices_m = yf.download(tickers, start="1990-01-01", interval="1mo",
                       progress=False, auto_adjust=False)["Adj Close"]
rets_m   = prices_m.pct_change().dropna(how="all")
rets_m.index = pd.to_datetime(rets_m.index).to_period("M").to_timestamp("M")
excess_m = rets_m.sub(factors["RF"], axis=0).dropna()
print("Monthly sample:", excess_m.index.min().date(), "to", excess_m.index.max().date())
""")

code(r"""
def stock_row(t, y, F, freq_unit):
    Xc = sm.add_constant(F)
    res = sm.OLS(y, Xc, missing="drop").fit()
    res_nw = sm.OLS(y, Xc, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    row = {"ticker": t, "N": int(res.nobs)}
    for k in F.columns:
        row[f"beta_{k}"] = res.params[k]
        row[f"se_{k}_NW"] = res_nw.bse[k]
    if freq_unit == "%/mo":
        row["alpha (%/mo)"] = res.params["const"] * 100
        row["se_alpha_NW"]  = res_nw.bse["const"] * 100
    else:
        row["alpha (bp/d)"] = res.params["const"] * 1e4
        row["se_alpha_NW"]  = res_nw.bse["const"] * 1e4
    row["R^2"] = res.rsquared
    return row

# CAPM monthly full / last 10y
cutoff = factors.index[-1] - pd.DateOffset(years=10)
print("=== CAPM monthly, full sample ===")
rows = [stock_row(t, excess_m[t].dropna(),
                  factors.loc[excess_m[t].dropna().index, ["Mkt-RF"]], "%/mo") for t in tickers]
display(pd.DataFrame(rows).round(3))

print("\n=== CAPM monthly, last 10 years ===")
rows = [stock_row(t, excess_m[t].loc[cutoff:].dropna(),
                  factors.loc[excess_m[t].loc[cutoff:].dropna().index, ["Mkt-RF"]], "%/mo") for t in tickers]
display(pd.DataFrame(rows).round(3))
""")

code(r"""
# Daily CAPM, last 5 years
prices_d = yf.download(tickers, start=(pd.Timestamp.today()-pd.DateOffset(years=5)).strftime("%Y-%m-%d"),
                       interval="1d", progress=False, auto_adjust=False)["Adj Close"]
rets_d = prices_d.pct_change().dropna(how="all")
ff_d   = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start=rets_d.index[0])[0] / 100.0
ff_d.index = pd.to_datetime(ff_d.index)
common = rets_d.index.intersection(ff_d.index); rets_d=rets_d.loc[common]; ff_d=ff_d.loc[common]
excess_d = rets_d.sub(ff_d["RF"], axis=0)

print("=== CAPM daily, last 5y ===")
rows = [stock_row(t, excess_d[t].dropna(),
                  ff_d.loc[excess_d[t].dropna().index, ["Mkt-RF"]], "bp") for t in tickers]
display(pd.DataFrame(rows).round(3))
""")

code(r"""
print("=== Carhart 4F monthly, full sample ===")
rows = [stock_row(t, excess_m[t].dropna(),
                  factors.loc[excess_m[t].dropna().index, ["Mkt-RF","SMB","HML","MOM"]], "%/mo")
        for t in tickers]
display(pd.DataFrame(rows).round(3))

print("\n=== Carhart 4F monthly, last 10 years ===")
rows = [stock_row(t, excess_m[t].loc[cutoff:].dropna(),
                  factors.loc[excess_m[t].loc[cutoff:].dropna().index, ["Mkt-RF","SMB","HML","MOM"]], "%/mo")
        for t in tickers]
display(pd.DataFrame(rows).round(3))
""")

md(r"""
**Discussion (Q9).**

- *AAPL.* CAPM beta is tightly above 1; alpha is large in monthly samples but shrinks in
  Carhart once MOM is loaded — much of Apple's "alpha" looks like exposure to the momentum
  factor in the 2000s–2010s.
- *HSY.* Defensive consumer staple. Low CAPM beta with low $R^2$; Carhart adds little.
- *CRMT.* Noisy small-cap. CAPM beta around 1 with wide SE; large positive SMB loading and
  meaningful HML tilt under Carhart. Estimates are highly sensitive to the sample window
  and to whether daily vs monthly returns are used (daily $\hat\beta$ tends to be biased
  downward by non-synchronous trading for very small caps).
""")

# ---------------------------------------------------------------------------
md(r"""
## Summary

| Test set | Model | GRS p-value | Mean |α| (%/mo) |
|---|---|---|---|
| 25 size/MOM | CAPM | $\approx 0$ | 0.28 |
| 25 size/MOM | FF3  | $\approx 0$ | 0.30 |
| 25 size/MOM | Carhart 4F | small | 0.13 |
| 50 (size/MOM + size/OP) | Carhart 4F | $\approx 0$ | 0.18 |

The momentum factor is essential for pricing the size/momentum cross-section; RMW would be
required to price size/OP. Time variation in betas (Q5d), and to a smaller extent
estimation noise in betas (Q8 Shanken), further weaken the unconditional 4-factor model.
""")

# ---------------------------------------------------------------------------
nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"name":"python3","display_name":"Python 3"},
    "language_info": {"name":"python"},
})

with OUT.open("w") as f: nbformat.write(nb, f)
print(f"Wrote skeleton: {OUT} ({len(cells)} cells)")

if "--no-run" not in sys.argv:
    print("Executing ...")
    ep = ExecutePreprocessor(timeout=900, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": str(HERE)}})
        with OUT.open("w") as f: nbformat.write(nb, f)
        print("Notebook executed successfully.")
    except Exception as e:
        with OUT.open("w") as f: nbformat.write(nb, f)
        print("Execution failed:", e)
        raise

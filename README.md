# MFE230E Problem Set 5 — Asset Pricing Tests

Empirical asset-pricing tests of the CAPM, Fama–French 3-factor, and Carhart 4-factor models
on Ken French's 25 size/momentum portfolios (and 25 size/operating-profitability portfolios as
a robustness check). Berkeley MFE 230E, Spring 2026.

Sample: monthly data, **July 1963** through the most recent month available on Ken French's
data library (data is fetched live).

## Contents

- [`MFE230E_PS5.ipynb`](MFE230E_PS5.ipynb) — unified, fully executed notebook with code,
  tables, and figures for all questions Q1–Q9. Q2(a) compares the size/BM and size/OP
  cross-sections; Q5(a) reports both intercept-included and no-intercept two-step CS
  regressions.
- [`MFE230E_PS5.pdf`](MFE230E_PS5.pdf) — 10-page LaTeX writeup (compiled).
- [`MFE230E_PS5.tex`](MFE230E_PS5.tex) — LaTeX source for the writeup.
- [`figs/`](figs/) — figures referenced by the LaTeX writeup.
- [`build_notebook.py`](build_notebook.py) — the builder script that constructs and executes
  the notebook from scratch via `nbformat` + `nbconvert`. Saves headline figures to `figs/`.

## Coverage

| Q | Topic |
|---|---|
| 1 | Fama–French factor summary statistics and cumulative returns |
| 2 | CAPM time-series regressions on 25 size/momentum portfolios; GRS; CML/MV-frontier; SML |
| 3 | Fama–French 3-factor model |
| 4 | Carhart 4-factor model |
| 5 | Cross-sectional risk premia (2-step), TS vs. CS pricing errors, rolling betas |
| 6 | Carhart 4-factor on combined 50 test assets (size/MOM + size/OP) |
| 7 | Fama–MacBeth (optional) |
| 8 | Shanken correction (optional) |
| 9 | Three single stocks (large-, mid-, small-cap) |

## Reproducing the analysis

Requirements: Python 3.10+ with `numpy`, `pandas`, `scipy`, `matplotlib`, `statsmodels`,
`pandas_datareader`, `yfinance`, `nbformat`, `nbconvert`.

```bash
python build_notebook.py
```

This regenerates `MFE230E_PS5.ipynb` end-to-end (downloads from Ken French and Yahoo Finance,
runs every cell, embeds outputs and figures).

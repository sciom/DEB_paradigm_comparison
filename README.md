# DEB Paradigm Comparison: Reproducible Benchmark

Reproducible benchmark code and results for:

> **Mechanistic, Bayesian, and Machine Learning Approaches to Metabolic Modelling: A Critical Review of Dynamic Energy Budget Theory and Its Alternatives**
>
> Branimir K. Hackenberger & Tamara Djerdj
>
> *Ecological Modelling* (submitted)

## Overview

This repository contains the complete R code for the head-to-head benchmark comparing four modelling approaches on simulated *Daphnia magna* body length growth data:

1. **Classical DEB** (least-squares optimisation)
2. **Bayesian DEB** (MCMC via BayesianTools, DEzs sampler)
3. **Random Forest** (ensemble ML)
4. **XGBoost** (gradient boosting ML)

Three benchmark scenarios are included:

| Scenario | Training | Test | Purpose |
|----------|----------|------|---------|
| 1. Standard | 20 °C (n=75) | 25 °C (n=75) | Interpolation vs. extrapolation |
| 2. Multi-temperature | 15+20+25 °C (n=225) | 28 °C (n=75) | Effect of richer training data on ML |
| 3. Model misspecification | 20 °C, variable κ (n=75) | — (interpolation only) | ML vs. misspecified DEB |

## Requirements

R >= 4.1 with the following packages:

```r
install.packages(c("deSolve", "BayesianTools", "randomForest",
                    "xgboost", "ggplot2", "patchwork", "dplyr", "tidyr"))
```

## Usage

```r
source("benchmark.R")
```

Running time: approximately 3-5 minutes on a standard desktop computer. All results are fully reproducible via `set.seed(42)`.

## Output files

| File | Description |
|------|-------------|
| `benchmark_results.csv` | Scenario 1 results (single-temperature) |
| `benchmark_results_multitemp.csv` | Scenario 2 results (multi-temperature) |
| `benchmark_results_misspec.csv` | Scenario 3 results (model misspecification) |
| `benchmark_figure.pdf` | Three-panel comparison figure (Scenario 1) |
| `benchmark_figure.png` | Same figure in PNG format |

## Key results

**Scenario 1** (correctly specified DEB, single-temperature training):
- DEB models achieve RMSE = 0.129 mm; ML models achieve 0.120 mm (XGBoost) to 0.635 mm (RF)
- Under extrapolation to 25 °C, DEB maintains RMSE = 0.113 mm while ML degrades to 0.437-0.771 mm

**Scenario 2** (multi-temperature training):
- ML interpolation improves to RMSE ~ 0.117 mm (comparable to DEB)
- ML extrapolation improves to ~ 0.23 mm (still 2x worse than DEB)

**Scenario 3** (misspecified DEB — time-varying κ):
- XGBoost outperforms DEB: RMSE = 0.120 vs. 0.204 mm
- Demonstrates that ML can beat mechanistic models when structural assumptions are violated

## Citation

If you use this code, please cite:

```
Hackenberger, B.K., Djerdj, T. (2026). Mechanistic, Bayesian, and Machine Learning
Approaches to Metabolic Modelling: A Critical Review of Dynamic Energy Budget Theory
and Its Alternatives. Ecological Modelling (submitted).
```

## Licence

MIT

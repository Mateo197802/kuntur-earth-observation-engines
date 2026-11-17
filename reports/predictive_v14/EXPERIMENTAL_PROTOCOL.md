# KUNTUR v1.4 experimental protocol (frozen before fitting)

## Scope

Development-only nested chronological evaluation. The source forecast is read with a target-month predicate through 2021-12. The monthly Gold input is limited to 2002-01 through 2021-12. The 2022–2024 test and sealed 2025 test are not passed to candidate scoring. Existing 2019–2021 validation months are acknowledged as previously inspected development data, not a fresh holdout.

## Outer folds

| Fold | Training target months | Scored target months |
|---|---|---|
| outer_2012_2014 | 2003-03–2011-12 | 2012-01–2014-12 |
| outer_2015_2017 | 2003-03–2014-12 | 2015-01–2017-12 |
| outer_2018_2021 | 2003-03–2017-12 | 2018-01–2021-12 |

Each outer training period has three expanding one-year inner validation windows ending at its cutoff. Inner folds select feature group, model configuration and residual shrinkage alpha. Outer months are scored once by the selection procedure. The 120 outer predictions are a development estimate, not an independent final test.

## Candidates and fixed budget

The feature groups and ten configurations (two each for Ridge, Elastic Net, RBF-SVR, shallow HistGradientBoosting and shallow Random Forest) are fixed in `scripts/run_predictive_v14.py`. Each residual prediction is evaluated at alpha {0, 0.25, 0.50, 0.75, 1.0}; alpha=0 is the climatology fallback. Selection minimizes pooled inner MAE, then RMSE, then feature count, model order and alpha. No neural network or unrestricted search is used.

## Primary metrics and acceptance rule

Primary: paired OOF MAE skill `1 - MAE(candidate)/MAE(training-only monthly climatology)`. Report RMSE, bias, median and 90th percentile absolute error, monthly win rate, fold scores and a paired moving-block bootstrap interval for candidate-minus-climatology absolute error (12-month blocks, 2,000 resamples). This interval is descriptive; temporal dependence, previous inspections and nested candidate search limit formal inference.

Predeclared preference for a trained pipeline requires: positive aggregate skill; the 95% paired moving-block interval for MAE difference entirely below zero; no outer-fold MAE degradation greater than 0.10 °C relative to climatology; absolute aggregate mean bias no greater than 0.15 °C; and valid issue-time features. Otherwise prefer climatology. If the interval overlaps zero, any positive mean skill remains exploratory.

## Augmentation and robustness

Compare clean training, one mild 10% Gaussian copy, one mild 10% feature-dropout copy, and a Gaussian intensity selected from 5%, 10%, 20% within the inner folds. Each augmented target's total sample weight remains one and true target labels are unchanged. Separately assess held-out feature corruption at 25% and 50% standardized RMS/missing fraction for Gaussian, missingness and combined corruption. Synthetic rows and perturbation replicas are not independent observations.

## Uncertainty

Compare earlier-residual expanding pooled, rolling-36 pooled and expanding same-calendar-month intervals at 80%, 90%, 95%. At each forecast, only earlier time-ordered residuals are eligible; previous outer OOF errors calibrate later folds. The first outer fold uses its three earlier inner-validation years as warm-up, which are selection-coupled. Coverage is descriptive and no exchangeability or conformal guarantee is claimed.

## Stop rule

No 2022–2024 or 2025 outcomes may change features, model, alpha, augmentation, threshold, or interval method. No independent unseen-period claim is available from this experiment.

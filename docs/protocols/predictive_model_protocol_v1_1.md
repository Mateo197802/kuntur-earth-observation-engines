# KUNTUR v1.1 predictive development protocol

Status: predeclared development protocol; not a final-test protocol.

## Scope and integrity

- Frozen Gold and original split artifacts are inputs only; this protocol never writes to them.
- Forecast targets stop at 2018-12. The old 2019-2021 validation and 2022-2024 test are not scored here. The source forecast Parquet is a single row group spanning 2003-2024, so a reader may decode the row group before applying the `target_month <= 2018-12` predicate. No post-2018 row is exposed to the model or included in metrics. This is a logical-use guarantee, not a claim of physical row-group pruning.
- The old 2022-2024 result remains archived evidence and is not a genuinely unseen v1.1 evaluation.
- The 2015-2018 years were previously used for v1 internal robustness. All v1.1 rolling results are therefore development evidence, not independent confirmation.

## Forecast alignment

Each target month is forecast from the preceding issue month using the frozen 12-month MERRA-2 history ending exactly three calendar months before the target (T-14 through T-3). The benchmark fails if the history end is not target minus three months or if the issue month is not target minus one month.

## Rolling folds

| Fold | Training targets | Validation targets |
|---|---|---|
| origin_2009 | through 2009-12 | 2010-01 to 2012-12 |
| origin_2012 | through 2012-12 | 2013-01 to 2015-12 |
| origin_2015 | through 2015-12 | 2016-01 to 2018-12 |

The 36 validation months in each fold are strictly later than all training targets. Imputation means, feature scaling, monthly climatology, and target scaling are fit on the training portion only. The temporal CNN reserves the final 12 training targets for early stopping and excludes them from gradient updates. No augmentation is used in this baseline corrected benchmark.

## Candidates and decision policy

All alternatives are eligible: persistence, seasonal naive, training-only monthly climatology, Ridge, Random Forest, Histogram Gradient Boosting, RBF-SVR, MLP, and a compact temporal CNN. Architecture hyperparameters are fixed in the versioned benchmark script; these folds are not used to tune them.

The clean ranking is the arithmetic mean of the three fold MAEs; ties are resolved by mean fold RMSE, then maximum fold MAE, then candidate ID. A separately reported robustness-aware development ranking uses an equal-weight composite of mean clean fold MAE and the mean across all predeclared 50% corruption scenarios (six individual families plus five combined families, each scored in each fold). The corruption scenarios do not alter target labels. This composite can support a development decision only; it cannot create an independent test estimate.

## Reproducibility

Random seeds are fixed and stochastic trained architectures use three declared seeds. Missing values are imputed with training-fold channel means and represented by explicit lag/channel indicators. Scalers and target transforms are training-fold-only. All candidate outputs are recorded by fold and seed. Comparisons between candidate losses are paired by the exact same target months; confidence intervals use a 12-month circular block bootstrap and are descriptive because these folds also participate in model selection.

# Predictive benchmark protocol v1.0

**Status:** Frozen before model fitting and external-result inspection.  
**Contract/data:** KUNTUR contract 1.0.0; Gold and forecast Parquet v1 remain immutable.

## Target, inputs, and split

Predict one-month-ahead Ecuador pilot-box mean 2 m temperature (`merra2_t2m_c`, °C) for target month `T`. The issue time is the start of `T-1`; the 12 history months run `T-14` through `T-3`, with `T-3` the latest eligible predictor. Inputs are the five MERRA-2 variables already in the v1 forecast table, arranged as 12 chronological months × 5 channels. Target/date/sample/provenance fields are excluded from predictors.

The v1 split remains unchanged: train 190 (2003-03–2018-12), validation 36 (2019-01–2021-12), test 36 (2022-01–2024-12). The first 152 train months form model-fit data; the final 38 form a chronological internal robustness diagnostic. The 38 are development data, not external validation. Validation selects models. Test is not read for tuning or selection.

## Augmentation

Estimate channel means and sample standard deviations only on the 152 model-fit samples, pooling their 12 history observations per variable. Keep authentic labels unchanged. Preserve six single-noise families and five defined combinations at 5%, 10%, 15%, and 20%, plus clean data. Additive intensity is realized RMS divided by model-fit SD. Dropout intensity is the missing fraction, with a per-feature indicator and model-fit-mean imputation. Impulses affect a fixed 5% of eligible cells with magnitude scaled to the requested RMS. Drift is centered AR(1), rho 0.8. Heteroscedastic noise has value-dependent variance then is normalized to the requested RMS.

Combined scenarios divide the total budget across components by square root of component count, then normalize additive RMS; for missingness combinations the effective budget is `sqrt(additive_RMS_fraction^2 + missing_fraction^2)`. No value is clipped. Any synthetic row outside frozen broad physical screens is rejected and recorded. Each original contributes unit clean weight and unit total accepted synthetic weight, so the 44 variants do not outweigh the authentic row. Synthetic examples are not independent observations.

## Model candidates and selection

Fit six families: Ridge (alpha 0.1/1/10), Random Forest (100/200 trees; depth 2/4), Histogram Gradient Boosting (learning rate 0.05/0.1; 100/150 iterations; max leaf nodes 7), RBF SVR (C 0.1/1/10), MLP (one hidden layer 16/32; alpha 0.001/0.01), and temporal 1D CNN (channels 8/16; fixed 100 epochs; Adam learning rate 0.001/0.003). All searches are bounded. Use identical 152 fit samples and sample-weighted clean/augmentation regimes. Standardization fits only the 152 clean training histories; binary missingness indicators stay unscaled.

Select by clean validation MAE, then validation RMSE, then the predeclared simpler model order Ridge, SVR, HGB, RF, MLP, CNN to resolve exact ties. Evaluate persistence (latest T-3 value), seasonal naive (T-12 value, the v1 `lag_10`), and monthly climatology estimated from model-fit target labels only. No baseline uses later inputs.

Use fixed seeds from the run manifest. MLP and CNN do not use random time-series cross-validation or a random internal holdout. The CNN uses validation-only early stopping with a 12-epoch patience, maximum 100 epochs, and best validation-loss checkpoint. The fixed architecture-specific search grid and all fit logs are retained.

## Robustness and test order

After fitting and selecting solely on clean validation, evaluate all six development checkpoints and three baselines on the 38 internal months: clean, each of six single families at 25%/50%, and each of five combinations at 25%/50%. Use one independent deterministic corruption realization per sample/scenario and seeds separate from training. A scenario’s repeated observations are not counted as additional independent months. Compute bootstrap intervals by resampling unique target months only.

Freeze selection and robustness results before any test read. Refit only the selected architecture on the original 190 train months using the same augmentation protocol, with scales re-estimated on those 190 training months. Calibrate a symmetric 90% residual interval from validation absolute errors and then run one final evaluation of the frozen model and baselines on the 36 test months. Test results cannot change the model, features, intervals, or protocol.

## Metrics

Report MAE, RMSE, mean bias, MAE/RMSE change versus clean within the same samples, degradation ratio, prediction-difference stability, invalid predictions, 90% interval coverage, sample count, and month-level bootstrap 95% intervals. Report baseline skill as `1 - model_error / baseline_error`. Keep the 38-month robustness audit, 36-month final test, and validation results visibly separate. Do not claim domain independence from synthetic perturbation tests.

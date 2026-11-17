# KUNTUR v1.4 feature-contract audit

- Target is MERRA-2 Ecuador pilot-bbox monthly mean T2M in °C for month T.
- Forecast origin is T-1; assumed issue time is 00:00 UTC on the first of T-1.
- All inputs are from exact monthly records T-14 through T-3, ending before issue month.
- The two-month release lag is inherited from the frozen contract. Actual NASA publication timestamps were not established; this is a retrospective availability assumption.
- Development predictors: MERRA-2 T2M, specific humidity, u/v wind components and pressure; IMERG monthly precipitation total; four GLDAS soil-moisture layers.
- IMERG precipitation is transformed with `log1p` before training-period calendar-month anomaly calculation. Wind speed is deterministically `hypot(u2m, v2m)`; the native source components remain in Gold.
- Calendar-month predictor means, target monthly climatology, imputation means and StandardScaler parameters are fitted separately inside each inner/outer training fold.
- Missing values are imputed with fold-training feature means and explicit missing indicators. No future observations are used to impute.
- The sealed 2025 predictor file has five MERRA-2 channels only. Cross-source groups are development-only and cannot be applied unchanged to 2025.
- All output remains for the Ecuador pilot bounding box; no Coast/Andes/Amazon mask is claimed.

Timeline and fold assertions are exercised by `tests/test_predictive_v14.py`.

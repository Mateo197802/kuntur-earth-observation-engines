# Public status reconciliation — 2026-11-17

This public-safe copy reports the historical six-regime v1.3 development expansion below. Its original “no independent confirmation” status was accurate when that report was generated. Later, a separate v1.3.2 correction compared IID interval methods and the finite-sample Student-t interval passed a fresh IID Gaussian confirmation; that result is documented in [the v1.3.2 confirmation report](v1_3_independent_confirmation_iid_20261117.md). It does not confirm any dependent or irregular-calendar regime. The v1.3 expansion produced no adequately supported candidate in five non-IID regimes, no confirmation was run for those regimes, and the inference registry remains unchanged. The reserved future confirmation namespace remains unused and is only a proposal.

The aggregate evidence in this report is public-safe. Seed-level checkpoints and raw replicate rows remain local and are not included. The PR containing this report remains draft and has not been merged; no scientific paper is being published here.

---
# KUNTUR v1.3 Corrective Development Candidate Selection

**Status:** DEVELOPMENT_EXPANSION_COMPLETE — development evidence only. No statistical inference registry entry was created and no independent confirmation was launched.

Generated: 2026-11-17T18:46:22.642900+00:00

## Execution

- Planned: 180,000 outer fits across 180 paired cells and 60 method-regime combinations.
- Completed: 180,000 fits in 180/180 complete cells.
- Accumulated process CPU: 18423.6 s (5.118 h); active wall: 18507.9 s; peak RSS: 203,730,944 bytes.
- Sampling is paired by regime, slope and replicate. Exact binomial intervals are Clopper–Pearson 95% intervals; continuous metric intervals are t-based Monte Carlo intervals.

## Candidate decision

- **iid_gaussian: CANDIDATE_READY_FOR_FREEZE_BEFORE_CONFIRMATION** — SARIMAX_SEASONAL_AR12
- **ar1_0844: NO_ADEQUATELY_SUPPORTED_CANDIDATE**
- **arma11: NO_ADEQUATELY_SUPPORTED_CANDIDATE**
- **seasonal_ar1_12: NO_ADEQUATELY_SUPPORTED_CANDIDATE**
- **ar1_060_mcar20: NO_ADEQUATELY_SUPPORTED_CANDIDATE**
- **ar1_060_seasonal_missing: NO_ADEQUATELY_SUPPORTED_CANDIDATE**

Development candidates are not validated. Freeze their exact code, regime scope, gates and independent namespace before any confirmation run.

## Type-I calibration and coverage

| Regime | Method | Slope | Type-I / power | 95% binomial interval | Coverage | 95% coverage interval |
|---|---|---:|---:|---|---:|---|
| ar1_060_mcar20 | OLS_HAC_CALENDAR12 | 0.00 | 0.112 | [0.093, 0.133] | 0.888 | [0.867, 0.907] |
| ar1_060_mcar20 | OLS_HAC_CALENDAR12 | 0.25 | 0.999 | [0.994, 1.000] | 0.889 | [0.868, 0.908] |
| ar1_060_mcar20 | OLS_HAC_CALENDAR12 | 0.50 | 1.000 | [0.996, 1.000] | 0.894 | [0.873, 0.912] |
| ar1_060_mcar20 | SARIMAX_AR1 | 0.00 | 0.054 | [0.041, 0.070] | 0.946 | [0.930, 0.959] |
| ar1_060_mcar20 | SARIMAX_AR1 | 0.25 | 0.996 | [0.990, 0.999] | 0.944 | [0.927, 0.957] |
| ar1_060_mcar20 | SARIMAX_AR1 | 0.50 | 1.000 | [0.996, 1.000] | 0.956 | [0.941, 0.968] |
| ar1_060_mcar20 | SARIMAX_ARMA11 | 0.00 | 0.054 | [0.041, 0.070] | 0.946 | [0.930, 0.959] |
| ar1_060_mcar20 | SARIMAX_ARMA11 | 0.25 | 0.996 | [0.990, 0.999] | 0.940 | [0.923, 0.954] |
| ar1_060_mcar20 | SARIMAX_ARMA11 | 0.50 | 1.000 | [0.996, 1.000] | 0.956 | [0.941, 0.968] |
| ar1_060_mcar20 | SARIMAX_SEASONAL_AR12 | 0.00 | 0.273 | [0.246, 0.302] | 0.727 | [0.698, 0.754] |
| ar1_060_mcar20 | SARIMAX_SEASONAL_AR12 | 0.25 | 1.000 | [0.996, 1.000] | 0.691 | [0.661, 0.719] |
| ar1_060_mcar20 | SARIMAX_SEASONAL_AR12 | 0.50 | 1.000 | [0.996, 1.000] | 0.718 | [0.689, 0.746] |
| ar1_060_mcar20 | SEASONAL_MK_YEAR1 | 0.00 | 0.087 | [0.070, 0.106] | NA | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR1 | 0.25 | 0.999 | [0.994, 1.000] | NA | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR1 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR2 | 0.00 | 0.099 | [0.081, 0.119] | NA | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR2 | 0.25 | 0.999 | [0.994, 1.000] | NA | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR2 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR3 | 0.00 | 0.113 | [0.094, 0.134] | NA | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR3 | 0.25 | 0.999 | [0.994, 1.000] | NA | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR3 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR4 | 0.00 | 0.128 | [0.108, 0.150] | NA | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR4 | 0.25 | 0.999 | [0.994, 1.000] | NA | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR4 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| ar1_060_seasonal_missing | OLS_HAC_CALENDAR12 | 0.00 | 0.100 | [0.082, 0.120] | 0.900 | [0.880, 0.918] |
| ar1_060_seasonal_missing | OLS_HAC_CALENDAR12 | 0.25 | 1.000 | [0.996, 1.000] | 0.885 | [0.864, 0.904] |
| ar1_060_seasonal_missing | OLS_HAC_CALENDAR12 | 0.50 | 1.000 | [0.996, 1.000] | 0.886 | [0.865, 0.905] |
| ar1_060_seasonal_missing | SARIMAX_AR1 | 0.00 | 0.047 | [0.035, 0.062] | 0.953 | [0.938, 0.965] |
| ar1_060_seasonal_missing | SARIMAX_AR1 | 0.25 | 1.000 | [0.996, 1.000] | 0.938 | [0.921, 0.952] |
| ar1_060_seasonal_missing | SARIMAX_AR1 | 0.50 | 1.000 | [0.996, 1.000] | 0.940 | [0.923, 0.954] |
| ar1_060_seasonal_missing | SARIMAX_ARMA11 | 0.00 | 0.054 | [0.041, 0.070] | 0.946 | [0.930, 0.959] |
| ar1_060_seasonal_missing | SARIMAX_ARMA11 | 0.25 | 1.000 | [0.996, 1.000] | 0.937 | [0.920, 0.951] |
| ar1_060_seasonal_missing | SARIMAX_ARMA11 | 0.50 | 1.000 | [0.996, 1.000] | 0.946 | [0.930, 0.959] |
| ar1_060_seasonal_missing | SARIMAX_SEASONAL_AR12 | 0.00 | 0.299 | [0.271, 0.329] | 0.701 | [0.671, 0.729] |
| ar1_060_seasonal_missing | SARIMAX_SEASONAL_AR12 | 0.25 | 1.000 | [0.996, 1.000] | 0.725 | [0.697, 0.753] |
| ar1_060_seasonal_missing | SARIMAX_SEASONAL_AR12 | 0.50 | 1.000 | [0.996, 1.000] | 0.710 | [0.681, 0.738] |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR1 | 0.00 | 0.071 | [0.056, 0.089] | NA | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR1 | 0.25 | 1.000 | [0.996, 1.000] | NA | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR1 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR2 | 0.00 | 0.086 | [0.069, 0.105] | NA | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR2 | 0.25 | 1.000 | [0.996, 1.000] | NA | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR2 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR3 | 0.00 | 0.096 | [0.078, 0.116] | NA | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR3 | 0.25 | 1.000 | [0.996, 1.000] | NA | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR3 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR4 | 0.00 | 0.115 | [0.096, 0.136] | NA | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR4 | 0.25 | 1.000 | [0.996, 1.000] | NA | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR4 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| ar1_0844 | FEGLS_AR1 | 0.00 | 0.079 | [0.063, 0.097] | 0.921 | [0.903, 0.937] |
| ar1_0844 | FEGLS_AR1 | 0.25 | 0.865 | [0.842, 0.886] | 0.915 | [0.896, 0.932] |
| ar1_0844 | FEGLS_AR1 | 0.50 | 1.000 | [0.996, 1.000] | 0.924 | [0.906, 0.940] |
| ar1_0844 | OLS_HAC_LAG12 | 0.00 | 0.173 | [0.150, 0.198] | 0.827 | [0.802, 0.850] |
| ar1_0844 | OLS_HAC_LAG12 | 0.25 | 0.955 | [0.940, 0.967] | 0.808 | [0.782, 0.832] |
| ar1_0844 | OLS_HAC_LAG12 | 0.50 | 1.000 | [0.996, 1.000] | 0.818 | [0.793, 0.841] |
| ar1_0844 | OLS_HAC_LAG6 | 0.00 | 0.238 | [0.212, 0.266] | 0.762 | [0.734, 0.788] |
| ar1_0844 | OLS_HAC_LAG6 | 0.25 | 0.978 | [0.967, 0.986] | 0.753 | [0.725, 0.779] |
| ar1_0844 | OLS_HAC_LAG6 | 0.50 | 1.000 | [0.996, 1.000] | 0.761 | [0.733, 0.787] |
| ar1_0844 | SARIMAX_AR1 | 0.00 | 0.079 | [0.063, 0.097] | 0.921 | [0.903, 0.937] |
| ar1_0844 | SARIMAX_AR1 | 0.25 | 0.892 | [0.871, 0.911] | 0.932 | [0.914, 0.946] |
| ar1_0844 | SARIMAX_AR1 | 0.50 | 1.000 | [0.996, 1.000] | 0.939 | [0.923, 0.953] |
| ar1_0844 | SARIMAX_ARMA11 | 0.00 | 0.078 | [0.062, 0.096] | 0.922 | [0.904, 0.938] |
| ar1_0844 | SARIMAX_ARMA11 | 0.25 | 0.889 | [0.868, 0.908] | 0.922 | [0.904, 0.938] |
| ar1_0844 | SARIMAX_ARMA11 | 0.50 | 1.000 | [0.996, 1.000] | 0.933 | [0.915, 0.947] |
| ar1_0844 | SARIMAX_SEASONAL_AR12 | 0.00 | 0.557 | [0.525, 0.588] | 0.443 | [0.412, 0.475] |
| ar1_0844 | SARIMAX_SEASONAL_AR12 | 0.25 | 0.995 | [0.988, 0.998] | 0.463 | [0.432, 0.494] |
| ar1_0844 | SARIMAX_SEASONAL_AR12 | 0.50 | 1.000 | [0.996, 1.000] | 0.450 | [0.419, 0.482] |
| ar1_0844 | SEASONAL_BLOCK_YEAR2 | 0.00 | 0.087 | [0.070, 0.106] | 0.855 | [0.832, 0.876] |
| ar1_0844 | SEASONAL_BLOCK_YEAR2 | 0.25 | 0.896 | [0.875, 0.914] | 0.838 | [0.814, 0.860] |
| ar1_0844 | SEASONAL_BLOCK_YEAR2 | 0.50 | 1.000 | [0.996, 1.000] | 0.858 | [0.835, 0.879] |
| ar1_0844 | SEASONAL_MK_YEAR1 | 0.00 | 0.141 | [0.120, 0.164] | NA | NA |
| ar1_0844 | SEASONAL_MK_YEAR1 | 0.25 | 0.937 | [0.920, 0.951] | NA | NA |
| ar1_0844 | SEASONAL_MK_YEAR1 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| ar1_0844 | SEASONAL_MK_YEAR2 | 0.00 | 0.108 | [0.089, 0.129] | NA | NA |
| ar1_0844 | SEASONAL_MK_YEAR2 | 0.25 | 0.919 | [0.900, 0.935] | NA | NA |
| ar1_0844 | SEASONAL_MK_YEAR2 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| ar1_0844 | SEASONAL_MK_YEAR3 | 0.00 | 0.113 | [0.094, 0.134] | NA | NA |
| ar1_0844 | SEASONAL_MK_YEAR3 | 0.25 | 0.914 | [0.895, 0.931] | NA | NA |
| ar1_0844 | SEASONAL_MK_YEAR3 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| ar1_0844 | SEASONAL_MK_YEAR4 | 0.00 | 0.136 | [0.115, 0.159] | NA | NA |
| ar1_0844 | SEASONAL_MK_YEAR4 | 0.25 | 0.916 | [0.897, 0.932] | NA | NA |
| ar1_0844 | SEASONAL_MK_YEAR4 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| arma11 | FEGLS_AR1 | 0.00 | 0.027 | [0.018, 0.039] | 0.973 | [0.961, 0.982] |
| arma11 | FEGLS_AR1 | 0.25 | 1.000 | [0.996, 1.000] | 0.969 | [0.956, 0.979] |
| arma11 | FEGLS_AR1 | 0.50 | 1.000 | [0.996, 1.000] | 0.978 | [0.967, 0.986] |
| arma11 | OLS_HAC_LAG12 | 0.00 | 0.093 | [0.076, 0.113] | 0.907 | [0.887, 0.924] |
| arma11 | OLS_HAC_LAG12 | 0.25 | 1.000 | [0.996, 1.000] | 0.889 | [0.868, 0.908] |
| arma11 | OLS_HAC_LAG12 | 0.50 | 1.000 | [0.996, 1.000] | 0.911 | [0.892, 0.928] |
| arma11 | OLS_HAC_LAG6 | 0.00 | 0.094 | [0.077, 0.114] | 0.906 | [0.886, 0.923] |
| arma11 | OLS_HAC_LAG6 | 0.25 | 1.000 | [0.996, 1.000] | 0.888 | [0.867, 0.907] |
| arma11 | OLS_HAC_LAG6 | 0.50 | 1.000 | [0.996, 1.000] | 0.911 | [0.892, 0.928] |
| arma11 | SARIMAX_AR1 | 0.00 | 0.022 | [0.014, 0.033] | 0.978 | [0.967, 0.986] |
| arma11 | SARIMAX_AR1 | 0.25 | 1.000 | [0.996, 1.000] | 0.976 | [0.964, 0.984] |
| arma11 | SARIMAX_AR1 | 0.50 | 1.000 | [0.996, 1.000] | 0.983 | [0.973, 0.990] |
| arma11 | SARIMAX_ARMA11 | 0.00 | 0.049 | [0.037, 0.065] | 0.951 | [0.935, 0.963] |
| arma11 | SARIMAX_ARMA11 | 0.25 | 1.000 | [0.996, 1.000] | 0.931 | [0.913, 0.946] |
| arma11 | SARIMAX_ARMA11 | 0.50 | 1.000 | [0.996, 1.000] | 0.951 | [0.936, 0.963] |
| arma11 | SARIMAX_SEASONAL_AR12 | 0.00 | 0.310 | [0.282, 0.340] | 0.690 | [0.660, 0.718] |
| arma11 | SARIMAX_SEASONAL_AR12 | 0.25 | 1.000 | [0.996, 1.000] | 0.670 | [0.639, 0.699] |
| arma11 | SARIMAX_SEASONAL_AR12 | 0.50 | 1.000 | [0.996, 1.000] | 0.725 | [0.696, 0.752] |
| arma11 | SEASONAL_BLOCK_YEAR2 | 0.00 | 0.063 | [0.049, 0.080] | 0.890 | [0.869, 0.909] |
| arma11 | SEASONAL_BLOCK_YEAR2 | 0.25 | 1.000 | [0.996, 1.000] | 0.868 | [0.845, 0.888] |
| arma11 | SEASONAL_BLOCK_YEAR2 | 0.50 | 1.000 | [0.996, 1.000] | 0.895 | [0.874, 0.913] |
| arma11 | SEASONAL_MK_YEAR1 | 0.00 | 0.074 | [0.059, 0.092] | NA | NA |
| arma11 | SEASONAL_MK_YEAR1 | 0.25 | 1.000 | [0.996, 1.000] | NA | NA |
| arma11 | SEASONAL_MK_YEAR1 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| arma11 | SEASONAL_MK_YEAR2 | 0.00 | 0.088 | [0.071, 0.107] | NA | NA |
| arma11 | SEASONAL_MK_YEAR2 | 0.25 | 1.000 | [0.996, 1.000] | NA | NA |
| arma11 | SEASONAL_MK_YEAR2 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| arma11 | SEASONAL_MK_YEAR3 | 0.00 | 0.101 | [0.083, 0.121] | NA | NA |
| arma11 | SEASONAL_MK_YEAR3 | 0.25 | 1.000 | [0.996, 1.000] | NA | NA |
| arma11 | SEASONAL_MK_YEAR3 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| arma11 | SEASONAL_MK_YEAR4 | 0.00 | 0.110 | [0.091, 0.131] | NA | NA |
| arma11 | SEASONAL_MK_YEAR4 | 0.25 | 1.000 | [0.996, 1.000] | NA | NA |
| arma11 | SEASONAL_MK_YEAR4 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| iid_gaussian | FEGLS_AR1 | 0.00 | 0.058 | [0.044, 0.074] | 0.942 | [0.926, 0.956] |
| iid_gaussian | FEGLS_AR1 | 0.25 | 1.000 | [0.996, 1.000] | 0.947 | [0.931, 0.960] |
| iid_gaussian | FEGLS_AR1 | 0.50 | 1.000 | [0.996, 1.000] | 0.937 | [0.920, 0.951] |
| iid_gaussian | OLS_HAC_LAG12 | 0.00 | 0.083 | [0.067, 0.102] | 0.917 | [0.898, 0.933] |
| iid_gaussian | OLS_HAC_LAG12 | 0.25 | 1.000 | [0.996, 1.000] | 0.930 | [0.912, 0.945] |
| iid_gaussian | OLS_HAC_LAG12 | 0.50 | 1.000 | [0.996, 1.000] | 0.917 | [0.898, 0.933] |
| iid_gaussian | OLS_HAC_LAG6 | 0.00 | 0.071 | [0.056, 0.089] | 0.929 | [0.911, 0.944] |
| iid_gaussian | OLS_HAC_LAG6 | 0.25 | 1.000 | [0.996, 1.000] | 0.942 | [0.926, 0.956] |
| iid_gaussian | OLS_HAC_LAG6 | 0.50 | 1.000 | [0.996, 1.000] | 0.929 | [0.911, 0.944] |
| iid_gaussian | SARIMAX_AR1 | 0.00 | 0.051 | [0.038, 0.067] | 0.949 | [0.933, 0.962] |
| iid_gaussian | SARIMAX_AR1 | 0.25 | 1.000 | [0.996, 1.000] | 0.951 | [0.936, 0.963] |
| iid_gaussian | SARIMAX_AR1 | 0.50 | 1.000 | [0.996, 1.000] | 0.946 | [0.930, 0.959] |
| iid_gaussian | SARIMAX_ARMA11 | 0.00 | 0.095 | [0.077, 0.115] | 0.905 | [0.885, 0.923] |
| iid_gaussian | SARIMAX_ARMA11 | 0.25 | 1.000 | [0.996, 1.000] | 0.915 | [0.896, 0.932] |
| iid_gaussian | SARIMAX_ARMA11 | 0.50 | 1.000 | [0.996, 1.000] | 0.911 | [0.891, 0.928] |
| iid_gaussian | SARIMAX_SEASONAL_AR12 | 0.00 | 0.064 | [0.050, 0.081] | 0.936 | [0.919, 0.950] |
| iid_gaussian | SARIMAX_SEASONAL_AR12 | 0.25 | 1.000 | [0.996, 1.000] | 0.944 | [0.927, 0.957] |
| iid_gaussian | SARIMAX_SEASONAL_AR12 | 0.50 | 1.000 | [0.996, 1.000] | 0.938 | [0.921, 0.952] |
| iid_gaussian | SEASONAL_BLOCK_YEAR2 | 0.00 | 0.057 | [0.043, 0.073] | 0.891 | [0.870, 0.910] |
| iid_gaussian | SEASONAL_BLOCK_YEAR2 | 0.25 | 1.000 | [0.996, 1.000] | 0.900 | [0.880, 0.918] |
| iid_gaussian | SEASONAL_BLOCK_YEAR2 | 0.50 | 1.000 | [0.996, 1.000] | 0.887 | [0.866, 0.906] |
| iid_gaussian | SEASONAL_MK_YEAR1 | 0.00 | 0.064 | [0.050, 0.081] | NA | NA |
| iid_gaussian | SEASONAL_MK_YEAR1 | 0.25 | 1.000 | [0.996, 1.000] | NA | NA |
| iid_gaussian | SEASONAL_MK_YEAR1 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| iid_gaussian | SEASONAL_MK_YEAR2 | 0.00 | 0.079 | [0.063, 0.097] | NA | NA |
| iid_gaussian | SEASONAL_MK_YEAR2 | 0.25 | 1.000 | [0.996, 1.000] | NA | NA |
| iid_gaussian | SEASONAL_MK_YEAR2 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| iid_gaussian | SEASONAL_MK_YEAR3 | 0.00 | 0.102 | [0.084, 0.122] | NA | NA |
| iid_gaussian | SEASONAL_MK_YEAR3 | 0.25 | 1.000 | [0.996, 1.000] | NA | NA |
| iid_gaussian | SEASONAL_MK_YEAR3 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| iid_gaussian | SEASONAL_MK_YEAR4 | 0.00 | 0.112 | [0.093, 0.133] | NA | NA |
| iid_gaussian | SEASONAL_MK_YEAR4 | 0.25 | 1.000 | [0.996, 1.000] | NA | NA |
| iid_gaussian | SEASONAL_MK_YEAR4 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| seasonal_ar1_12 | FEGLS_AR1 | 0.00 | 0.302 | [0.274, 0.332] | 0.698 | [0.668, 0.726] |
| seasonal_ar1_12 | FEGLS_AR1 | 0.25 | 1.000 | [0.996, 1.000] | 0.673 | [0.643, 0.702] |
| seasonal_ar1_12 | FEGLS_AR1 | 0.50 | 1.000 | [0.996, 1.000] | 0.721 | [0.692, 0.749] |
| seasonal_ar1_12 | OLS_HAC_LAG12 | 0.00 | 0.352 | [0.322, 0.383] | 0.648 | [0.617, 0.678] |
| seasonal_ar1_12 | OLS_HAC_LAG12 | 0.25 | 1.000 | [0.996, 1.000] | 0.620 | [0.589, 0.650] |
| seasonal_ar1_12 | OLS_HAC_LAG12 | 0.50 | 1.000 | [0.996, 1.000] | 0.658 | [0.628, 0.687] |
| seasonal_ar1_12 | OLS_HAC_LAG6 | 0.00 | 0.326 | [0.297, 0.356] | 0.674 | [0.644, 0.703] |
| seasonal_ar1_12 | OLS_HAC_LAG6 | 0.25 | 1.000 | [0.996, 1.000] | 0.645 | [0.614, 0.675] |
| seasonal_ar1_12 | OLS_HAC_LAG6 | 0.50 | 1.000 | [0.996, 1.000] | 0.690 | [0.660, 0.719] |
| seasonal_ar1_12 | SARIMAX_AR1 | 0.00 | 0.286 | [0.258, 0.315] | 0.714 | [0.685, 0.742] |
| seasonal_ar1_12 | SARIMAX_AR1 | 0.25 | 1.000 | [0.996, 1.000] | 0.686 | [0.657, 0.715] |
| seasonal_ar1_12 | SARIMAX_AR1 | 0.50 | 1.000 | [0.996, 1.000] | 0.724 | [0.695, 0.752] |
| seasonal_ar1_12 | SARIMAX_ARMA11 | 0.00 | 0.294 | [0.266, 0.324] | 0.706 | [0.676, 0.734] |
| seasonal_ar1_12 | SARIMAX_ARMA11 | 0.25 | 0.999 | [0.994, 1.000] | 0.673 | [0.643, 0.702] |
| seasonal_ar1_12 | SARIMAX_ARMA11 | 0.50 | 1.000 | [0.996, 1.000] | 0.712 | [0.683, 0.740] |
| seasonal_ar1_12 | SARIMAX_SEASONAL_AR12 | 0.00 | 0.079 | [0.063, 0.098] | 0.921 | [0.902, 0.937] |
| seasonal_ar1_12 | SARIMAX_SEASONAL_AR12 | 0.25 | 1.000 | [0.996, 1.000] | 0.917 | [0.898, 0.933] |
| seasonal_ar1_12 | SARIMAX_SEASONAL_AR12 | 0.50 | 1.000 | [0.996, 1.000] | 0.922 | [0.904, 0.938] |
| seasonal_ar1_12 | SEASONAL_BLOCK_YEAR2 | 0.00 | 0.206 | [0.181, 0.232] | 0.689 | [0.659, 0.718] |
| seasonal_ar1_12 | SEASONAL_BLOCK_YEAR2 | 0.25 | 1.000 | [0.996, 1.000] | 0.667 | [0.637, 0.696] |
| seasonal_ar1_12 | SEASONAL_BLOCK_YEAR2 | 0.50 | 1.000 | [0.996, 1.000] | 0.702 | [0.673, 0.730] |
| seasonal_ar1_12 | SEASONAL_MK_YEAR1 | 0.00 | 0.312 | [0.283, 0.342] | NA | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR1 | 0.25 | 1.000 | [0.996, 1.000] | NA | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR1 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR2 | 0.00 | 0.257 | [0.230, 0.285] | NA | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR2 | 0.25 | 1.000 | [0.996, 1.000] | NA | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR2 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR3 | 0.00 | 0.240 | [0.214, 0.268] | NA | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR3 | 0.25 | 1.000 | [0.996, 1.000] | NA | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR3 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR4 | 0.00 | 0.239 | [0.213, 0.267] | NA | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR4 | 0.25 | 1.000 | [0.996, 1.000] | NA | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR4 | 0.50 | 1.000 | [0.996, 1.000] | NA | NA |

## Convergence, bias and interval efficiency

| Regime | Method | Slope | Failures / supported | Availability | Bias (95% MC interval) | Mean width/oracle |
|---|---|---:|---:|---:|---|---:|
| ar1_060_mcar20 | OLS_HAC_CALENDAR12 | 0.00 | 0/1000 | 1.000 | 0.0002 [-0.0026, 0.0030] | 0.861 |
| ar1_060_mcar20 | OLS_HAC_CALENDAR12 | 0.25 | 0/1000 | 1.000 | -0.0015 [-0.0045, 0.0014] | 0.858 |
| ar1_060_mcar20 | OLS_HAC_CALENDAR12 | 0.50 | 0/1000 | 1.000 | 0.0008 [-0.0020, 0.0037] | 0.863 |
| ar1_060_mcar20 | SARIMAX_AR1 | 0.00 | 4/1000 | 0.996 | 0.0002 [-0.0026, 0.0030] | 1.011 |
| ar1_060_mcar20 | SARIMAX_AR1 | 0.25 | 7/1000 | 0.993 | -0.0017 [-0.0046, 0.0013] | 1.009 |
| ar1_060_mcar20 | SARIMAX_AR1 | 0.50 | 3/1000 | 0.997 | 0.0007 [-0.0021, 0.0034] | 1.012 |
| ar1_060_mcar20 | SARIMAX_ARMA11 | 0.00 | 0/1000 | 1.000 | 0.0001 [-0.0027, 0.0029] | 1.015 |
| ar1_060_mcar20 | SARIMAX_ARMA11 | 0.25 | 6/1000 | 0.994 | -0.0016 [-0.0045, 0.0014] | 1.009 |
| ar1_060_mcar20 | SARIMAX_ARMA11 | 0.50 | 3/1000 | 0.997 | 0.0007 [-0.0021, 0.0035] | 1.014 |
| ar1_060_mcar20 | SARIMAX_SEASONAL_AR12 | 0.00 | 1/1000 | 0.999 | 0.0001 [-0.0028, 0.0029] | 0.545 |
| ar1_060_mcar20 | SARIMAX_SEASONAL_AR12 | 0.25 | 1/1000 | 0.999 | -0.0016 [-0.0046, 0.0013] | 0.546 |
| ar1_060_mcar20 | SARIMAX_SEASONAL_AR12 | 0.50 | 3/1000 | 0.997 | 0.0008 [-0.0021, 0.0036] | 0.543 |
| ar1_060_mcar20 | SEASONAL_MK_YEAR1 | 0.00 | 0/1000 | 1.000 | 0.0003 [-0.0026, 0.0032] | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR1 | 0.25 | 0/1000 | 1.000 | -0.0018 [-0.0048, 0.0012] | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR1 | 0.50 | 0/1000 | 1.000 | 0.0006 [-0.0023, 0.0035] | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR2 | 0.00 | 0/1000 | 1.000 | 0.0003 [-0.0026, 0.0032] | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR2 | 0.25 | 0/1000 | 1.000 | -0.0018 [-0.0048, 0.0012] | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR2 | 0.50 | 0/1000 | 1.000 | 0.0006 [-0.0023, 0.0035] | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR3 | 0.00 | 0/1000 | 1.000 | 0.0003 [-0.0026, 0.0032] | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR3 | 0.25 | 0/1000 | 1.000 | -0.0018 [-0.0048, 0.0012] | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR3 | 0.50 | 0/1000 | 1.000 | 0.0006 [-0.0023, 0.0035] | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR4 | 0.00 | 0/1000 | 1.000 | 0.0003 [-0.0026, 0.0032] | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR4 | 0.25 | 0/1000 | 1.000 | -0.0018 [-0.0048, 0.0012] | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR4 | 0.50 | 0/1000 | 1.000 | 0.0006 [-0.0023, 0.0035] | NA |
| ar1_060_seasonal_missing | OLS_HAC_CALENDAR12 | 0.00 | 0/1000 | 1.000 | -0.0009 [-0.0038, 0.0020] | 0.870 |
| ar1_060_seasonal_missing | OLS_HAC_CALENDAR12 | 0.25 | 0/1000 | 1.000 | 0.0010 [-0.0019, 0.0039] | 0.865 |
| ar1_060_seasonal_missing | OLS_HAC_CALENDAR12 | 0.50 | 0/1000 | 1.000 | 0.0017 [-0.0012, 0.0046] | 0.865 |
| ar1_060_seasonal_missing | SARIMAX_AR1 | 0.00 | 4/1000 | 0.996 | -0.0010 [-0.0038, 0.0019] | 1.012 |
| ar1_060_seasonal_missing | SARIMAX_AR1 | 0.25 | 4/1000 | 0.996 | 0.0008 [-0.0020, 0.0036] | 1.010 |
| ar1_060_seasonal_missing | SARIMAX_AR1 | 0.50 | 1/1000 | 0.999 | 0.0018 [-0.0011, 0.0047] | 1.013 |
| ar1_060_seasonal_missing | SARIMAX_ARMA11 | 0.00 | 3/1000 | 0.997 | -0.0013 [-0.0041, 0.0016] | 1.012 |
| ar1_060_seasonal_missing | SARIMAX_ARMA11 | 0.25 | 0/1000 | 1.000 | 0.0008 [-0.0020, 0.0036] | 1.016 |
| ar1_060_seasonal_missing | SARIMAX_ARMA11 | 0.50 | 3/1000 | 0.997 | 0.0018 [-0.0011, 0.0046] | 1.016 |
| ar1_060_seasonal_missing | SARIMAX_SEASONAL_AR12 | 0.00 | 1/1000 | 0.999 | -0.0010 [-0.0039, 0.0019] | 0.539 |
| ar1_060_seasonal_missing | SARIMAX_SEASONAL_AR12 | 0.25 | 2/1000 | 0.998 | 0.0008 [-0.0020, 0.0037] | 0.538 |
| ar1_060_seasonal_missing | SARIMAX_SEASONAL_AR12 | 0.50 | 3/1000 | 0.997 | 0.0016 [-0.0013, 0.0045] | 0.539 |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR1 | 0.00 | 0/1000 | 1.000 | -0.0005 [-0.0035, 0.0024] | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR1 | 0.25 | 0/1000 | 1.000 | 0.0008 [-0.0022, 0.0038] | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR1 | 0.50 | 0/1000 | 1.000 | 0.0018 [-0.0012, 0.0048] | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR2 | 0.00 | 0/1000 | 1.000 | -0.0005 [-0.0035, 0.0024] | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR2 | 0.25 | 0/1000 | 1.000 | 0.0008 [-0.0022, 0.0038] | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR2 | 0.50 | 0/1000 | 1.000 | 0.0018 [-0.0012, 0.0048] | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR3 | 0.00 | 0/1000 | 1.000 | -0.0005 [-0.0035, 0.0024] | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR3 | 0.25 | 0/1000 | 1.000 | 0.0008 [-0.0022, 0.0038] | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR3 | 0.50 | 0/1000 | 1.000 | 0.0018 [-0.0012, 0.0048] | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR4 | 0.00 | 0/1000 | 1.000 | -0.0005 [-0.0035, 0.0024] | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR4 | 0.25 | 0/1000 | 1.000 | 0.0008 [-0.0022, 0.0038] | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR4 | 0.50 | 0/1000 | 1.000 | 0.0018 [-0.0012, 0.0048] | NA |
| ar1_0844 | FEGLS_AR1 | 0.00 | 0/1000 | 1.000 | -0.0003 [-0.0051, 0.0046] | 0.974 |
| ar1_0844 | FEGLS_AR1 | 0.25 | 0/1000 | 1.000 | -0.0050 [-0.0100, 0.0001] | 0.985 |
| ar1_0844 | FEGLS_AR1 | 0.50 | 0/1000 | 1.000 | 0.0046 [-0.0003, 0.0095] | 1.005 |
| ar1_0844 | OLS_HAC_LAG12 | 0.00 | 0/1000 | 1.000 | -0.0002 [-0.0049, 0.0045] | 0.700 |
| ar1_0844 | OLS_HAC_LAG12 | 0.25 | 0/1000 | 1.000 | -0.0039 [-0.0088, 0.0009] | 0.703 |
| ar1_0844 | OLS_HAC_LAG12 | 0.50 | 0/1000 | 1.000 | 0.0036 [-0.0011, 0.0083] | 0.719 |
| ar1_0844 | OLS_HAC_LAG6 | 0.00 | 0/1000 | 1.000 | -0.0002 [-0.0049, 0.0045] | 0.606 |
| ar1_0844 | OLS_HAC_LAG6 | 0.25 | 0/1000 | 1.000 | -0.0039 [-0.0088, 0.0009] | 0.608 |
| ar1_0844 | OLS_HAC_LAG6 | 0.50 | 0/1000 | 1.000 | 0.0036 [-0.0011, 0.0083] | 0.621 |
| ar1_0844 | SARIMAX_AR1 | 0.00 | 9/1000 | 0.991 | 0.0003 [-0.0043, 0.0050] | 0.950 |
| ar1_0844 | SARIMAX_AR1 | 0.25 | 7/1000 | 0.993 | -0.0035 [-0.0082, 0.0011] | 0.960 |
| ar1_0844 | SARIMAX_AR1 | 0.50 | 11/1000 | 0.989 | 0.0042 [-0.0004, 0.0088] | 0.978 |
| ar1_0844 | SARIMAX_ARMA11 | 0.00 | 9/1000 | 0.991 | 0.0007 [-0.0039, 0.0053] | 0.954 |
| ar1_0844 | SARIMAX_ARMA11 | 0.25 | 8/1000 | 0.992 | -0.0034 [-0.0081, 0.0013] | 0.958 |
| ar1_0844 | SARIMAX_ARMA11 | 0.50 | 5/1000 | 0.995 | 0.0045 [-0.0001, 0.0091] | 0.973 |
| ar1_0844 | SARIMAX_SEASONAL_AR12 | 0.00 | 7/1000 | 0.993 | -0.0003 [-0.0050, 0.0044] | 0.319 |
| ar1_0844 | SARIMAX_SEASONAL_AR12 | 0.25 | 4/1000 | 0.996 | -0.0041 [-0.0089, 0.0008] | 0.322 |
| ar1_0844 | SARIMAX_SEASONAL_AR12 | 0.50 | 5/1000 | 0.995 | 0.0039 [-0.0009, 0.0086] | 0.328 |
| ar1_0844 | SEASONAL_BLOCK_YEAR2 | 0.00 | 0/1000 | 1.000 | -0.0002 [-0.0049, 0.0045] | 0.755 |
| ar1_0844 | SEASONAL_BLOCK_YEAR2 | 0.25 | 0/1000 | 1.000 | -0.0039 [-0.0088, 0.0009] | 0.759 |
| ar1_0844 | SEASONAL_BLOCK_YEAR2 | 0.50 | 0/1000 | 1.000 | 0.0036 [-0.0011, 0.0083] | 0.774 |
| ar1_0844 | SEASONAL_MK_YEAR1 | 0.00 | 0/1000 | 1.000 | -0.0003 [-0.0050, 0.0044] | NA |
| ar1_0844 | SEASONAL_MK_YEAR1 | 0.25 | 0/1000 | 1.000 | -0.0038 [-0.0087, 0.0011] | NA |
| ar1_0844 | SEASONAL_MK_YEAR1 | 0.50 | 0/1000 | 1.000 | 0.0035 [-0.0013, 0.0082] | NA |
| ar1_0844 | SEASONAL_MK_YEAR2 | 0.00 | 0/1000 | 1.000 | -0.0003 [-0.0050, 0.0044] | NA |
| ar1_0844 | SEASONAL_MK_YEAR2 | 0.25 | 0/1000 | 1.000 | -0.0038 [-0.0087, 0.0011] | NA |
| ar1_0844 | SEASONAL_MK_YEAR2 | 0.50 | 0/1000 | 1.000 | 0.0035 [-0.0013, 0.0082] | NA |
| ar1_0844 | SEASONAL_MK_YEAR3 | 0.00 | 0/1000 | 1.000 | -0.0003 [-0.0050, 0.0044] | NA |
| ar1_0844 | SEASONAL_MK_YEAR3 | 0.25 | 0/1000 | 1.000 | -0.0038 [-0.0087, 0.0011] | NA |
| ar1_0844 | SEASONAL_MK_YEAR3 | 0.50 | 0/1000 | 1.000 | 0.0035 [-0.0013, 0.0082] | NA |
| ar1_0844 | SEASONAL_MK_YEAR4 | 0.00 | 0/1000 | 1.000 | -0.0003 [-0.0050, 0.0044] | NA |
| ar1_0844 | SEASONAL_MK_YEAR4 | 0.25 | 0/1000 | 1.000 | -0.0038 [-0.0087, 0.0011] | NA |
| ar1_0844 | SEASONAL_MK_YEAR4 | 0.50 | 0/1000 | 1.000 | 0.0035 [-0.0013, 0.0082] | NA |
| arma11 | FEGLS_AR1 | 0.00 | 0/1000 | 1.000 | 0.0009 [-0.0018, 0.0035] | 1.206 |
| arma11 | FEGLS_AR1 | 0.25 | 0/1000 | 1.000 | 0.0006 [-0.0022, 0.0033] | 1.207 |
| arma11 | FEGLS_AR1 | 0.50 | 0/1000 | 1.000 | -0.0005 [-0.0031, 0.0021] | 1.201 |
| arma11 | OLS_HAC_LAG12 | 0.00 | 0/1000 | 1.000 | 0.0010 [-0.0016, 0.0036] | 0.890 |
| arma11 | OLS_HAC_LAG12 | 0.25 | 0/1000 | 1.000 | 0.0005 [-0.0022, 0.0032] | 0.887 |
| arma11 | OLS_HAC_LAG12 | 0.50 | 0/1000 | 1.000 | -0.0001 [-0.0027, 0.0024] | 0.891 |
| arma11 | OLS_HAC_LAG6 | 0.00 | 0/1000 | 1.000 | 0.0010 [-0.0016, 0.0036] | 0.871 |
| arma11 | OLS_HAC_LAG6 | 0.25 | 0/1000 | 1.000 | 0.0005 [-0.0022, 0.0032] | 0.873 |
| arma11 | OLS_HAC_LAG6 | 0.50 | 0/1000 | 1.000 | -0.0001 [-0.0027, 0.0024] | 0.875 |
| arma11 | SARIMAX_AR1 | 0.00 | 5/1000 | 0.995 | 0.0012 [-0.0014, 0.0038] | 1.210 |
| arma11 | SARIMAX_AR1 | 0.25 | 6/1000 | 0.994 | 0.0004 [-0.0023, 0.0032] | 1.206 |
| arma11 | SARIMAX_AR1 | 0.50 | 4/1000 | 0.996 | 0.0001 [-0.0024, 0.0025] | 1.199 |
| arma11 | SARIMAX_ARMA11 | 0.00 | 5/1000 | 0.995 | 0.0012 [-0.0014, 0.0037] | 1.024 |
| arma11 | SARIMAX_ARMA11 | 0.25 | 3/1000 | 0.997 | 0.0004 [-0.0023, 0.0031] | 1.022 |
| arma11 | SARIMAX_ARMA11 | 0.50 | 1/1000 | 0.999 | -0.0001 [-0.0026, 0.0024] | 1.017 |
| arma11 | SARIMAX_SEASONAL_AR12 | 0.00 | 8/1000 | 0.992 | 0.0009 [-0.0017, 0.0035] | 0.539 |
| arma11 | SARIMAX_SEASONAL_AR12 | 0.25 | 4/1000 | 0.996 | 0.0005 [-0.0023, 0.0032] | 0.536 |
| arma11 | SARIMAX_SEASONAL_AR12 | 0.50 | 1/1000 | 0.999 | -0.0002 [-0.0027, 0.0024] | 0.534 |
| arma11 | SEASONAL_BLOCK_YEAR2 | 0.00 | 0/1000 | 1.000 | 0.0010 [-0.0016, 0.0036] | 0.870 |
| arma11 | SEASONAL_BLOCK_YEAR2 | 0.25 | 0/1000 | 1.000 | 0.0005 [-0.0022, 0.0032] | 0.863 |
| arma11 | SEASONAL_BLOCK_YEAR2 | 0.50 | 0/1000 | 1.000 | -0.0001 [-0.0027, 0.0024] | 0.859 |
| arma11 | SEASONAL_MK_YEAR1 | 0.00 | 0/1000 | 1.000 | 0.0008 [-0.0019, 0.0035] | NA |
| arma11 | SEASONAL_MK_YEAR1 | 0.25 | 0/1000 | 1.000 | 0.0009 [-0.0019, 0.0037] | NA |
| arma11 | SEASONAL_MK_YEAR1 | 0.50 | 0/1000 | 1.000 | 0.0004 [-0.0022, 0.0029] | NA |
| arma11 | SEASONAL_MK_YEAR2 | 0.00 | 0/1000 | 1.000 | 0.0008 [-0.0019, 0.0035] | NA |
| arma11 | SEASONAL_MK_YEAR2 | 0.25 | 0/1000 | 1.000 | 0.0009 [-0.0019, 0.0037] | NA |
| arma11 | SEASONAL_MK_YEAR2 | 0.50 | 0/1000 | 1.000 | 0.0004 [-0.0022, 0.0029] | NA |
| arma11 | SEASONAL_MK_YEAR3 | 0.00 | 0/1000 | 1.000 | 0.0008 [-0.0019, 0.0035] | NA |
| arma11 | SEASONAL_MK_YEAR3 | 0.25 | 0/1000 | 1.000 | 0.0009 [-0.0019, 0.0037] | NA |
| arma11 | SEASONAL_MK_YEAR3 | 0.50 | 0/1000 | 1.000 | 0.0004 [-0.0022, 0.0029] | NA |
| arma11 | SEASONAL_MK_YEAR4 | 0.00 | 0/1000 | 1.000 | 0.0008 [-0.0019, 0.0035] | NA |
| arma11 | SEASONAL_MK_YEAR4 | 0.25 | 0/1000 | 1.000 | 0.0009 [-0.0019, 0.0037] | NA |
| arma11 | SEASONAL_MK_YEAR4 | 0.50 | 0/1000 | 1.000 | 0.0004 [-0.0022, 0.0029] | NA |
| iid_gaussian | FEGLS_AR1 | 0.00 | 0/1000 | 1.000 | 0.0004 [-0.0011, 0.0018] | 1.004 |
| iid_gaussian | FEGLS_AR1 | 0.25 | 0/1000 | 1.000 | -0.0008 [-0.0022, 0.0006] | 1.007 |
| iid_gaussian | FEGLS_AR1 | 0.50 | 0/1000 | 1.000 | 0.0006 [-0.0008, 0.0021] | 1.007 |
| iid_gaussian | OLS_HAC_LAG12 | 0.00 | 0/1000 | 1.000 | 0.0003 [-0.0012, 0.0017] | 0.941 |
| iid_gaussian | OLS_HAC_LAG12 | 0.25 | 0/1000 | 1.000 | -0.0006 [-0.0020, 0.0008] | 0.942 |
| iid_gaussian | OLS_HAC_LAG12 | 0.50 | 0/1000 | 1.000 | 0.0006 [-0.0009, 0.0020] | 0.947 |
| iid_gaussian | OLS_HAC_LAG6 | 0.00 | 0/1000 | 1.000 | 0.0003 [-0.0012, 0.0017] | 0.971 |
| iid_gaussian | OLS_HAC_LAG6 | 0.25 | 0/1000 | 1.000 | -0.0006 [-0.0020, 0.0008] | 0.968 |
| iid_gaussian | OLS_HAC_LAG6 | 0.50 | 0/1000 | 1.000 | 0.0006 [-0.0009, 0.0020] | 0.974 |
| iid_gaussian | SARIMAX_AR1 | 0.00 | 3/1000 | 0.997 | 0.0003 [-0.0011, 0.0018] | 1.022 |
| iid_gaussian | SARIMAX_AR1 | 0.25 | 2/1000 | 0.998 | -0.0006 [-0.0020, 0.0008] | 1.026 |
| iid_gaussian | SARIMAX_AR1 | 0.50 | 3/1000 | 0.997 | 0.0006 [-0.0009, 0.0020] | 1.027 |
| iid_gaussian | SARIMAX_ARMA11 | 0.00 | 10/1000 | 0.990 | 0.0002 [-0.0012, 0.0017] | 0.981 |
| iid_gaussian | SARIMAX_ARMA11 | 0.25 | 8/1000 | 0.992 | -0.0007 [-0.0021, 0.0007] | 0.987 |
| iid_gaussian | SARIMAX_ARMA11 | 0.50 | 5/1000 | 0.995 | 0.0006 [-0.0009, 0.0020] | 0.997 |
| iid_gaussian | SARIMAX_SEASONAL_AR12 | 0.00 | 4/1000 | 0.996 | 0.0002 [-0.0012, 0.0017] | 0.981 |
| iid_gaussian | SARIMAX_SEASONAL_AR12 | 0.25 | 7/1000 | 0.993 | -0.0006 [-0.0020, 0.0008] | 0.985 |
| iid_gaussian | SARIMAX_SEASONAL_AR12 | 0.50 | 4/1000 | 0.996 | 0.0007 [-0.0008, 0.0021] | 0.989 |
| iid_gaussian | SEASONAL_BLOCK_YEAR2 | 0.00 | 0/1000 | 1.000 | 0.0003 [-0.0012, 0.0017] | 0.887 |
| iid_gaussian | SEASONAL_BLOCK_YEAR2 | 0.25 | 0/1000 | 1.000 | -0.0006 [-0.0020, 0.0008] | 0.891 |
| iid_gaussian | SEASONAL_BLOCK_YEAR2 | 0.50 | 0/1000 | 1.000 | 0.0006 [-0.0009, 0.0020] | 0.899 |
| iid_gaussian | SEASONAL_MK_YEAR1 | 0.00 | 0/1000 | 1.000 | 0.0003 [-0.0012, 0.0018] | NA |
| iid_gaussian | SEASONAL_MK_YEAR1 | 0.25 | 0/1000 | 1.000 | -0.0006 [-0.0021, 0.0009] | NA |
| iid_gaussian | SEASONAL_MK_YEAR1 | 0.50 | 0/1000 | 1.000 | 0.0008 [-0.0007, 0.0023] | NA |
| iid_gaussian | SEASONAL_MK_YEAR2 | 0.00 | 0/1000 | 1.000 | 0.0003 [-0.0012, 0.0018] | NA |
| iid_gaussian | SEASONAL_MK_YEAR2 | 0.25 | 0/1000 | 1.000 | -0.0006 [-0.0021, 0.0009] | NA |
| iid_gaussian | SEASONAL_MK_YEAR2 | 0.50 | 0/1000 | 1.000 | 0.0008 [-0.0007, 0.0023] | NA |
| iid_gaussian | SEASONAL_MK_YEAR3 | 0.00 | 0/1000 | 1.000 | 0.0003 [-0.0012, 0.0018] | NA |
| iid_gaussian | SEASONAL_MK_YEAR3 | 0.25 | 0/1000 | 1.000 | -0.0006 [-0.0021, 0.0009] | NA |
| iid_gaussian | SEASONAL_MK_YEAR3 | 0.50 | 0/1000 | 1.000 | 0.0008 [-0.0007, 0.0023] | NA |
| iid_gaussian | SEASONAL_MK_YEAR4 | 0.00 | 0/1000 | 1.000 | 0.0003 [-0.0012, 0.0018] | NA |
| iid_gaussian | SEASONAL_MK_YEAR4 | 0.25 | 0/1000 | 1.000 | -0.0006 [-0.0021, 0.0009] | NA |
| iid_gaussian | SEASONAL_MK_YEAR4 | 0.50 | 0/1000 | 1.000 | 0.0008 [-0.0007, 0.0023] | NA |
| seasonal_ar1_12 | FEGLS_AR1 | 0.00 | 0/1000 | 1.000 | -0.0032 [-0.0057, -0.0008] | 0.534 |
| seasonal_ar1_12 | FEGLS_AR1 | 0.25 | 0/1000 | 1.000 | -0.0026 [-0.0051, -0.0001] | 0.532 |
| seasonal_ar1_12 | FEGLS_AR1 | 0.50 | 0/1000 | 1.000 | 0.0008 [-0.0016, 0.0032] | 0.532 |
| seasonal_ar1_12 | OLS_HAC_LAG12 | 0.00 | 0/1000 | 1.000 | -0.0031 [-0.0056, -0.0007] | 0.489 |
| seasonal_ar1_12 | OLS_HAC_LAG12 | 0.25 | 0/1000 | 1.000 | -0.0026 [-0.0051, -0.0001] | 0.490 |
| seasonal_ar1_12 | OLS_HAC_LAG12 | 0.50 | 0/1000 | 1.000 | 0.0008 [-0.0016, 0.0032] | 0.487 |
| seasonal_ar1_12 | OLS_HAC_LAG6 | 0.00 | 0/1000 | 1.000 | -0.0031 [-0.0056, -0.0007] | 0.506 |
| seasonal_ar1_12 | OLS_HAC_LAG6 | 0.25 | 0/1000 | 1.000 | -0.0026 [-0.0051, -0.0001] | 0.506 |
| seasonal_ar1_12 | OLS_HAC_LAG6 | 0.50 | 0/1000 | 1.000 | 0.0008 [-0.0016, 0.0032] | 0.504 |
| seasonal_ar1_12 | SARIMAX_AR1 | 0.00 | 3/1000 | 0.997 | -0.0031 [-0.0055, -0.0006] | 0.544 |
| seasonal_ar1_12 | SARIMAX_AR1 | 0.25 | 2/1000 | 0.998 | -0.0026 [-0.0051, -0.0001] | 0.541 |
| seasonal_ar1_12 | SARIMAX_AR1 | 0.50 | 6/1000 | 0.994 | 0.0006 [-0.0018, 0.0030] | 0.543 |
| seasonal_ar1_12 | SARIMAX_ARMA11 | 0.00 | 5/1000 | 0.995 | -0.0031 [-0.0056, -0.0007] | 0.559 |
| seasonal_ar1_12 | SARIMAX_ARMA11 | 0.25 | 10/1000 | 0.990 | -0.0027 [-0.0052, -0.0002] | 0.555 |
| seasonal_ar1_12 | SARIMAX_ARMA11 | 0.50 | 4/1000 | 0.996 | 0.0005 [-0.0019, 0.0030] | 0.563 |
| seasonal_ar1_12 | SARIMAX_SEASONAL_AR12 | 0.00 | 6/1000 | 0.994 | -0.0023 [-0.0046, 0.0000] | 0.869 |
| seasonal_ar1_12 | SARIMAX_SEASONAL_AR12 | 0.25 | 5/1000 | 0.995 | -0.0019 [-0.0043, 0.0005] | 0.862 |
| seasonal_ar1_12 | SARIMAX_SEASONAL_AR12 | 0.50 | 8/1000 | 0.992 | 0.0006 [-0.0017, 0.0029] | 0.863 |
| seasonal_ar1_12 | SEASONAL_BLOCK_YEAR2 | 0.00 | 0/1000 | 1.000 | -0.0031 [-0.0056, -0.0007] | 0.535 |
| seasonal_ar1_12 | SEASONAL_BLOCK_YEAR2 | 0.25 | 0/1000 | 1.000 | -0.0026 [-0.0051, -0.0001] | 0.537 |
| seasonal_ar1_12 | SEASONAL_BLOCK_YEAR2 | 0.50 | 0/1000 | 1.000 | 0.0008 [-0.0016, 0.0032] | 0.538 |
| seasonal_ar1_12 | SEASONAL_MK_YEAR1 | 0.00 | 0/1000 | 1.000 | -0.0030 [-0.0055, -0.0005] | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR1 | 0.25 | 0/1000 | 1.000 | -0.0027 [-0.0053, -0.0001] | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR1 | 0.50 | 0/1000 | 1.000 | 0.0012 [-0.0013, 0.0037] | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR2 | 0.00 | 0/1000 | 1.000 | -0.0030 [-0.0055, -0.0005] | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR2 | 0.25 | 0/1000 | 1.000 | -0.0027 [-0.0053, -0.0001] | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR2 | 0.50 | 0/1000 | 1.000 | 0.0012 [-0.0013, 0.0037] | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR3 | 0.00 | 0/1000 | 1.000 | -0.0030 [-0.0055, -0.0005] | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR3 | 0.25 | 0/1000 | 1.000 | -0.0027 [-0.0053, -0.0001] | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR3 | 0.50 | 0/1000 | 1.000 | 0.0012 [-0.0013, 0.0037] | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR4 | 0.00 | 0/1000 | 1.000 | -0.0030 [-0.0055, -0.0005] | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR4 | 0.25 | 0/1000 | 1.000 | -0.0027 [-0.0053, -0.0001] | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR4 | 0.50 | 0/1000 | 1.000 | 0.0012 [-0.0013, 0.0037] | NA |

## Residual dependence

Complete-support dependence-model diagnostics use Ljung–Box at calendar lags 12 and 24, with p < .01 marked as a residual-dependence failure. Irregular-calendar exact-offset ACFs and pair counts are descriptive and do not pass the residual-dependence gate.

| Regime | Method | Slope | Ljung–Box failures / tested | Exact-calendar ACF (lag 1 / 12 / 24) |
|---|---|---:|---:|---|
| ar1_060_mcar20 | OLS_HAC_CALENDAR12 | 0.00 | NA / 0 | 0.581 (n=1000) / -0.073 (n=1000) / -0.065 (n=1000) |
| ar1_060_mcar20 | OLS_HAC_CALENDAR12 | 0.25 | NA / 0 | 0.580 (n=1000) / -0.067 (n=1000) / -0.065 (n=1000) |
| ar1_060_mcar20 | OLS_HAC_CALENDAR12 | 0.50 | NA / 0 | 0.582 (n=1000) / -0.069 (n=1000) / -0.072 (n=1000) |
| ar1_060_mcar20 | SARIMAX_AR1 | 0.00 | NA / 0 | 0.007 (n=1000) / -0.059 (n=1000) / -0.055 (n=1000) |
| ar1_060_mcar20 | SARIMAX_AR1 | 0.25 | NA / 0 | 0.006 (n=1000) / -0.057 (n=1000) / -0.056 (n=1000) |
| ar1_060_mcar20 | SARIMAX_AR1 | 0.50 | NA / 0 | 0.006 (n=1000) / -0.058 (n=1000) / -0.060 (n=1000) |
| ar1_060_mcar20 | SARIMAX_ARMA11 | 0.00 | NA / 0 | -0.001 (n=1000) / -0.059 (n=1000) / -0.055 (n=1000) |
| ar1_060_mcar20 | SARIMAX_ARMA11 | 0.25 | NA / 0 | -0.003 (n=1000) / -0.057 (n=1000) / -0.056 (n=1000) |
| ar1_060_mcar20 | SARIMAX_ARMA11 | 0.50 | NA / 0 | -0.001 (n=1000) / -0.059 (n=1000) / -0.060 (n=1000) |
| ar1_060_mcar20 | SARIMAX_SEASONAL_AR12 | 0.00 | NA / 0 | 0.575 (n=1000) / -0.006 (n=1000) / -0.083 (n=1000) |
| ar1_060_mcar20 | SARIMAX_SEASONAL_AR12 | 0.25 | NA / 0 | 0.574 (n=1000) / -0.005 (n=1000) / -0.082 (n=1000) |
| ar1_060_mcar20 | SARIMAX_SEASONAL_AR12 | 0.50 | NA / 0 | 0.577 (n=1000) / -0.005 (n=1000) / -0.089 (n=1000) |
| ar1_060_mcar20 | SEASONAL_MK_YEAR1 | 0.00 | NA / 0 | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR1 | 0.25 | NA / 0 | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR1 | 0.50 | NA / 0 | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR2 | 0.00 | NA / 0 | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR2 | 0.25 | NA / 0 | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR2 | 0.50 | NA / 0 | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR3 | 0.00 | NA / 0 | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR3 | 0.25 | NA / 0 | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR3 | 0.50 | NA / 0 | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR4 | 0.00 | NA / 0 | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR4 | 0.25 | NA / 0 | NA |
| ar1_060_mcar20 | SEASONAL_MK_YEAR4 | 0.50 | NA / 0 | NA |
| ar1_060_seasonal_missing | OLS_HAC_CALENDAR12 | 0.00 | NA / 0 | 0.583 (n=1000) / -0.071 (n=1000) / -0.072 (n=1000) |
| ar1_060_seasonal_missing | OLS_HAC_CALENDAR12 | 0.25 | NA / 0 | 0.580 (n=1000) / -0.072 (n=1000) / -0.068 (n=1000) |
| ar1_060_seasonal_missing | OLS_HAC_CALENDAR12 | 0.50 | NA / 0 | 0.583 (n=1000) / -0.077 (n=1000) / -0.066 (n=1000) |
| ar1_060_seasonal_missing | SARIMAX_AR1 | 0.00 | NA / 0 | 0.011 (n=1000) / -0.059 (n=1000) / -0.057 (n=1000) |
| ar1_060_seasonal_missing | SARIMAX_AR1 | 0.25 | NA / 0 | 0.004 (n=1000) / -0.061 (n=1000) / -0.055 (n=1000) |
| ar1_060_seasonal_missing | SARIMAX_AR1 | 0.50 | NA / 0 | 0.008 (n=1000) / -0.057 (n=1000) / -0.054 (n=1000) |
| ar1_060_seasonal_missing | SARIMAX_ARMA11 | 0.00 | NA / 0 | 0.001 (n=1000) / -0.059 (n=1000) / -0.057 (n=1000) |
| ar1_060_seasonal_missing | SARIMAX_ARMA11 | 0.25 | NA / 0 | -0.002 (n=1000) / -0.062 (n=1000) / -0.055 (n=1000) |
| ar1_060_seasonal_missing | SARIMAX_ARMA11 | 0.50 | NA / 0 | 0.000 (n=1000) / -0.058 (n=1000) / -0.053 (n=1000) |
| ar1_060_seasonal_missing | SARIMAX_SEASONAL_AR12 | 0.00 | NA / 0 | 0.578 (n=1000) / -0.006 (n=1000) / -0.090 (n=1000) |
| ar1_060_seasonal_missing | SARIMAX_SEASONAL_AR12 | 0.25 | NA / 0 | 0.575 (n=1000) / -0.006 (n=1000) / -0.087 (n=1000) |
| ar1_060_seasonal_missing | SARIMAX_SEASONAL_AR12 | 0.50 | NA / 0 | 0.577 (n=1000) / -0.006 (n=1000) / -0.085 (n=1000) |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR1 | 0.00 | NA / 0 | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR1 | 0.25 | NA / 0 | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR1 | 0.50 | NA / 0 | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR2 | 0.00 | NA / 0 | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR2 | 0.25 | NA / 0 | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR2 | 0.50 | NA / 0 | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR3 | 0.00 | NA / 0 | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR3 | 0.25 | NA / 0 | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR3 | 0.50 | NA / 0 | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR4 | 0.00 | NA / 0 | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR4 | 0.25 | NA / 0 | NA |
| ar1_060_seasonal_missing | SEASONAL_MK_YEAR4 | 0.50 | NA / 0 | NA |
| ar1_0844 | FEGLS_AR1 | 0.00 | 1000 / 1000 | NA |
| ar1_0844 | FEGLS_AR1 | 0.25 | 1000 / 1000 | NA |
| ar1_0844 | FEGLS_AR1 | 0.50 | 1000 / 1000 | NA |
| ar1_0844 | OLS_HAC_LAG12 | 0.00 | 1000 / 1000 | NA |
| ar1_0844 | OLS_HAC_LAG12 | 0.25 | 1000 / 1000 | NA |
| ar1_0844 | OLS_HAC_LAG12 | 0.50 | 1000 / 1000 | NA |
| ar1_0844 | OLS_HAC_LAG6 | 0.00 | 1000 / 1000 | NA |
| ar1_0844 | OLS_HAC_LAG6 | 0.25 | 1000 / 1000 | NA |
| ar1_0844 | OLS_HAC_LAG6 | 0.50 | 1000 / 1000 | NA |
| ar1_0844 | SARIMAX_AR1 | 0.00 | 23 / 1000 | NA |
| ar1_0844 | SARIMAX_AR1 | 0.25 | 33 / 1000 | NA |
| ar1_0844 | SARIMAX_AR1 | 0.50 | 26 / 1000 | NA |
| ar1_0844 | SARIMAX_ARMA11 | 0.00 | 12 / 1000 | NA |
| ar1_0844 | SARIMAX_ARMA11 | 0.25 | 22 / 1000 | NA |
| ar1_0844 | SARIMAX_ARMA11 | 0.50 | 16 / 1000 | NA |
| ar1_0844 | SARIMAX_SEASONAL_AR12 | 0.00 | 1000 / 1000 | NA |
| ar1_0844 | SARIMAX_SEASONAL_AR12 | 0.25 | 1000 / 1000 | NA |
| ar1_0844 | SARIMAX_SEASONAL_AR12 | 0.50 | 1000 / 1000 | NA |
| ar1_0844 | SEASONAL_BLOCK_YEAR2 | 0.00 | 1000 / 1000 | NA |
| ar1_0844 | SEASONAL_BLOCK_YEAR2 | 0.25 | 1000 / 1000 | NA |
| ar1_0844 | SEASONAL_BLOCK_YEAR2 | 0.50 | 1000 / 1000 | NA |
| ar1_0844 | SEASONAL_MK_YEAR1 | 0.00 | NA / 0 | NA |
| ar1_0844 | SEASONAL_MK_YEAR1 | 0.25 | NA / 0 | NA |
| ar1_0844 | SEASONAL_MK_YEAR1 | 0.50 | NA / 0 | NA |
| ar1_0844 | SEASONAL_MK_YEAR2 | 0.00 | NA / 0 | NA |
| ar1_0844 | SEASONAL_MK_YEAR2 | 0.25 | NA / 0 | NA |
| ar1_0844 | SEASONAL_MK_YEAR2 | 0.50 | NA / 0 | NA |
| ar1_0844 | SEASONAL_MK_YEAR3 | 0.00 | NA / 0 | NA |
| ar1_0844 | SEASONAL_MK_YEAR3 | 0.25 | NA / 0 | NA |
| ar1_0844 | SEASONAL_MK_YEAR3 | 0.50 | NA / 0 | NA |
| ar1_0844 | SEASONAL_MK_YEAR4 | 0.00 | NA / 0 | NA |
| ar1_0844 | SEASONAL_MK_YEAR4 | 0.25 | NA / 0 | NA |
| ar1_0844 | SEASONAL_MK_YEAR4 | 0.50 | NA / 0 | NA |
| arma11 | FEGLS_AR1 | 0.00 | 1000 / 1000 | NA |
| arma11 | FEGLS_AR1 | 0.25 | 1000 / 1000 | NA |
| arma11 | FEGLS_AR1 | 0.50 | 1000 / 1000 | NA |
| arma11 | OLS_HAC_LAG12 | 0.00 | 1000 / 1000 | NA |
| arma11 | OLS_HAC_LAG12 | 0.25 | 1000 / 1000 | NA |
| arma11 | OLS_HAC_LAG12 | 0.50 | 1000 / 1000 | NA |
| arma11 | OLS_HAC_LAG6 | 0.00 | 1000 / 1000 | NA |
| arma11 | OLS_HAC_LAG6 | 0.25 | 1000 / 1000 | NA |
| arma11 | OLS_HAC_LAG6 | 0.50 | 1000 / 1000 | NA |
| arma11 | SARIMAX_AR1 | 0.00 | 674 / 1000 | NA |
| arma11 | SARIMAX_AR1 | 0.25 | 679 / 1000 | NA |
| arma11 | SARIMAX_AR1 | 0.50 | 686 / 1000 | NA |
| arma11 | SARIMAX_ARMA11 | 0.00 | 16 / 1000 | NA |
| arma11 | SARIMAX_ARMA11 | 0.25 | 16 / 1000 | NA |
| arma11 | SARIMAX_ARMA11 | 0.50 | 19 / 1000 | NA |
| arma11 | SARIMAX_SEASONAL_AR12 | 0.00 | 1000 / 1000 | NA |
| arma11 | SARIMAX_SEASONAL_AR12 | 0.25 | 1000 / 1000 | NA |
| arma11 | SARIMAX_SEASONAL_AR12 | 0.50 | 1000 / 1000 | NA |
| arma11 | SEASONAL_BLOCK_YEAR2 | 0.00 | 1000 / 1000 | NA |
| arma11 | SEASONAL_BLOCK_YEAR2 | 0.25 | 1000 / 1000 | NA |
| arma11 | SEASONAL_BLOCK_YEAR2 | 0.50 | 1000 / 1000 | NA |
| arma11 | SEASONAL_MK_YEAR1 | 0.00 | NA / 0 | NA |
| arma11 | SEASONAL_MK_YEAR1 | 0.25 | NA / 0 | NA |
| arma11 | SEASONAL_MK_YEAR1 | 0.50 | NA / 0 | NA |
| arma11 | SEASONAL_MK_YEAR2 | 0.00 | NA / 0 | NA |
| arma11 | SEASONAL_MK_YEAR2 | 0.25 | NA / 0 | NA |
| arma11 | SEASONAL_MK_YEAR2 | 0.50 | NA / 0 | NA |
| arma11 | SEASONAL_MK_YEAR3 | 0.00 | NA / 0 | NA |
| arma11 | SEASONAL_MK_YEAR3 | 0.25 | NA / 0 | NA |
| arma11 | SEASONAL_MK_YEAR3 | 0.50 | NA / 0 | NA |
| arma11 | SEASONAL_MK_YEAR4 | 0.00 | NA / 0 | NA |
| arma11 | SEASONAL_MK_YEAR4 | 0.25 | NA / 0 | NA |
| arma11 | SEASONAL_MK_YEAR4 | 0.50 | NA / 0 | NA |
| iid_gaussian | FEGLS_AR1 | 0.00 | 37 / 1000 | NA |
| iid_gaussian | FEGLS_AR1 | 0.25 | 40 / 1000 | NA |
| iid_gaussian | FEGLS_AR1 | 0.50 | 43 / 1000 | NA |
| iid_gaussian | OLS_HAC_LAG12 | 0.00 | 37 / 1000 | NA |
| iid_gaussian | OLS_HAC_LAG12 | 0.25 | 40 / 1000 | NA |
| iid_gaussian | OLS_HAC_LAG12 | 0.50 | 43 / 1000 | NA |
| iid_gaussian | OLS_HAC_LAG6 | 0.00 | 37 / 1000 | NA |
| iid_gaussian | OLS_HAC_LAG6 | 0.25 | 40 / 1000 | NA |
| iid_gaussian | OLS_HAC_LAG6 | 0.50 | 43 / 1000 | NA |
| iid_gaussian | SARIMAX_AR1 | 0.00 | 23 / 1000 | NA |
| iid_gaussian | SARIMAX_AR1 | 0.25 | 27 / 1000 | NA |
| iid_gaussian | SARIMAX_AR1 | 0.50 | 23 / 1000 | NA |
| iid_gaussian | SARIMAX_ARMA11 | 0.00 | 18 / 1000 | NA |
| iid_gaussian | SARIMAX_ARMA11 | 0.25 | 15 / 1000 | NA |
| iid_gaussian | SARIMAX_ARMA11 | 0.50 | 20 / 1000 | NA |
| iid_gaussian | SARIMAX_SEASONAL_AR12 | 0.00 | 26 / 1000 | NA |
| iid_gaussian | SARIMAX_SEASONAL_AR12 | 0.25 | 27 / 1000 | NA |
| iid_gaussian | SARIMAX_SEASONAL_AR12 | 0.50 | 26 / 1000 | NA |
| iid_gaussian | SEASONAL_BLOCK_YEAR2 | 0.00 | 37 / 1000 | NA |
| iid_gaussian | SEASONAL_BLOCK_YEAR2 | 0.25 | 40 / 1000 | NA |
| iid_gaussian | SEASONAL_BLOCK_YEAR2 | 0.50 | 43 / 1000 | NA |
| iid_gaussian | SEASONAL_MK_YEAR1 | 0.00 | NA / 0 | NA |
| iid_gaussian | SEASONAL_MK_YEAR1 | 0.25 | NA / 0 | NA |
| iid_gaussian | SEASONAL_MK_YEAR1 | 0.50 | NA / 0 | NA |
| iid_gaussian | SEASONAL_MK_YEAR2 | 0.00 | NA / 0 | NA |
| iid_gaussian | SEASONAL_MK_YEAR2 | 0.25 | NA / 0 | NA |
| iid_gaussian | SEASONAL_MK_YEAR2 | 0.50 | NA / 0 | NA |
| iid_gaussian | SEASONAL_MK_YEAR3 | 0.00 | NA / 0 | NA |
| iid_gaussian | SEASONAL_MK_YEAR3 | 0.25 | NA / 0 | NA |
| iid_gaussian | SEASONAL_MK_YEAR3 | 0.50 | NA / 0 | NA |
| iid_gaussian | SEASONAL_MK_YEAR4 | 0.00 | NA / 0 | NA |
| iid_gaussian | SEASONAL_MK_YEAR4 | 0.25 | NA / 0 | NA |
| iid_gaussian | SEASONAL_MK_YEAR4 | 0.50 | NA / 0 | NA |
| seasonal_ar1_12 | FEGLS_AR1 | 0.00 | 1000 / 1000 | NA |
| seasonal_ar1_12 | FEGLS_AR1 | 0.25 | 1000 / 1000 | NA |
| seasonal_ar1_12 | FEGLS_AR1 | 0.50 | 1000 / 1000 | NA |
| seasonal_ar1_12 | OLS_HAC_LAG12 | 0.00 | 1000 / 1000 | NA |
| seasonal_ar1_12 | OLS_HAC_LAG12 | 0.25 | 1000 / 1000 | NA |
| seasonal_ar1_12 | OLS_HAC_LAG12 | 0.50 | 1000 / 1000 | NA |
| seasonal_ar1_12 | OLS_HAC_LAG6 | 0.00 | 1000 / 1000 | NA |
| seasonal_ar1_12 | OLS_HAC_LAG6 | 0.25 | 1000 / 1000 | NA |
| seasonal_ar1_12 | OLS_HAC_LAG6 | 0.50 | 1000 / 1000 | NA |
| seasonal_ar1_12 | SARIMAX_AR1 | 0.00 | 1000 / 1000 | NA |
| seasonal_ar1_12 | SARIMAX_AR1 | 0.25 | 1000 / 1000 | NA |
| seasonal_ar1_12 | SARIMAX_AR1 | 0.50 | 1000 / 1000 | NA |
| seasonal_ar1_12 | SARIMAX_ARMA11 | 0.00 | 1000 / 1000 | NA |
| seasonal_ar1_12 | SARIMAX_ARMA11 | 0.25 | 1000 / 1000 | NA |
| seasonal_ar1_12 | SARIMAX_ARMA11 | 0.50 | 999 / 1000 | NA |
| seasonal_ar1_12 | SARIMAX_SEASONAL_AR12 | 0.00 | 17 / 1000 | NA |
| seasonal_ar1_12 | SARIMAX_SEASONAL_AR12 | 0.25 | 20 / 1000 | NA |
| seasonal_ar1_12 | SARIMAX_SEASONAL_AR12 | 0.50 | 27 / 1000 | NA |
| seasonal_ar1_12 | SEASONAL_BLOCK_YEAR2 | 0.00 | 1000 / 1000 | NA |
| seasonal_ar1_12 | SEASONAL_BLOCK_YEAR2 | 0.25 | 1000 / 1000 | NA |
| seasonal_ar1_12 | SEASONAL_BLOCK_YEAR2 | 0.50 | 1000 / 1000 | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR1 | 0.00 | NA / 0 | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR1 | 0.25 | NA / 0 | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR1 | 0.50 | NA / 0 | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR2 | 0.00 | NA / 0 | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR2 | 0.25 | NA / 0 | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR2 | 0.50 | NA / 0 | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR3 | 0.00 | NA / 0 | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR3 | 0.25 | NA / 0 | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR3 | 0.50 | NA / 0 | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR4 | 0.00 | NA / 0 | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR4 | 0.25 | NA / 0 | NA |
| seasonal_ar1_12 | SEASONAL_MK_YEAR4 | 0.50 | NA / 0 | NA |

## Paired comparisons

The machine-readable metrics file gives paired mean differences versus SARIMAX AR(1) by regime and slope, with t-based 95% Monte Carlo intervals for absolute slope error, rejection/power, interval coverage and oracle-normalized interval width. Only generated-series streams are shared across methods; estimator streams are method-specific.

## v1.2 selection root cause

The original reconstruction is preserved in results/statistical_correction/selection_reconstruction_v1_3.json. It found no arithmetic score error. The score was capability-asymmetric because methods without confidence intervals skipped coverage penalties; only one candidate passed the MCAR selection gates because exact-calendar diagnostics prescribed for missing support were not implemented and unavailable diagnostics were treated as failed. With 500 selection replications, Monte Carlo standard error near a 5% rate was about .010. Selection versus independent validation differences were compatible with sampling variability (two-sided Fisher exact p=.225 for IID and p=.112 for MCAR), but variability does not negate the independent failures. Misspecification was not established as a sole cause.

## Frozen evidence and reproducibility

- Expanded run manifest: results/statistical_correction/corrective_development_v1_3/expanded_20261117/development_manifest.json (SHA-256 3791306f1e9fa25e942eb2ce89aa1fb72e68625aeb6f54bbe167631344fb5078).
- Metrics JSON: results/statistical_correction/corrective_development_v1_3/expanded_20261117/development_metrics.json; the exact metrics and report SHA-256 values are recorded in summary_receipt.json.
- Protocol SHA-256: cecfd2939a5849eeeeb729cc9effdb4d1f460df359796bb4bca892136c4935ee; config: 0ffbe6a20c40c21f0b7b59bed6027547837cb11b93f9a25ce4505ce665eac186; runner: dad399b42c66d9b355a02ddab59496a690e26ebceb2ec23238f7582f299f131a; summarizer: b362accf19080972dfe5ed1cc755d82e15a140a178ff3e74221c510c17e589b1.
- v1.2 closure manifest SHA-256: 8b0677caead3ae6e3fe22999f0c6b078f9d2f596051ac670f4de2d7d313825e5. Older reports remain historical snapshots and are reconciled only by reports/statistical_correction/v1_2_report_reconciliation_addendum_v1_3.md.
- v1.2 historical evidence, validated inference registry, real-series scores and the 2025 hindcast were not changed. At the time the source report was generated, no publication or merge had occurred. This public-safe copy is now included in the draft PR; no PR merge has occurred.
- Reproduce/verify with: python scripts/statistical_correction/run_corrective_development_v1_3_expansion.py --plan-only; python scripts/statistical_correction/run_corrective_development_v1_3_expansion.py; python scripts/statistical_correction/summarize_corrective_development_v1_3_expansion.py.

## Bounded independent-confirmation proposal

Do not run confirmation unless this report identifies a candidate and its exact code, regime scope and acceptance gates are frozen. If a future candidate qualifies, use only the untouched namespace KUNTUR-STAT-CORRECTIVE-INDEPENDENT-CONFIRMATION-V1.3-2027-01-15, 1,000 paired outer series for null, +.25 and +.50 per-decade cells, exact two-sided intervals, the unchanged v1.2 gates, a 2× measured pilot allowance capped at 9 CPU-hours, 10 active wall-hours and 1 GiB RSS, 50-replicate atomic checkpoints, and no reuse of development or v1.2 streams. This proposal is not authorization to start the phase.

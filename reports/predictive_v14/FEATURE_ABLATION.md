# KUNTUR v1.4 feature ablation

Each group has its own inner-selected architecture/configuration/alpha and is scored on the same 120 outer months.

```
                feature_group               selected_architectures_by_outer_fold  feature_count  oof_months    mae   rmse  mean_bias  mae_skill_vs_training_climatology  monthly_win_rate  mean_prediction_variance  fold_mae_min  fold_mae_max  fold_mae_sd
             temperature_only random_forest;hist_gradient_boosting;random_forest             10         120 0.3161 0.3879    -0.0259                            -0.0244            0.5083                    0.1458        0.2745        0.3598       0.0431
                   atmosphere                      svr_rbf;svr_rbf;random_forest             16         120 0.3249 0.4027    -0.0677                            -0.0528            0.4750                    0.1842        0.2708        0.4088       0.0707
     atmosphere_precipitation                random_forest;svr_rbf;random_forest             19         120 0.3234 0.4069    -0.0793                            -0.0479            0.4833                    0.1583        0.2718        0.4081       0.0707
              atmosphere_soil                random_forest;svr_rbf;random_forest             20         120 0.3372 0.4211    -0.0869                            -0.0927            0.5000                    0.1546        0.2705        0.4383       0.0856
                  all_sources                random_forest;svr_rbf;random_forest             23         120 0.3262 0.4104    -0.0818                            -0.0569            0.4833                    0.1530        0.2682        0.4195       0.0781
all_without_atmosphere_extras                random_forest;svr_rbf;random_forest             17         120 0.3051 0.3784    -0.0398                             0.0114            0.5083                    0.1663        0.2680        0.3413       0.0375
    all_without_precipitation                random_forest;svr_rbf;random_forest             20         120 0.3337 0.4194    -0.0887                            -0.0814            0.5083                    0.1557        0.2611        0.4383       0.0897
             all_without_soil                random_forest;svr_rbf;random_forest             19         120 0.3329 0.4179    -0.0697                            -0.0788            0.4750                    0.1586        0.2716        0.4081       0.0682
```

Precipitation and soil-moisture groups are development-only because sealed 2025 predictors contain MERRA-2 only. No 2025 score is used to rank groups.

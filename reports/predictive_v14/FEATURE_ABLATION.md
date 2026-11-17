# KUNTUR v1.4 feature ablation

Each group has its own inner-selected architecture/configuration/alpha and is scored on the same 120 outer months.

```
                feature_group    selected_architectures_by_outer_fold  feature_count  oof_months    mae   rmse  mean_bias  mae_skill_vs_training_climatology  monthly_win_rate  mean_prediction_variance  fold_mae_min  fold_mae_max  fold_mae_sd
             temperature_only random_forest;elastic_net;random_forest             10         120 0.3115 0.3815    -0.0131                            -0.0102            0.5167                    0.1527        0.2730        0.3580       0.0425
                   atmosphere     random_forest;svr_rbf;random_forest             16         120 0.3265 0.4093    -0.0780                            -0.0590            0.4917                    0.1668        0.2671        0.4045       0.0688
     atmosphere_precipitation     random_forest;svr_rbf;random_forest             19         120 0.3243 0.4099    -0.0865                            -0.0518            0.4917                    0.1613        0.2692        0.4049       0.0688
              atmosphere_soil     random_forest;svr_rbf;random_forest             20         120 0.3326 0.4166    -0.0778                            -0.0786            0.4750                    0.1626        0.2662        0.4223       0.0784
                  all_sources     random_forest;svr_rbf;random_forest             23         120 0.3271 0.4111    -0.0832                            -0.0608            0.4750                    0.1578        0.2700        0.4063       0.0686
all_without_atmosphere_extras     random_forest;svr_rbf;random_forest             17         120 0.3038 0.3740    -0.0446                             0.0145            0.5000                    0.1667        0.2660        0.3394       0.0378
    all_without_precipitation     random_forest;svr_rbf;random_forest             20         120 0.3298 0.4136    -0.0778                            -0.0697            0.4667                    0.1597        0.2621        0.4223       0.0806
             all_without_soil     random_forest;svr_rbf;random_forest             19         120 0.3173 0.4035    -0.0840                            -0.0292            0.4833                    0.1661        0.2568        0.4049       0.0750
```

Precipitation and soil-moisture groups are development-only because sealed 2025 predictors contain MERRA-2 only. No 2025 score is used to rank groups.

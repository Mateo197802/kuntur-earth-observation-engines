# KUNTUR v1.4 nested model comparison

Primary comparison uses 120 chronological outer development months. Scores are paired against training-only monthly climatology.

```
                             model_id    mae   rmse  mean_bias  mae_difference_vs_training_climatology  mae_skill_vs_training_climatology  monthly_win_rate_vs_training_climatology  worst_fold_mae  paired_mae_difference_ci95_low  paired_mae_difference_ci95_high
         training_monthly_climatology 0.3083 0.4000     0.0004                                  0.0000                             0.0000                                    0.0000          0.3620                          0.0000                           0.0000
        expanding_monthly_climatology 0.3050 0.3989    -0.0042                                 -0.0033                             0.0107                                    0.3583          0.3613                         -0.0126                           0.0042
            seasonal_naive_t_minus_12 0.3894 0.5002    -0.0296                                  0.0811                            -0.2629                                    0.4167          0.5615                          0.0067                           0.2189
                persistence_t_minus_3 0.5538 0.6880    -0.0152                                  0.2454                            -0.7960                                    0.2917          0.6155                          0.1537                           0.3622
             nested_selected_pipeline 0.3051 0.3779    -0.0481                                 -0.0032                             0.0104                                    0.5083          0.3394                         -0.0614                           0.0709
nested_selected_augmentation_pipeline 0.3151 0.3894    -0.0469                                  0.0067                            -0.0218                                    0.5000          0.3404                         -0.0578                           0.0906
```

Clean nested model MAE: 0.3051 °C. Nested selected training-policy MAE: 0.3151 °C. Training-only climatology MAE: 0.3083 °C. Signed MAE difference (candidate minus climatology): 0.0067 °C; skill: -2.18%. The paired 12-month moving-block 95% interval is [-0.0578, 0.0906] °C.

```
                             model_id           scope  sample_count    mae   rmse  mean_bias
             nested_selected_pipeline outer_2012_2014            36 0.3188 0.3958     0.1089
             nested_selected_pipeline outer_2015_2017            36 0.3394 0.3974    -0.1763
             nested_selected_pipeline outer_2018_2021            48 0.2692 0.3479    -0.0696
         training_monthly_climatology outer_2012_2014            36 0.3396 0.4257     0.2225
         training_monthly_climatology outer_2015_2017            36 0.3620 0.4879    -0.2360
         training_monthly_climatology outer_2018_2021            48 0.2446 0.2923     0.0113
nested_selected_augmentation_pipeline outer_2012_2014            36 0.3207 0.3996     0.1134
nested_selected_augmentation_pipeline outer_2015_2017            36 0.3404 0.3998    -0.1778
nested_selected_augmentation_pipeline outer_2018_2021            48 0.2918 0.3733    -0.0689
```

Do not compare these scores to the previously consumed 2022–2024 or 2025 scores as if they were fresh independent tests.

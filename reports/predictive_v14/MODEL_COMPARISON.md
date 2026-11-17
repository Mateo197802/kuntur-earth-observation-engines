# KUNTUR v1.4 nested model comparison

Primary comparison uses 120 chronological outer development months. Scores are paired against training-only monthly climatology.

```
                             model_id    mae   rmse  mean_bias  mae_difference_vs_training_climatology  mae_skill_vs_training_climatology  monthly_win_rate_vs_training_climatology  worst_fold_mae  paired_mae_difference_ci95_low  paired_mae_difference_ci95_high
         training_monthly_climatology 0.3086 0.4007     0.0029                                  0.0000                             0.0000                                    0.0000          0.3640                          0.0000                           0.0000
        expanding_monthly_climatology 0.3051 0.3996    -0.0017                                 -0.0034                             0.0111                                    0.3667          0.3622                         -0.0130                           0.0044
            seasonal_naive_t_minus_12 0.3894 0.5002    -0.0296                                  0.0808                            -0.2619                                    0.4250          0.5615                          0.0061                           0.2192
                persistence_t_minus_3 0.5538 0.6880    -0.0152                                  0.2452                            -0.7946                                    0.2917          0.6155                          0.1528                           0.3621
             nested_selected_pipeline 0.3326 0.4138    -0.0826                                  0.0241                            -0.0780                                    0.4500          0.4383                         -0.0137                           0.0832
nested_selected_augmentation_pipeline 0.3409 0.4201    -0.0788                                  0.0323                            -0.1048                                    0.4500          0.4354                         -0.0112                           0.0973
```

Clean nested model MAE: 0.3326 °C. Nested selected training-policy MAE: 0.3409 °C. Training-only climatology MAE: 0.3086 °C. Signed MAE difference (candidate minus climatology): 0.0323 °C; skill: -10.48%. The paired 12-month moving-block 95% interval is [-0.0112, 0.0973] °C.

```
                             model_id           scope  sample_count    mae   rmse  mean_bias
             nested_selected_pipeline outer_2012_2014            36 0.3129 0.3831     0.1710
             nested_selected_pipeline outer_2015_2017            36 0.4383 0.5213    -0.3688
             nested_selected_pipeline outer_2018_2021            48 0.2682 0.3379    -0.0580
         training_monthly_climatology outer_2012_2014            36 0.3393 0.4258     0.2322
         training_monthly_climatology outer_2015_2017            36 0.3640 0.4900    -0.2385
         training_monthly_climatology outer_2018_2021            48 0.2440 0.2922     0.0121
nested_selected_augmentation_pipeline outer_2012_2014            36 0.3129 0.3831     0.1710
nested_selected_augmentation_pipeline outer_2015_2017            36 0.4354 0.5185    -0.3601
nested_selected_augmentation_pipeline outer_2018_2021            48 0.2911 0.3599    -0.0551
```

Do not compare these scores to the previously consumed 2022–2024 or 2025 scores as if they were fresh independent tests.

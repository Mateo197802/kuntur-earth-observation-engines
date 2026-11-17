# KUNTUR v1.4 robustness comparison

Corruption is applied only to outer-fold validation inputs after fitting. Targets stay authentic. Intensities 25% and 50% are diagnostic RMS relative to fold-training feature SD (Gaussian), missing fraction (dropout), or an L2 split budget (combined).

```
            scenario  intensity   scope  sample_count  invalid_prediction_count    mae   rmse  mean_bias  median_absolute_error  p90_absolute_error  monthly_win_rate_vs_baseline  mae_skill_vs_baseline  rmse_skill_vs_baseline  baseline_mae  baseline_rmse  delta_mae_from_clean_c  prediction_stability_rms_change_c  input_failure_count
            gaussian     0.2500 ALL_OOF           120                         0 0.3384 0.4278    -0.0622                 0.2914              0.7571                        0.4500                -0.0968                 -0.0676        0.3086         0.4007                 -0.0025                             0.1474                    0
            gaussian     0.5000 ALL_OOF           120                         0 0.3593 0.4344    -0.0407                 0.3249              0.7131                        0.4250                -0.1645                 -0.0841        0.3086         0.4007                  0.0184                             0.1708                    0
         missingness     0.2500 ALL_OOF           120                         0 0.3312 0.4101    -0.0811                 0.2827              0.7149                        0.4417                -0.0734                 -0.0235        0.3086         0.4007                 -0.0097                             0.0919                    0
         missingness     0.5000 ALL_OOF           120                         0 0.3184 0.3971    -0.0944                 0.2821              0.6974                        0.4750                -0.0319                  0.0090        0.3086         0.4007                 -0.0225                             0.1830                    0
gaussian+missingness     0.2500 ALL_OOF           120                         0 0.3341 0.4152    -0.0844                 0.2884              0.7085                        0.4667                -0.0826                 -0.0362        0.3086         0.4007                 -0.0068                             0.1176                    0
gaussian+missingness     0.5000 ALL_OOF           120                         0 0.3506 0.4317    -0.0544                 0.2920              0.7227                        0.4417                -0.1363                 -0.0774        0.3086         0.4007                  0.0097                             0.1532                    0
```

These scenarios measure synthetic input sensitivity, not generalization to an independent sensor or domain.

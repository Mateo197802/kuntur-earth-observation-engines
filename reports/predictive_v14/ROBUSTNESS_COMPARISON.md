# KUNTUR v1.4 robustness comparison

Corruption is applied only to outer-fold validation inputs after fitting. Targets stay authentic. Intensities 25% and 50% are diagnostic RMS relative to fold-training feature SD (Gaussian), missing fraction (dropout), or an L2 split budget (combined).

```
            scenario  intensity   scope  sample_count  invalid_prediction_count    mae   rmse  mean_bias  median_absolute_error  p90_absolute_error  monthly_win_rate_vs_baseline  mae_skill_vs_baseline  rmse_skill_vs_baseline  baseline_mae  baseline_rmse  delta_mae_from_clean_c  prediction_stability_rms_change_c  input_failure_count
            gaussian     0.2500 ALL_OOF           120                         0 0.3231 0.4036    -0.0347                 0.2722              0.6588                        0.4833                -0.0478                 -0.0092        0.3083         0.4000                  0.0080                             0.1547                    0
            gaussian     0.5000 ALL_OOF           120                         0 0.3422 0.4192    -0.0106                 0.2811              0.6998                        0.4667                -0.1098                 -0.0481        0.3083         0.4000                  0.0271                             0.1991                    0
         missingness     0.2500 ALL_OOF           120                         0 0.3202 0.3935    -0.0739                 0.2828              0.6250                        0.4917                -0.0384                  0.0162        0.3083         0.4000                  0.0051                             0.1134                    0
         missingness     0.5000 ALL_OOF           120                         0 0.3077 0.3858    -0.1004                 0.2513              0.6579                        0.4500                 0.0021                  0.0354        0.3083         0.4000                 -0.0074                             0.1936                    0
gaussian+missingness     0.2500 ALL_OOF           120                         0 0.3161 0.3941    -0.0833                 0.2763              0.6372                        0.4833                -0.0251                  0.0147        0.3083         0.4000                  0.0010                             0.1330                    0
gaussian+missingness     0.5000 ALL_OOF           120                         0 0.3116 0.3911    -0.0683                 0.2613              0.6390                        0.4833                -0.0108                  0.0222        0.3083         0.4000                 -0.0034                             0.1650                    0
```

These scenarios measure synthetic input sensitivity, not generalization to an independent sensor or domain.

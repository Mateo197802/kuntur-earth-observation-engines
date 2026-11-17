# KUNTUR v1.4 anomaly forecasting

The fitted model predicts a target departure from fold-training monthly climatology and applies an inner-selected alpha in [0,1]. Alpha zero is eligible in every inner search. Target labels are never perturbed.

```
                             model_id anomaly_regime  sample_count  mae_c  mean_absolute_anomaly_c
nested_selected_augmentation_pipeline  large_anomaly            13 0.6058                   0.8621
nested_selected_augmentation_pipeline       ordinary           107 0.2797                   0.2410
             nested_selected_pipeline  large_anomaly            13 0.6032                   0.8621
             nested_selected_pipeline       ordinary           107 0.2689                   0.2410
         training_monthly_climatology  large_anomaly            13 0.8621                   0.8621
         training_monthly_climatology       ordinary           107 0.2410                   0.2410
```

Seasonal prediction-bias evidence:

```
                             model_id  target_calendar_month  sample_count  mean_bias_c  mae_c  mean_prediction_c  mean_observed_c
nested_selected_augmentation_pipeline                      1            10       0.1518 0.4480            23.0482          22.8964
nested_selected_augmentation_pipeline                      2            10       0.0235 0.4238            23.0324          23.0089
nested_selected_augmentation_pipeline                      3            10      -0.0757 0.2348            22.8561          22.9317
nested_selected_augmentation_pipeline                      4            10      -0.0914 0.2291            22.7360          22.8274
nested_selected_augmentation_pipeline                      5            10      -0.0874 0.3065            22.5110          22.5984
nested_selected_augmentation_pipeline                      6            10      -0.0611 0.3439            21.9442          22.0053
nested_selected_augmentation_pipeline                      7            10       0.1451 0.3210            21.9106          21.7655
nested_selected_augmentation_pipeline                      8            10       0.1797 0.3618            22.3820          22.2023
nested_selected_augmentation_pipeline                      9            10      -0.0762 0.2590            22.6529          22.7292
nested_selected_augmentation_pipeline                     10            10      -0.1820 0.2829            22.5956          22.7776
nested_selected_augmentation_pipeline                     11            10      -0.2125 0.2936            22.5345          22.7469
nested_selected_augmentation_pipeline                     12            10      -0.2762 0.2762            22.5597          22.8359
             nested_selected_pipeline                      1            10       0.1464 0.4376            23.0428          22.8964
             nested_selected_pipeline                      2            10       0.0294 0.4224            23.0382          23.0089
             nested_selected_pipeline                      3            10      -0.0525 0.2508            22.8792          22.9317
             nested_selected_pipeline                      4            10      -0.1003 0.2079            22.7271          22.8274
             nested_selected_pipeline                      5            10      -0.0814 0.3279            22.5170          22.5984
             nested_selected_pipeline                      6            10      -0.0537 0.3306            21.9516          22.0053
             nested_selected_pipeline                      7            10       0.1293 0.2817            21.8948          21.7655
             nested_selected_pipeline                      8            10       0.1461 0.3336            22.3484          22.2023
             nested_selected_pipeline                      9            10      -0.0790 0.2543            22.6502          22.7292
             nested_selected_pipeline                     10            10      -0.1752 0.2629            22.6024          22.7776
             nested_selected_pipeline                     11            10      -0.2144 0.2804            22.5325          22.7469
             nested_selected_pipeline                     12            10      -0.2715 0.2715            22.5644          22.8359
         training_monthly_climatology                      1            10       0.1253 0.4489            23.0217          22.8964
         training_monthly_climatology                      2            10       0.0450 0.4607            23.0539          23.0089
         training_monthly_climatology                      3            10      -0.0272 0.2945            22.9046          22.9317
         training_monthly_climatology                      4            10      -0.0007 0.2916            22.8267          22.8274
         training_monthly_climatology                      5            10      -0.0294 0.3529            22.5690          22.5984
         training_monthly_climatology                      6            10       0.0119 0.2875            22.0172          22.0053
         training_monthly_climatology                      7            10       0.1711 0.3366            21.9366          21.7655
         training_monthly_climatology                      8            10       0.1442 0.3087            22.3465          22.2023
         training_monthly_climatology                      9            10      -0.0649 0.2016            22.6643          22.7292
         training_monthly_climatology                     10            10      -0.0913 0.2862            22.6863          22.7776
         training_monthly_climatology                     11            10      -0.1074 0.2177            22.6396          22.7469
         training_monthly_climatology                     12            10      -0.1713 0.2129            22.6646          22.8359
```

Large-anomaly bands use each outer fold's training-only 75th percentile of absolute target anomaly. The report groups are a development diagnostic, not a post-hoc threshold for model selection.

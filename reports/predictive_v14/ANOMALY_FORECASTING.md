# KUNTUR v1.4 anomaly forecasting

The fitted model predicts a target departure from fold-training monthly climatology and applies an inner-selected alpha in [0,1]. Alpha zero is eligible in every inner search. Target labels are never perturbed.

```
                             model_id anomaly_regime  sample_count  mae_c  mean_absolute_anomaly_c
nested_selected_augmentation_pipeline  large_anomaly            13 0.7197                   0.8644
nested_selected_augmentation_pipeline       ordinary           107 0.2949                   0.2411
             nested_selected_pipeline  large_anomaly            13 0.7287                   0.8644
             nested_selected_pipeline       ordinary           107 0.2845                   0.2411
         training_monthly_climatology  large_anomaly            13 0.8644                   0.8644
         training_monthly_climatology       ordinary           107 0.2411                   0.2411
```

Seasonal prediction-bias evidence:

```
                             model_id  target_calendar_month  sample_count  mean_bias_c  mae_c  mean_prediction_c  mean_observed_c
nested_selected_augmentation_pipeline                      1            10       0.1302 0.4610            23.0266          22.8964
nested_selected_augmentation_pipeline                      2            10      -0.0125 0.4739            22.9964          23.0089
nested_selected_augmentation_pipeline                      3            10      -0.0942 0.3674            22.8376          22.9317
nested_selected_augmentation_pipeline                      4            10      -0.1073 0.3023            22.7201          22.8274
nested_selected_augmentation_pipeline                      5            10      -0.1358 0.3872            22.4626          22.5984
nested_selected_augmentation_pipeline                      6            10      -0.1426 0.3501            21.8627          22.0053
nested_selected_augmentation_pipeline                      7            10       0.0560 0.2659            21.8215          21.7655
nested_selected_augmentation_pipeline                      8            10       0.0535 0.3032            22.2558          22.2023
nested_selected_augmentation_pipeline                      9            10      -0.1043 0.2540            22.6248          22.7292
nested_selected_augmentation_pipeline                     10            10      -0.1716 0.3603            22.6060          22.7776
nested_selected_augmentation_pipeline                     11            10      -0.1927 0.2887            22.5542          22.7469
nested_selected_augmentation_pipeline                     12            10      -0.2240 0.2771            22.6120          22.8359
             nested_selected_pipeline                      1            10       0.1232 0.4528            23.0196          22.8964
             nested_selected_pipeline                      2            10      -0.0052 0.4626            23.0037          23.0089
             nested_selected_pipeline                      3            10      -0.0934 0.3653            22.8383          22.9317
             nested_selected_pipeline                      4            10      -0.1213 0.2843            22.7061          22.8274
             nested_selected_pipeline                      5            10      -0.1330 0.3797            22.4654          22.5984
             nested_selected_pipeline                      6            10      -0.1351 0.3422            21.8702          22.0053
             nested_selected_pipeline                      7            10       0.0499 0.2597            21.8154          21.7655
             nested_selected_pipeline                      8            10       0.0315 0.2904            22.2338          22.2023
             nested_selected_pipeline                      9            10      -0.1140 0.2322            22.6152          22.7292
             nested_selected_pipeline                     10            10      -0.1752 0.3586            22.6025          22.7776
             nested_selected_pipeline                     11            10      -0.1952 0.2879            22.5518          22.7469
             nested_selected_pipeline                     12            10      -0.2230 0.2762            22.6129          22.8359
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
         training_monthly_climatology                     11            10      -0.0820 0.2188            22.6650          22.7469
         training_monthly_climatology                     12            10      -0.1668 0.2149            22.6691          22.8359
```

Large-anomaly bands use each outer fold's training-only 75th percentile of absolute target anomaly. The report groups are a development diagnostic, not a post-hoc threshold for model selection.

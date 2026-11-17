# KUNTUR v1.4 uncertainty calibration

```
                             model_id            calibration_method  nominal_coverage  sample_count  empirical_coverage  mean_width_c  median_width_c  mean_winkler_score_c  coverage_large_anomaly
nested_selected_augmentation_pipeline              expanding_pooled            0.8000           120              0.8583        1.2638          1.2106                1.5542                  0.5333
nested_selected_augmentation_pipeline              expanding_pooled            0.9000           120              0.9417        1.6920          1.6942                1.8012                  0.8000
nested_selected_augmentation_pipeline              expanding_pooled            0.9500           120              0.9833        2.0152          1.9895                2.0562                  0.9667
nested_selected_augmentation_pipeline              rolling36_pooled            0.8000           120              0.8250        1.2325          1.2106                1.6541                  0.5333
nested_selected_augmentation_pipeline              rolling36_pooled            0.9000           120              0.9000        1.6073          1.6942                1.8856                  0.7667
nested_selected_augmentation_pipeline              rolling36_pooled            0.9500           120              0.9417        1.8163          1.8315                2.1030                  0.8333
nested_selected_augmentation_pipeline expanding_same_calendar_month            0.8000           120              0.8917        1.5753          1.5997                1.7417                  0.6000
nested_selected_augmentation_pipeline expanding_same_calendar_month            0.9000           120              0.8917        1.5841          1.6017                1.9170                  0.6000
nested_selected_augmentation_pipeline expanding_same_calendar_month            0.9500           120              0.8917        1.5936          1.6017                2.2593                  0.6000
         training_monthly_climatology              expanding_pooled            0.8000           120              0.9000        1.3930          1.4038                1.5617                  0.6000
         training_monthly_climatology              expanding_pooled            0.9000           120              0.9500        1.6668          1.6122                1.8287                  0.8000
         training_monthly_climatology              expanding_pooled            0.9500           120              0.9750        1.9317          1.9097                2.1317                  0.9000
         training_monthly_climatology              rolling36_pooled            0.8000           120              0.8500        1.2842          1.5127                1.5732                  0.4667
         training_monthly_climatology              rolling36_pooled            0.9000           120              0.9333        1.7539          1.6122                1.9465                  0.7333
         training_monthly_climatology              rolling36_pooled            0.9500           120              0.9583        1.9231          2.0233                2.2523                  0.8333
         training_monthly_climatology expanding_same_calendar_month            0.8000           120              0.8917        1.5745          1.5580                1.7445                  0.5667
         training_monthly_climatology expanding_same_calendar_month            0.9000           120              0.8917        1.5800          1.5625                1.9200                  0.5667
         training_monthly_climatology expanding_same_calendar_month            0.9500           120              0.8917        1.5823          1.5625                2.2623                  0.5667
```

Coverage by season is in `results/predictive_v14/interval_calibration.csv`. Calibration is prequential and uses only earlier residuals; the first-fold warm-up residuals come from inner validation used for selection. Monthly serial dependence and small calibration samples preclude a finite-sample coverage guarantee. A narrow width is not interpreted as useful if empirical coverage is low.

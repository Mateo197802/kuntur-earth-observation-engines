# KUNTUR v1.4 uncertainty calibration

```
                             model_id            calibration_method  nominal_coverage  sample_count  empirical_coverage  mean_width_c  median_width_c  mean_winkler_score_c  coverage_large_anomaly
nested_selected_augmentation_pipeline              expanding_pooled            0.8000           120              0.8750        1.1515          1.1656                1.3841                  0.6333
nested_selected_augmentation_pipeline              expanding_pooled            0.9000           120              0.9417        1.4359          1.4071                1.6236                  0.8333
nested_selected_augmentation_pipeline              expanding_pooled            0.9500           120              0.9667        1.6490          1.6380                1.8006                  0.9333
nested_selected_augmentation_pipeline              rolling36_pooled            0.8000           120              0.8417        1.1170          1.1622                1.3875                  0.6333
nested_selected_augmentation_pipeline              rolling36_pooled            0.9000           120              0.9167        1.5095          1.6013                1.7412                  0.7333
nested_selected_augmentation_pipeline              rolling36_pooled            0.9500           120              0.9583        1.6450          1.6715                1.8371                  0.9000
nested_selected_augmentation_pipeline expanding_same_calendar_month            0.8000           120              0.8917        1.3872          1.3878                1.5517                  0.6667
nested_selected_augmentation_pipeline expanding_same_calendar_month            0.9000           120              0.8917        1.3872          1.3878                1.7162                  0.6667
nested_selected_augmentation_pipeline expanding_same_calendar_month            0.9500           120              0.8917        1.3872          1.3878                2.0451                  0.6667
         training_monthly_climatology              expanding_pooled            0.8000           120              0.9000        1.4002          1.4264                1.5467                  0.6000
         training_monthly_climatology              expanding_pooled            0.9000           120              0.9667        1.6796          1.6122                1.8243                  0.8667
         training_monthly_climatology              expanding_pooled            0.9500           120              0.9750        1.9245          1.9097                2.1134                  0.9000
         training_monthly_climatology              rolling36_pooled            0.8000           120              0.8500        1.2674          1.4602                1.5109                  0.4667
         training_monthly_climatology              rolling36_pooled            0.9000           120              0.9583        1.7341          1.6834                1.9084                  0.8333
         training_monthly_climatology              rolling36_pooled            0.9500           120              0.9667        1.9138          2.0233                2.1038                  0.8667
         training_monthly_climatology expanding_same_calendar_month            0.8000           120              0.9000        1.5841          1.5580                1.7399                  0.6000
         training_monthly_climatology expanding_same_calendar_month            0.9000           120              0.9000        1.5841          1.5580                1.8956                  0.6000
         training_monthly_climatology expanding_same_calendar_month            0.9500           120              0.9000        1.5841          1.5580                2.2071                  0.6000
```

Coverage by season is in `results/predictive_v14/interval_calibration.csv`. Calibration is prequential and uses only earlier residuals; the first-fold warm-up residuals come from inner validation used for selection. Monthly serial dependence and small calibration samples preclude a finite-sample coverage guarantee. A narrow width is not interpreted as useful if empirical coverage is low.

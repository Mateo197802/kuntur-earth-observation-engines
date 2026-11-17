# KUNTUR Predictive Engine v1.4 — scientific verdict

Preferred fallback: **training-only monthly climatology**.

- Training-only climatology OOF MAE: 0.3083 °C; RMSE 0.4000 °C; bias 0.0004 °C.
- Nested selected augmentation-policy candidate OOF MAE: 0.3151 °C; RMSE 0.3894 °C; bias -0.0469 °C.
- Candidate-minus-climatology MAE: 0.0067 °C (-2.18% skill); paired 95% moving-block interval [-0.0578, 0.0906] °C.
- Worst-fold candidate degradation versus climatology: 0.0472 °C. Acceptance gate for a trained candidate: NOT MET.
- This is development-only evidence. The 2022–2024 and 2025 evaluations were already consumed; neither was used in this search. No new independent temporal test was completed.
- The 2025 sealed feature archive is MERRA-2 only. No cross-source model score is claimed for that period.
- Forecast issue-time availability still relies on the frozen two-month product-release assumption; actual NASA publication timestamps were not verified.

Execution revision: `b1a507b2311d63674f4f776e6c0b2bf494d8a1d9`. Full fit pass: 172.25 s wall, 237.09 s CPU, 223.8 MiB peak RSS. Checkpoint-only replay: 9.76 s wall, 9.48 s CPU, 200.5 MiB peak RSS. Source SHA-256: `{"forecast_features_sha256": "67c69236a8516b11b900b90f63926c1047aae3af991b726c567b5c85020ce2ff", "frozen_split_indices_sha256": "c2fbca4ac9636622a08e778913e2edfb23cc210453bf05114e1b3775b9b0726c", "frozen_split_manifest_sha256": "6b72ab2f5db8107e9980f5e0435f94a3fe1220a4de8e055d5f6dae5e6332258e", "monthly_gold_sha256": "5883fe2f7af8a8d0a58dd87c6bc90340a157c71e0d3fed51fff3a850eb2368b2"}`.

See `MODEL_COMPARISON.md`, `FEATURE_ABLATION.md`, `UNCERTAINTY_CALIBRATION.md` and `ROBUSTNESS_COMPARISON.md` for the full numeric tables and limitations.

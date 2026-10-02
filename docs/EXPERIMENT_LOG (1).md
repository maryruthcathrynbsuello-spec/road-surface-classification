# Experiment Log

One row per training run. The run name must match `RUN_NAME` in the notebook, the saved model in Drive (`models/<run>.keras`) and `outputs/metrics_<run>.json`.
Only **validation** numbers go here. The test set is used once, in Stage 3.

| Run name | Date | Who | Model | Key settings | Epochs | Val acc | Val macro F1 | Weakest class | Notes |
|---|---|---|---|---|---|---|---|---|---|
| baseline_cnn_v1 |2026-10-02 |Mary Ruth Cathryn B. Suello | Baseline CNN | lr 1e-3, dropout 0.5, filters 32-256 |9 |1.000 |1.000 |"train" |First Run |
| mobilenetv2_v1 |2026-10-02 |Mary Ruth Cathryn B. Suello | MobileNetV2 | head lr 1e-3, fine lr 1e-5, 30 layers |epochs_head: 6, epoch_fine: 7 |1.0 |1.0 |"train" |First Run |
| efficientnetb0_v1 | | | EfficientNetB0 | same as above | | | | | |

## Rules
- Change one thing at a time, then bump the version (`_v2`, `_v3`).
- Never overwrite an old run. Old runs are evidence for the paper.
- Write down *why* you made each change in Notes.

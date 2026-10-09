# Experiment Log

One row per training run. The run name must match `RUN_NAME` in the notebook, the saved model in Drive (`models/<run>.keras`) and `outputs/metrics_<run>.json`.
Only **validation** numbers go here. The test set is used once, in Stage 3.

| Run name | Date | Who | Model | Key settings | Epochs | Val acc | Val macro F1 | Weakest class | Notes |
|---|---|---|---|---|---|---|---|---|---|
| baseline_cnn_v1 | 2026-10-02 | Mary Ruth Cathryn B. Suello | Baseline CNN | lr 1e-3, dropout 0.5, filters 32-256, batch 32 | not verified | not verified | not verified | old leaky split | INVALID. Photo copies were shared between train and val/test, so scores are inflated. Superseded by baseline_cnn_v2. First run. |

| mobilenetv2_v1 | 2026-10-02 | Mary Ruth Cathryn B. Suello | MobileNetV2 | head lr 1e-3, fine lr 1e-5, 30 layers, batch 32 | not verified | not verified | not verified | old leaky split | INVALID (leaky split). Fine-tuning code had the same base-selection bug later found in v2. Superseded by mobilenetv2_v3. First run. |

| efficientnetb0_v1 | 2026-10-02 | Mary Ruth Cathryn B. Suello | EfficientNetB0 | head lr 1e-3, fine lr 1e-5, 30 layers, batch 32 | not verified | not verified | not verified | old leaky split | INVALID (leaky split). Superseded by efficientnetb0_v2. First run. |

| baseline_cnn_v2 | 2026-10-06 | Mary Ruth Cathryn B. Suello | Baseline CNN | lr 1e-3, dropout 0.5, filters 32-256, batch 32, 423,526 params | 40 | 0.325 | 0.339 | val | Underfitting (train 0.34, val 0.33). Open manhole is the only class learned well (8 of 9). Normal recall 0.01. |

| mobilenetv2_v2 | 2026-10-06 | Mary Ruth Cathryn B. Suello | MobileNetV2 | head lr 1e-3, fine lr 1e-5, 30 layers, dropout 0.3, batch 32 | head 15 + fine 11 = 26 | 0.544 | 0.575 | val | BUG: phase 2 unfroze the augmentation block, not the backbone. Effectively a frozen-backbone result. |

| mobilenetv2_v3 | 2026-10-06 | Mary Ruth Cathryn B. Suello | MobileNetV2 | head lr 1e-3, fine lr 1e-5, 30 layers, dropout 0.3, batch 32 | head 15 + fine 10 = 25 | 0.575 | 0.598 | val | Bug fixed (base picked by largest nested model). Real fine-tuning, +3 points over v2 (noise level). Overfits after about epoch 20. |

| efficientnetb0_v2 | 2026-10-06 | Mary Ruth Cathryn B. Suello | EfficientNetB0 | head lr 1e-3, fine lr 1e-5, 30 layers, dropout 0.3, batch 32 | head 15 + fine 20 = 35 | 0.628 | 0.646 | val | Best so far. Ran all 35 epochs, so early stopping did not fire. Weakest class: pothole (F1 0.426). |

FINAL MODEL : efficientnetb0_v2
2026-10-08
--- efficientnetb0_v2 ---
backbone: efficientnetb0
layers in base: 238 | trainable layers: 24
total params: 4,057,257
file size (MB): 29.0

# efficientnetb0_v2 Testing
test accuracy 0.609, macro F1 0.652, open manhole 9 of 11, plus the confusion matrix. Present the baseline (0.325 on validation) to EfficientNetB0 as the main comparison.

Class | Test Result | Notes
water-filled pothole |	67 of 81 (83%)	| Strong
open manhole	| 9 of 11 (82%)	| Precision 1.000 means no false alarms. With 11 images, the 95% range is roughly 52% to 95%, so report the counts.
crack	| 50 of 67 (75%) | 	Good recall, but low precision (0.49) because damaged asphalt and normal get called crack
normal	| 36 of 59 (61%) | 	Fair
pothole	| 30 of 62 (48%) | 	Weak
damaged asphalt	| 42 of 104 (40%) | Weakest. 32% go to crack and 19% to pothole.

With 384 images, the overall accuracy has about ±5 points of uncertainty, about 61%.

## Rules
- Change one thing at a time, then bump the version (`_v2`, `_v3`).
- Never overwrite an old run. Old runs are evidence for the paper.
- Write down *why* you made each change in Notes.

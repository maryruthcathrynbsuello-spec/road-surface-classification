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

| efficientnetb0_v3 | 2026-11-18 | Mary Ruth Cathryn B. Suello | EfficientNetB0 | head lr 1e-3, fine lr 5e-5, 100 layers*, dropout 0.3*, batch 32 | head 10 + fine 12 = 22 | 0.692 | 0.716* | pothole | Big gain over v2 (+6.4 points, 313 vs 284 of 452 correct). Early stopping fired in fine-tuning: best val loss 0.773 at epoch 16, then 6 epochs without improvement. Val accuracy jumped from 0.586 to 0.637 in the first fine-tuning epoch. Overfit after epoch 16 (train 0.82 vs val 0.70). Weakest class: pothole (F1 about 0.47). Over-predicts pothole (97 predicted vs 68 true); 38% of damaged asphalt go to pothole. |

| efficientnetb0_v4 | 2026-11-18 | Mary Ruth Cathryn B. Suello | EfficientNetB0 | head lr 1e-3, fine lr 5e-5, 100 layers*, dropout 0.5*, batch 32, stronger augmentation (rotation 0.10, zoom 0.20, shift 0.10, brightness 0.25, contrast 0.3) | head 10 + fine 17 = 27 | 0.697 | 0.724 | pothole | Same accuracy as v3 within noise (315 vs 313 of 452) but a smaller train-validation gap (0.71 vs 0.70 at the best epoch) and lower val loss (0.760 vs 0.773). Early stopping fired: best epoch 21, stopped 6 epochs later. LR halved by ReduceLROnPlateau at epoch 8 (head) and epoch 25 (fine). Weakest classes: pothole (F1 0.464) and damaged asphalt (0.511). Chosen as the final model. |

FINAL MODEL : efficientnetb0_v4
2026-11-19
--- efficientnetb0_v4 ---
backbone: efficientnetb0
layers in base: 238 | trainable layers: [run model_details; expect about 80, v2 had 24]
total params: 4,057,257
file size (MB): 31.4

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
the test result (accuracy 0.609, macro F1 0.652, open manhole 9 of 11), the Grad-CAM note, and the border-test result.
## Rules
- Change one thing at a time, then bump the version (`_v2`, `_v3`).
- Never overwrite an old run. Old runs are evidence for the paper.
- Write down *why* you made each change in Notes.

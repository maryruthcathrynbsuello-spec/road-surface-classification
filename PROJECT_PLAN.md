# Project Plan

## Stages
| Stage | Goal | Deliverable | Owner |
|---|---|---|---|
| 0 | Understand the data: class counts, sizes, samples, mislabels | Data Description section (paper) | |
| 1 | Clean: corrupt, duplicates, blur/exposure flags, stratified split, resize 224x224 | `processed_dataset.zip`, `splits.csv`, cleaning summary | |
| 2 | Baseline CNN + transfer learning (MobileNetV2 / EfficientNetB0) | `02_baseline_cnn.ipynb`, `03_transfer_learning.ipynb` | |
| 3 | Train, validate, test once; per-class metrics; confusion matrix; Grad-CAM | Results section, figures in `docs/figures/` | |
| 4 | Streamlit app: upload image -> class, confidence, heatmap | `app/` | |

## Timeline (adjust to your deadline)
| Week | Work |
|---|---|
| 1 | Setup + Stage 0-1 (dataset exploration, cleaning, splitting) |
| 2 | Stage 2: baseline CNN and transfer learning model |
| 3 | Stage 3: tuning, testing, Grad-CAM |
| 4 | Stage 4: app, paper, presentation |

## Stage 2-3 requirements
- Training: early stopping, model checkpointing, learning-rate reduction, class weights
- Track train vs. validation curves (overfitting check)
- Evaluate **once** on the test set: accuracy, precision, recall, F1 per class, confusion matrix
- Watch confusions: crack vs. damaged asphalt, pothole vs. water-filled pothole
- Compare baseline CNN vs. transfer learning

## Definition of done (per stage)
- Notebook runs top to bottom on a fresh Colab runtime
- Merged via reviewed Pull Request
- Key numbers and figures saved in `docs/`
- Paper section drafted

## Tools by stage
| Stage | Tools |
|---|---|
| 0-1 | Colab, OpenCV, Pillow, imagehash, pandas, scikit-learn, matplotlib/seaborn |
| 2-3 | TensorFlow/Keras, KerasTuner (optional), TensorBoard, scikit-learn metrics, Grad-CAM (`tf-keras-vis` or custom) |
| 4 | Streamlit or Gradio, Hugging Face Spaces or Streamlit Community Cloud |
| Teamwork | GitHub, GitHub Desktop, shared Google Drive |

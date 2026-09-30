# Classification of Road Surface Conditions for Automated Road Maintenance Monitoring

Data Mining project for the Data Analytics course.
A Convolutional Neural Network (CNN) that classifies road-surface images into six classes.

## Objectives
- Classify road-surface images into: **crack, pothole, damaged asphalt, water-filled pothole, open manhole, normal road surface**
- Build a model that automatically identifies road-surface conditions from images
- Evaluate the model on unseen test images

**Type:** Classification  |  **Algorithm:** CNN  |  **Tools:** Python, TensorFlow/Keras, OpenCV

## Dataset
Mendeley Data: https://data.mendeley.com/datasets/c43y93xswd/1
> The dataset is **not stored in this repository.** Download it from the link above (see [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)).
> Check the license and citation details on the Mendeley page and cite the dataset in the paper.

## Pipeline

| Stage | What | Where | Status |
|---|---|---|---|
| 0 | Data understanding | `notebooks/01_data_cleaning.ipynb` | Ready |
| 1 | Cleaning (CV): corrupt files, duplicates, quality flags, split, resize | `notebooks/01_data_cleaning.ipynb` | Ready |
| 2 | Algorithm: baseline CNN + transfer learning (MobileNetV2 / EfficientNetB0) | `notebooks/02_*`, `03_*` | To do |
| 3 | Training, testing, evaluation, Grad-CAM | same notebooks | To do |
| 4 | Results system (Streamlit app) | `app/` | To do |

Textbook mapping (Data Processing Chain / DW architecture):
data source -> transformation (cleaning) -> data mart (split folders) -> data mining (CNN) -> visualization (app).

## Repository structure
```
road-surface-classification/
├── notebooks/            Colab notebooks (one owner per notebook)
├── src/                  Reusable Python code (config, helpers)
├── app/                  Streamlit app (Stage 4)
├── docs/
│   ├── SETUP_GUIDE.md    START HERE: setup for new teammates
│   ├── PROJECT_PLAN.md   Stages, tasks, timeline
│   ├── data_reports/     Small outputs: splits.csv, cleaning_summary.json
│   └── figures/          Charts and images used in the paper
├── CONTRIBUTING.md       How we work with Git (branches, PRs)
├── requirements.txt      Local installs only
└── .gitignore
```

## Quick start
1. Read [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) and finish the setup checklist.
2. Read [CONTRIBUTING.md](CONTRIBUTING.md) before making your first commit.
3. Open `notebooks/01_data_cleaning.ipynb` in Google Colab and run it top to bottom, or load the shared processed dataset from Drive.

## Golden rules
1. **Code on GitHub, data on Google Drive.** Never commit images, zips, or models.
2. **`SEED = 42`, always.** Everyone uses the same split so results are comparable.
3. **Everyone uses the same `processed_dataset.zip`** from the shared Drive folder.
4. **Split before augmenting.** Augmentation is for the training set only.
5. **Never commit directly to `main`.** Use a branch and a Pull Request.
6. **Test set is used once**, at the very end, for final evaluation.
7. **Clear notebook outputs** before committing.

## Team

| Member | Role | Branch prefix |
|---|---|---|
| _Name 1_ | Data cleaning (Stage 0-1) | `feature/data-*` |
| _Name 2_ | Models (Stage 2-3) | `feature/model-*` |
| _Name 3_ | App (Stage 4) | `feature/app-*` |
| _Name 4_ | Documentation / paper | `docs/*` |

## Results
_To be filled after Stage 3._

| Model | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| Baseline CNN | | | |
| MobileNetV2 | | | |

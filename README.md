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

## Streamlit Setup Guide

Streamlit Project Setup Guide

Follow the steps below to set up and run the Streamlit application locally.

1. Clone the Repository (IF NAGAWA NA, NO NEED)

Open your terminal and clone the repository:

git clone [REPOSITORY_URL]

Then navigate into the project folder:

cd [PROJECT_FOLDER]
2. Install Streamlit

Install Streamlit using pip:

pip install streamlit

You can verify that Streamlit was installed successfully with:

streamlit --version

3. Set Up the Virtual Environment (.venv)

Create a virtual environment named .venv:

python -m venv .venv
Activate the Virtual Environment

Windows:

.venv\Scripts\activate

macOS / Linux:

source .venv/bin/activate

Once activated, your terminal should show something similar to:

(.venv)
Install Streamlit Inside .venv

After activating the virtual environment, install Streamlit:

pip install streamlit

Important: Make sure .venv is activated before installing packages or running the application.

4. Run the Streamlit Application

From the project directory, navigate to the app folder:

cd app

Then navigate to the specific folder you are working on.

For example, if you are working on the dashboard folder:

cd dashboard

Run the Streamlit application:

streamlit run main.py

The complete command sequence will look like:

cd app
cd dashboard
streamlit run main.py

Streamlit will provide a local URL in the terminal, usually:

http://localhost:8501

Open the URL in your browser to access the application.

Quick Setup

If you already have Python and Git installed, the basic setup is:

git clone [REPOSITORY_URL]
cd [PROJECT_FOLDER]

python -m venv .venv
.venv\Scripts\activate

pip install streamlit

cd app
cd dashboard

streamlit run main.py
Notes
Always activate .venv before running the application.
Replace dashboard with the folder you are currently working on.
Replace [REPOSITORY_URL] and [PROJECT_FOLDER] with the actual repository information.
To leave the virtual environment, use:
deactivate
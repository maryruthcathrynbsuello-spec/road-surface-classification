# Setup Guide for Teammates

Do this **once**. It takes about 20-30 minutes, mostly waiting for the dataset download.

## Checklist
- [ ] 1. Google account ready
- [ ] 2. GitHub account + repo invitation accepted
- [ ] 3. Git tool installed (or GitHub Desktop)
- [ ] 4. Shared Google Drive folder added to *your* My Drive
- [ ] 5. Colab opens the notebook from GitHub
- [ ] 6. Test commit on your own branch

---

## 1. Accounts
| What | Link | Notes |
|---|---|---|
| Google account | https://accounts.google.com | Used for Drive and Colab. Use the **same** account in both. |
| GitHub account | https://github.com/signup | Send your username to the repo owner so you can be added. |

## 2. Accept the GitHub invitation
The repo owner adds you under **Settings -> Collaborators**. Check your email (or https://github.com/notifications) and click **Accept invitation**.

## 3. Download and install (pick what you need)
| Tool | Link | Needed for |
|---|---|---|
| **GitHub Desktop** (easiest) | https://desktop.github.com | Pull / commit / push with buttons |
| **or Git** (command line) | https://git-scm.com/downloads | Same, with commands |
| VS Code (optional) | https://code.visualstudio.com | Editing `.py` and `.md` files, docs |
| Python 3.10+ (only for local runs) | https://www.python.org/downloads | Not needed if you only use Colab |

If you only work in Colab you can skip installing Python. You still need GitHub Desktop or Git for `docs/` and `src/` changes.

## 4. Get the repository on your computer
**GitHub Desktop:** File -> Clone repository -> pick `road-surface-classification` -> Clone.
**Command line:**
```bash
git clone https://github.com/<OWNER>/road-surface-classification.git
cd road-surface-classification
```

## 5. Get the data (Drive, not GitHub)
The dataset is large and is **never** committed to GitHub.

### 5a. Owner only (one person, once)
1. Download the dataset: https://data.mendeley.com/datasets/c43y93xswd/1 (click **Download All**).
2. In Google Drive create a folder `road_project`.
3. Upload the zip and rename it `dataset.zip`.
4. Run notebook 01 to produce the processed dataset (see 7). It saves `outputs/processed_dataset.zip` here.
5. Right-click `road_project` -> **Share** -> add every teammate's Google email as **Editor**.

### 5b. Everyone else
1. Open the invitation email or go to Drive -> **Shared with me**.
2. Find `road_project`, right-click -> **Organize -> Add shortcut** -> choose **My Drive**.
3. Check it appears under **My Drive** as `road_project`. The notebooks expect this exact path:
   `/content/drive/MyDrive/road_project/`

Expected contents:
```
road_project/
├── dataset.zip                      raw download from Mendeley
├── models/                          trained models (.keras)
└── outputs/
    ├── processed_dataset.zip        cleaned 224x224 images in train/val/test folders
    ├── splits.csv
    ├── preprocessing_config.json
    ├── cleaning_summary.json
    └── removed_*.csv                logs of removed files
```

## 6. Open the notebook in Colab
1. Go to https://colab.research.google.com
2. **File -> Open notebook -> GitHub** tab.
3. Paste `https://github.com/<OWNER>/road-surface-classification` and pick `notebooks/01_data_cleaning.ipynb`.
   (If the repo is private, click *Authorize* in that tab first.)
4. Run the first cells. When asked, **allow Google Drive access**.

For training later, switch to a GPU: **Runtime -> Change runtime type -> T4 GPU.**

## 7. Who runs what
- **Cleaning notebook (01):** run **once by the owner**, who shares the resulting `outputs/` folder.
  Everyone else **does not rerun the cleaning.** They use the shared `processed_dataset.zip`.
  Otherwise each person could end up with slightly different data.
- **Later notebooks (02, 03):** start by unzipping the shared dataset to fast local storage:
```python
!unzip -q /content/drive/MyDrive/road_project/outputs/processed_dataset.zip -d /content/processed
```

## 8. Verify you have the same data as everyone else
Open `outputs/cleaning_summary.json` (or `docs/data_reports/cleaning_summary.json` in the repo) and compare the numbers. After unzipping, image counts per split must match `splits.csv`. If they don't, tell the group before training anything.

## 9. Make a test commit
```bash
git checkout -b test/<yourname>
echo "hello from <yourname>" > docs/test_<yourname>.txt
git add docs/test_<yourname>.txt
git commit -m "Test commit from <yourname>"
git push -u origin test/<yourname>
```
Open a Pull Request, get it reviewed, then **close it without merging** and delete the branch. Now you know the workflow works.

## Troubleshooting
| Problem | Fix |
|---|---|
| `FileNotFoundError ... road_project/dataset.zip` | The Drive shortcut isn't in **My Drive**, or the file is named differently. Check the exact path. |
| Drive mount pop-up doesn't appear | Allow pop-ups for colab.research.google.com; sign in with the right Google account. |
| Colab is very slow reading images | You're reading from Drive. Unzip to `/content/` first. |
| "Runtime disconnected" | Sessions time out. Rerun the setup and Drive-mount cells; anything saved to Drive is safe. |
| GPU not available | Free quota used up. Wait a few hours, or try later in the day. |
| `git push` rejected | Run `git pull` first, resolve any conflicts, push again. |
| Accidentally committed images or a zip | Tell the owner **immediately**. Large files stay in Git history even after deleting. |
| Notebook conflict on merge | See "Merge conflicts" in CONTRIBUTING.md |

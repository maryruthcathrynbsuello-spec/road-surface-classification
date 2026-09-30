# How We Work

We use GitHub the same way for every task: **branch -> commit -> pull request -> review -> merge.**

## The rules
- `main` always contains working code. **Nobody pushes to `main` directly.**
- One task = one branch. Branch names: `feature/<what>`, `fix/<what>`, `docs/<what>`
  (e.g. `feature/baseline-cnn`, `docs/methodology`).
- **One owner per notebook.** Two people must not edit the same `.ipynb` at the same time.
  If you need to try something on someone else's notebook, copy it to a new notebook.
- Every merge needs **at least one teammate's review.**
- Pull the latest `main` **before** you start working each day.

## Daily routine (command line)
```bash
git checkout main
git pull                              # get teammates' latest work
git checkout -b feature/my-task       # first time only; afterwards: git checkout feature/my-task
# ... do your work ...
git add notebooks/02_baseline_cnn.ipynb
git commit -m "Add baseline CNN with dropout"
git push -u origin feature/my-task
```
Then open GitHub -> **Compare & pull request** -> fill the template -> request a reviewer.

After your PR is merged:
```bash
git checkout main
git pull
git branch -d feature/my-task
```

## No terminal? Use Colab + GitHub
1. In Colab: **File -> Open notebook -> GitHub** tab, paste the repo URL, choose the notebook.
2. Work on it and run it.
3. **Edit -> Clear all outputs.**
4. **File -> Save a copy in GitHub.** Pick the repo, choose **your branch (not `main`)**, keep the same file path, and write a commit message.
5. Create the branch beforehand on GitHub (branch dropdown -> type a new name -> Create branch).
6. Open a Pull Request on GitHub.

Alternative: **GitHub Desktop** (https://desktop.github.com) gives you buttons for pull / commit / push / branch.

## Notebook rules (important)
Notebooks are JSON files that also store outputs (tables, images). That makes merge conflicts ugly.
- **Clear all outputs before committing.**
- Optional but recommended: run `pip install nbstripout && nbstripout --install` once inside your local clone. It strips outputs automatically on commit.
- Put stable, reusable code in `src/*.py` and import it. Text files merge cleanly.
- Run the notebook **top to bottom** on a fresh runtime before opening a PR, so we know it works.

## What goes where
| GitHub | Shared Google Drive |
|---|---|
| Notebooks, `.py`, README, docs | `dataset.zip`, `processed_dataset.zip` |
| `splits.csv`, `cleaning_summary.json` (in `docs/data_reports/`) | Trained models (`.keras`) |
| Figures for the paper (`docs/figures/`) | Large outputs, TensorBoard logs |

Models: save to Drive as `road_project/models/<model_name>_v<N>.keras` and put the exact filename and its test results in your PR description.

## Commit messages
Short, present tense, say what changed.
Good: `Add MobileNetV2 transfer learning notebook`, `Fix class weights order`
Bad: `update`, `final`, `asdf`

## Merge conflicts
1. Don't panic. Ask in the group chat before deleting anything.
2. `git checkout main && git pull`, then `git checkout your-branch && git merge main`.
3. If a `.ipynb` conflicts, easiest fix: keep **your** version (`git checkout --ours <file>`), then re-apply teammates' changes by hand. Or ask the notebook owner.

## Review checklist
- Runs top to bottom without errors?
- Outputs cleared? No data or model files committed?
- Uses `SEED = 42` and the shared processed dataset?
- Test set untouched until final evaluation?
- Results and settings written in the PR description?

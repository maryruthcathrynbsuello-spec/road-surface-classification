# Repo owner: how to create the repository (one time)

1. GitHub -> **New repository** -> name: `road-surface-classification` -> **Private** -> tick **Add a README: NO** (we already have one) -> Create.
2. Upload these files:
   - **Easiest:** on the empty repo page click **uploading an existing file**, drag in everything from this folder (keep the folder structure), commit to `main`.
   - **Or command line:**
     ```bash
     cd road-surface-classification
     git init
     git add .
     git commit -m "Initial project structure"
     git branch -M main
     git remote add origin https://github.com/<OWNER>/road-surface-classification.git
     git push -u origin main
     ```
3. Add teammates: **Settings -> Collaborators -> Add people**.
4. Protect `main`: **Settings -> Branches -> Add branch protection rule** -> branch `main` -> tick **Require a pull request before merging** and **Require approvals (1)**.
5. Edit `README.md`: fill in the team table and replace `<OWNER>` in docs with your GitHub username.
6. Send teammates the repo link and tell them to start with `docs/SETUP_GUIDE.md`.

> Note: GitHub's web upload skips empty folders, and files starting with a dot (like `.gitignore`) can be hidden in your file explorer. Make sure `.gitignore` and `.github/` are included, or the data-protection rules won't apply.

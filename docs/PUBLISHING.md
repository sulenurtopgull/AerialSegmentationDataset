# Publishing to GitHub

This local repository is ready to publish as `AerialSegmentationDataset` under the `sulenurtopgull` GitHub account.

## Required Authentication

GitHub CLI is installed, but the current machine is not authenticated. Sign in first:

```powershell
gh auth login
```

## Create Repository and Push

Run the following commands from this folder:

```powershell
git add .
git commit -m "Add aerial segmentation benchmark repository"
gh repo create sulenurtopgull/AerialSegmentationDataset --public --source . --remote origin --push --description "Aerial semantic segmentation dataset and benchmark results"
```

If the GitHub repository already exists, use:

```powershell
git remote add origin https://github.com/sulenurtopgull/AerialSegmentationDataset.git
git branch -M main
git push -u origin main
```

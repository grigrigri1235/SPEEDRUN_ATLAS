# Planning: Fix Git Push Blocked by Secrets

## Goal
The user's git push was rejected because the local commit `608dfc584b29071e774399fef2af07dacc26ff07` contains the Hugging Face User Access Token (`hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`). We need to reset the local commit, obscure the token, re-commit, push, and then restore the token.

## User Review Required
> [!IMPORTANT]
> We will reset the last local commit (`608dfc5`) to unstaged changes, replace the secret token in the files, commit, push, and then restore the secret token locally. No changes will be lost, but the commit history will be clean.

## Open Questions
None.

## Proposed Changes
We will run a series of git and python commands to resolve this.

### Part 1: Reset Last Local Commit
- Run `git reset HEAD~1` to undo the commit while keeping the changes.

### Part 2: Obscure Secrets in Workspace
- Run a Python script to find all occurrences of `hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` in the workspace and replace it with a placeholder `hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`.

### Part 3: Verify Obscuration
- Verify using a grep command that no instances of the token remain in the workspace.

### Part 4: Stage and Commit Changes
- Stage the clean files: `git add .`
- Commit: `git commit -m "sync (secrets scrubbed)"`

### Part 5: Push to Git
- Push to remote: `git push origin HEAD`

### Part 6: Restore Secrets Locally
- Restore the original token in local workspace files using a Python script.

### Part 7: Verification of Git Status and Workspace
- Run `git status` to ensure working directory is clean except for the unstaged/modified tokens (which we expect to be modified back to the real token).

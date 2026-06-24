# Plan: Delete Legacy Scripts and Update Folder Structure

We want to remove the legacy `scripts/` folder to clean up the workspace and update our repository documentation to reflect this change.

## Proposed Changes

### [Component: Cleanup]

#### [DELETE] [scripts/](file:///home/eran.b/takehome/scripts)
* Delete the entire legacy scripts directory containing outdated python sweep scripts and slurm configurations.

#### [MODIFY] [folder_structure.md](file:///home/eran.b/takehome/folder_structure.md)
* Remove the entry for `scripts/` from the folder organization listing.

## Verification Plan

### Manual Verification
* Run `ls /home/eran.b/takehome/` to confirm `scripts/` is removed.
* Inspect `folder_structure.md` to ensure the clean format.

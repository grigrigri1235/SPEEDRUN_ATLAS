# README & Legacy Data Cleanup Plan

## 1. Objective
To remove the "Version History Graveyard" of legacy L1/L2 sweeps from `outputs/README.md` and delete the corresponding obsolete JSON files. This will ensure future agents only use the definitive activation-space metrics (v5/v6 methodology).

## 2. Identifying Legacy Versions
According to `README.md`, the final methodology is **ACTIVATION-SPACE** similarity, established in `L1 v5` (which became script v6) and `L2 v2` (which became script v3). 

All previous versions used the deprecated **WEIGHT-SPACE** methodology.

### Legacy Targets for Deletion:
1.  **L1 Analysis Sweep (v1):** 
    *   File to delete: `outputs/l1_analysis_results.json`
    *   Action: Remove bullet point 17 from README.
2.  **L1 Analysis Sweep (v2 — Layer-wise):** 
    *   File to delete: `outputs/l1_analysis_v2_results.json` (If it still exists).
    *   Action: Remove bullet point 18 from README.
3.  **L1 Analysis Sweep (v3 — Ghost Isolation):** 
    *   File to delete: `outputs/l1_analysis_v3_results.json`
    *   Action: Remove bullet point 19 from README.
4.  **L1 Analysis Sweep (v4 — Extended Range & SD):** 
    *   File to delete: `outputs/l1_analysis_v4_results.json`
    *   Action: Remove bullet point 20 from README.
5.  **L2 Analysis Sweep (v1 — Mechanistic Comparison):** 
    *   File to delete: `outputs/l2_analysis_v1_results.json`
    *   Action: Remove bullet point 21 from README.

## 3. The "Definitive" Files to Keep
We will strictly retain:
*   `outputs/l1_analysis_v5_results.json` (Bullet point 22, the stagnation metric using activation-space).
*   `outputs/l2_analysis_v2_results.json` (Bullet point 23, the L2 stagnation metric using activation-space).

## 4. Execution Steps
1.  **File Deletion:** Run a terminal command to `rm` the 5 legacy JSON files listed above from the `outputs/` directory.
2.  **README Pruning:** Edit `outputs/README.md` to delete bullet points 17 through 21.
3.  **README Renumbering:** Renumber the remaining definitive sweeps (v5 and L2v2) so the document flows cleanly.

## MANDATORY STOP
Awaiting your approval to proceed with this cleanup plan.

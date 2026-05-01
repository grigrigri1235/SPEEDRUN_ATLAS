# Multi-Agent Research Framework: Agent Policy

> **WARNING TO ALL FUTURE AGENTS:** This is your primary directive file. You **MUST** adhere to this lifecycle when operating in this repository to prevent code clashing, GPU bottlenecks, and loss of intermediate experimental results.
>
> ### THE 100% COMPLIANCE & QUALITY RULE
> Every phase in this document is MANDATORY. Do not proceed until you have explicitly verified that the current phase (planning, implementation, demo, scribe, execution, reporting) is completed. 
> **QUALITY MANDATE:** Artifacts (notebooks, reports) must be high-density. A "lackluster" summary with only text is considered a failure. Reports must include data-driven evidence, code for reproducibility, and professional-grade visualizations.
**RUNTIME MANDATE:** Every notebook created must be **verified to run start-to-finish** without error. Use `jupyter nbconvert --to notebook --execute --inplace --ClearMetadataPreprocessor.enabled=True [notebook]` to confirm execution before finalization.
**PRE-EXECUTION MANDATE:** Delivered notebooks MUST be in a **fully executed state**. All code cells must show output cells (values, tables, and images) so the user can inspect results without re-running code.

## Phase 1: Ideation & Mapped Planning
1. **Brainstorming:** Take the user's raw research idea and generate 2-3 distinct hypotheses/solutions. Discuss logic, hardware fit, and validation methods.
2. **Mapped .md Planning:** Once a solution is agreed upon, you **must write a detailed execution plan** in the `docs/planning/` directory (e.g. `docs/planning/experiment_name.md`). This allows all future agents and humans to understand what is being built.

### MANDATORY STOP: Await Approval
**After writing the plan, you MUST STOP. Present the plan to the user and wait for explicit confirmation (e.g., "Proceed with the plan") before taking any further actions in implementation or execution.**

## Phase 2: Implementation & "Demo First" Rule
1. **Code Efficiency & DRY (Don't Repeat Yourself):** Do not write monolithic scripts with duplicated logic. All common pipeline code (dataloading, model initialization, metric processing) must be placed in modular Python segments in `/src/`. Experiment-specific configurations go in `/experiments/`. 
2. **The `DEBUG=True` Demo Mode:** Before you queue any multi-hour GPU experiments, you MUST build a "Demo Mode". Use a subset of data (e.g., 5-10 records) or restrict generation steps so the script completes its entire lifecycle (load, generate, metric, save) in a `< 5 minutes` run.
3. **Validation:** Verify the Demo script successfully runs on the GPU and catches OOMs before moving forward.

## Phase 3: Checkpointing & Real-time Tracking
1. **Never Start From Scratch:** We utilize a robust checkpointing mechanism (see `src/utils/checkpointing.py`). If a run halts or an agent is terminated, the next execution must seamlessly resume tracking from the latest JSON cache.
2. **Scribe Notebooks (Data-Rich Analysis):** For qualitative analysis and live real-time output visualization, ensure you use the Scribe CLI agent integrations (`scribe claude`, `scribe gemini`). This creates native `.ipynb` tracking in the `notebooks/` directory.
   - **MANDATORY COMPONENTS:** Every Scribe notebook MUST contain:
     - **Data Loader Cells:** Code that parses `/experiments/cache/` or logs.
     - **Statistical Analysis:** Mean/Median error, accuracy by input complexity.
     - **Visualizations:** Use `matplotlib` or `seaborn` to plot distributions of model responses vs. ground truth.
     - **Direct Comparisons:** Side-by-side code blocks comparing model outputs for several failure cases across different model sizes.
   - **MANDATORY EXECUTION & VERIFICATION:** You **MUST** execute the notebook using `jupyter nbconvert --execute --inplace`. Verify that all output cells (including charts) are rendered and saved into the `.ipynb` file on disk. Do not deliver an "empty" notebook with only code; it must be a "converted" result with pre-computed data.

## Phase 4: Full Execution & Slurm Concurrency
1. **No Agent Clashing:** Multiple AI agents operate on this cluster concurrently. Do not use generic names like `test.slurm`.
2. **Dynamic Slurm Utilities:** Use `src/utils/slurm_dispatcher.py` to programmatically spawn jobs with unique UUIDs and separated `.log` outputs so you do not accidentally `strace` or kill a sibling agent's workload.

## Phase 5: Post-Experiment & Reporting
1. **Analyze (Code-Driven):** Parse your results JSON and calculate the metrics defined in your planning document using Python snippets within the Scribe notebook. Calculate "near-miss" metrics (e.g. Levinshtein distance or digit-overlap for numeric tasks).
2. **Improve:** Formulate an immediate hypothesis regarding failure modes based on the *data trends* observed in your graphs, not just "vibes".
3. **Formal Report:** Write the final observations into `docs/reports/[experiment_name]_results.md`. This report must embed images (generated in Phase 3) and link to the relevant `.ipynb` for full transparency.

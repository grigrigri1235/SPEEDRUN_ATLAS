# Universal Output Protocol (Uni-Code)

This plan outlines the standardization of experimental outputs across the NeurIPS suite to prevent "KeyErrors" and facilitate automated `main.tex` updates.

## 1. Problem Definition
Currently, outputs are fragmented:
- **Scripts**: CSV with varying column names (`Intensity` vs `Temperature`).
- **Yanai**: JSON with a flat `Data` structure and ad-hoc descriptive keys.
- **Amit**: JSON with nested `accuracies` structure.

This makes it impossible to write a single analysis tool to aggregate data across the whole Topic A suite.

## 2. Revised Standard Schema: The "Series" Approach
To support multiple experiment types (L1, L2, etc.) and future plotting, we use a **Series-based JSON Schema**. This allows us to group related data points for line/bar charts while keeping baselines isolated for reference.

```json
{
  "metadata": {
    "experiment_id": "01_mechanism_sweep",
    "target_model": "Both",
    "experiment_phase": "Distillation",
    "n_models": 10
  },
  "baselines": {
    "teacher_standard": { "mean": 0.943, "std": 0.002, "raw": [...] },
    "student_chance": { "mean": 0.10, "std": 0.0, "raw": [0.10, ...] }
  },
  "data_series": [
    {
      "series_id": "L1_Sparsity_Sweep",
      "group": "Both",
      "x_axis": { "label": "lambda", "value": 0.0001 },
      "metrics": { "accuracy_mean": 0.110, "accuracy_std": 0.015 },
      "raw": [...]
    },
    {
      "series_id": "L1_Sparsity_Sweep",
      "group": "Student-Only",
      "x_axis": { "label": "lambda", "value": 0.0001 },
      "metrics": { "accuracy_mean": 0.323, "accuracy_std": 0.121 },
      "raw": [...]
    }
  ]
}
```

## 3. Implementation Features

- **Plot-Ready**: `data_series` can be converted to a Pandas DataFrame in one line for `seaborn.lineplot(hue="group", x="lambda", y="accuracy_mean")`.
- **In-File Baselines**: Every result file carries the baseline it was compared against, ensuring no "drift" when analyzing files from different dates.
- **Multimodal**: A single file can contain many `series_id` sets (e.g., one for L1, one for L2).

## 4. Migration Strategy
1. **The `UniLogger`**: A Python class that:
   - Initializes with mandatory `target_model` (Teacher/Student/Both) and `experiment_phase` (Training/Distillation/Both).
   - Manages the `data_series` buffer and the `baselines` dictionary.
2. **Context-Aware Logging**: In your sweep loops, instead of manually appending to lists, you call:
   ```python
   logger.log_point(series="L1", x_label="lambda", x_val=1e-4, group="Both", raw=accs)
   ```
3. **Save and Aggregate**: At the end of the script, `logger.save()` writes the unified JSON.

## 4. Immediate Benefits
- **Robustness**: No more `KeyError: "Intensity"` when running extraction.
- **Traceability**: Every output file contains the metadata (N_MODELS, Phase) required by the Paper standard.
- **Automation**: The Latex update becomes a 1-click operation.

---
**Next Step**: Create the `UniLogger` implementation and start migrating `01_mechanism_sweep.py` as a template.

import json
import os
import numpy as np
from datetime import datetime

class UniLogger:
    def __init__(self, experiment_id, target_model, experiment_phase, n_models):
        """
        experiment_id: str (e.g. '01_mechanism_sweep')
        target_model: str ('Teacher', 'Student', or 'Both')
        experiment_phase: str ('Training', 'Distillation', or 'Both')
        n_models: int (Number of models in ensemble)
        """
        self.output_data = {
            "metadata": {
                "experiment_id": experiment_id,
                "target_model": target_model,
                "experiment_phase": experiment_phase,
                "n_models": n_models,
                "timestamp": datetime.now().isoformat()
            },
            "baselines": {},
            "data_series": []
        }

    def log_baseline(self, name, raw_accuracies):
        """
        Store a reference baseline (e.g. Teacher accuracy).
        """
        self.output_data["baselines"][name] = {
            "accuracy_mean": float(np.mean(raw_accuracies)),
            "accuracy_std": float(np.std(raw_accuracies)),
            "raw": [float(x) for x in raw_accuracies]
        }

    def log_point(self, series_id, group, x_label, x_value, raw_accuracies, target_model=None, experiment_phase=None):
        """
        Log a single data point in a series (e.g. L1 sweep student-only).
        target_model and experiment_phase can be overridden if they differ from the global metadata.
        """
        self.output_data["data_series"].append({
            "series_id": series_id,
            "group": group,
            "target_model": target_model or self.output_data["metadata"]["target_model"],
            "experiment_phase": experiment_phase or self.output_data["metadata"]["experiment_phase"],
            "x_axis": {"label": x_label, "value": x_value},
            "metrics": {
                "accuracy_mean": float(np.mean(raw_accuracies)),
                "accuracy_std": float(np.std(raw_accuracies))
            },
            "raw": [float(x) for x in raw_accuracies]
        })

    def save(self, filename):
        """
        Save the unified JSON to the designated outputs folder.
        """
        out_dir = "/home/eran.b/takehome/outputs"
        os.makedirs(out_dir, exist_ok=True)
        
        # Ensure .json extension
        if not filename.endswith(".json"):
            filename += ".json"
            
        target_path = os.path.join(out_dir, filename)
        
        with open(target_path, "w") as f:
            json.dump(self.output_data, f, indent=2)
        print(f"✅ Uni-Code JSON saved: {target_path}")

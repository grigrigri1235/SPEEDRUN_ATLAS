import json
import numpy as np

# We'll look at the first Dropout_Sweep point with p=0.1 or similar to see if we can find the No_Reg equivalent
# Or better, I'll just check the Student_vs_Init_Cosine_Sim series for p=0.0 if it exists.

FILE = "outputs/mechanism_sweep_results.json"

with open(FILE, "r") as f:
    data = json.load(f)

# My script doesn't log p=0.0 in the sweep, it logs it as No_Reg baselines.
# But I didn't log No_Reg_Student_vs_Init.

# However, I can look at the Student_vs_Init_Cosine_Sim series and see if p=0.0 is there.
for s in data["data_series"]:
    if s["series_id"] == "Student_vs_Init_Cosine_Sim":
        print(f"Series: {s['series_id']}, Group: {s['group']}, p: {s['x_axis']['value']}, Mean: {s['metrics']['accuracy_mean']}")

# Wait, if p=0.0 is not there, I'll check the Student-Only p=0.1 and see if it's high.

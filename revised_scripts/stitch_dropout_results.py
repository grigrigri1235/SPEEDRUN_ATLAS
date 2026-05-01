"""
revised_scripts/stitch_dropout_results.py
Merges the new 15-epoch dropout data back into mechanism_sweep_results.json.
"""
import json
import os

MASTER_JSON = "/home/eran.b/takehome/outputs/mechanism_sweep_results.json"
STAGE_JSON = "/home/eran.b/takehome/outputs/dropout_15e_stage.json"

if not os.path.exists(MASTER_JSON) or not os.path.exists(STAGE_JSON):
    print("❌ Cannot stitch: Missing JSON files.")
    exit(1)

with open(MASTER_JSON, "r") as f:
    master_data = json.load(f)

with open(STAGE_JSON, "r") as f:
    stage_data = json.load(f)

# Purge legacy Dropout_Sweep
filtered_series = [
    series for series in master_data.get("data_series", [])
    if series.get("series_id") != "Dropout_Sweep"
]

# Append new data (including new metrics logged in the stage file)
new_series = stage_data.get("data_series", [])
filtered_series.extend(new_series)

master_data["data_series"] = filtered_series

# Overwrite mechanism sweep
with open(MASTER_JSON, "w") as f:
    json.dump(master_data, f, indent=2)

print(f"✅ Stitched {len(new_series)} new series. Legacy Dropout_Sweep overridden.")

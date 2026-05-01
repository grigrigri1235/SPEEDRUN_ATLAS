import json, glob, re, os

def get_grouped_avg(filename, series_id=None):
    path = f"outputs/{filename}"
    if not os.path.exists(path): return None
    with open(path) as f:
        data = json.load(f)
    if "data_series" not in data: return None
    
    stds = {}
    for p in data["data_series"]:
        _sid = p.get("series_id", "")
        if series_id is None or _sid == series_id or (isinstance(series_id, str) and series_id in _sid):
            g = p.get("group", "None")
            if g not in stds: stds[g] = []
            stds[g].append(p["metrics"]["accuracy_std"])
    res = {}
    for g, vals in stds.items():
        if vals: res[g] = sum(vals)/len(vals)
    return res

def get_globbed_avg(pattern, target_group="Ghost_Logits"):
    paths = glob.glob(f"outputs/{pattern}")
    vals = {}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for p in data.get("data_series", []):
            if p.get("group") == target_group or (target_group == "Ghost_Logits" and p.get("series_id") == "Shared_Init"):
                g = "All"
                if g not in vals: vals[g] = []
                vals[g].append(p["metrics"]["accuracy_std"])
    res = {}
    for g, v in vals.items():
        if v: res[g] = sum(v)/len(v)
    return res

print("Fig 4: Dropout")
print(get_grouped_avg("mechanism_sweep_results.json", "Dropout_Sweep"))

print("Fig 6: Clipping")
print(get_globbed_avg("clip_*.json"))

print("Fig 8: Temperature")
print(get_grouped_avg("geometry_sweep_results.json", "Temp_"))

print("Fig 9: Distill Epochs")
print(get_globbed_avg("distill_ep_*.json"))

print("Fig 10: Learning Rate")
print(get_globbed_avg("lr_*.json"))

print("Fig 11: Batch Size")
print(get_grouped_avg("batch_size_dynamics.json", "Ghost_Logits_Sweep"))

print("Fig 12: Teacher Drift")
print(get_globbed_avg("teacher_ep_*.json"))

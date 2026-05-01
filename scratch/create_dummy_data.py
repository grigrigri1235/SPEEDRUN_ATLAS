import sys
import os
import numpy as np
sys.path.append("/home/eran.b/takehome")
from utils.logger import UniLogger

def create_dummy_data():
    out_dir = "/home/eran.b/takehome/outputs"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Frankenstein
    ul = UniLogger("frankenstein_teacher", "Both", "Both", 10)
    ul.log_baseline("Standard Teacher", [0.9439] * 10)
    ul.log_point("Classification_Intervention", "Re-init_Classification", "intensity", 0, [0.9324] * 10)
    ul.save("frankenstein_teacher.json")

    # 2. Structural Sweep
    ul = UniLogger("structural_sweep", "Student", "Both", 10)
    # L1
    ul.log_point("L1_Sparsity_Sweep", "Teacher_Only", "lambda", 0.0001, [0.101] * 10)
    ul.log_point("L1_Sparsity_Sweep", "Student_Only", "lambda", 0.0001, [0.321] * 10)
    ul.log_point("L1_Sparsity_Sweep", "Both", "lambda", 0.0001, [0.106] * 10)
    # L2
    ul.log_point("L2_WeightDecay_Sweep", "Both", "wd", 0.0001, [0.133] * 10)
    ul.log_point("L2_WeightDecay_Sweep", "Teacher_Only", "wd", 0.0001, [0.161] * 10)
    ul.log_point("L2_WeightDecay_Sweep", "Student_Only", "wd", 0.0001, [0.427] * 10)
    ul.save("structural_sweep.json")

    # 3. Mechanism Sweep (Dropout)
    ul = UniLogger("mechanism_sweep", "Both", "Distillation", 10)
    ul.log_point("Dropout_Sweep", "Teacher_Only", "p", 0.5, [0.493] * 10)
    ul.log_point("Dropout_Sweep", "Student_Only", "p", 0.5, [0.134] * 10)
    ul.log_point("Dropout_Sweep", "Both", "p", 0.5, [0.158] * 10)
    ul.save("mechanism_sweep.json")

if __name__ == "__main__":
    create_dummy_data()

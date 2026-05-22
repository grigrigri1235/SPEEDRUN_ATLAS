import os
import uuid
import subprocess

def launch_job(script_path: str, name: str = "job", gpus: int = 1, time_str: str = "01:00:00", demo: bool = False):
    job_id = uuid.uuid4().hex[:8]
    job_name = f"{name}_{job_id}"
    slurm_script_path = f"slurm_jobs/{job_name}.slurm"
    
    demo_flag = " --demo" if demo else ""
    
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_jobs/{job_name}_%j.log
#SBATCH --partition=all
#SBATCH --gres=gpu:{gpus}
#SBATCH --time={time_str}

eval "$(conda shell.bash hook)"
conda activate hf_research
export HF_TOKEN="hf_DUMMY_TOKEN_PLACEHOLDER"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd ~/assignment

echo "Starting job {job_name}..."
python3 {script_path}{demo_flag}
echo "Job completed!"
"""
    os.makedirs("slurm_jobs", exist_ok=True)
    with open(slurm_script_path, "w") as f:
        f.write(slurm_content)
    
    print(f"Submitting {slurm_script_path}")
    subprocess.run(["sbatch", slurm_script_path])
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("--name", default="batch_size_dynamics")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    
    launch_job(args.script, args.name, demo=args.demo)

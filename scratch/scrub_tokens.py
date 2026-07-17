import os
import glob

def scrub_dir(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".slurm"):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                
                if 'HF_TOKEN="hf_' in content or "HF_TOKEN='hf_" in content:
                    # Replace the actual token with a placeholder
                    import re
                    new_content = re.sub(r'HF_TOKEN=[\'"]hf_[a-zA-Z0-9]+[\'"]', 'HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"', content)
                    with open(filepath, 'w') as f:
                        f.write(new_content)
                    print(f"Scrubbed {filepath}")

scrub_dir(".")

import os
import re

files_to_scrub = [
    "yanai/src/utils/slurm_dispatcher.py",
    "yanai/test_models_deleted.slurm.bak"
]

for filepath in files_to_scrub:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Replace the actual token with a placeholder
        new_content = re.sub(r'[\'"]hf_[a-zA-Z0-9]+[\'"]', '"hf_DUMMY_TOKEN_PLACEHOLDER"', content)
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Scrubbed {filepath}")


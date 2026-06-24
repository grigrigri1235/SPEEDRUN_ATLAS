import os
import re

def restore_dir(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".slurm") or file.endswith(".py") or file.endswith(".bak"):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                
                if '<HIDDEN_TOKEN_FOR_GIT>' in content:
                    new_content = content.replace('<HIDDEN_TOKEN_FOR_GIT>', '<HIDDEN_TOKEN_FOR_GIT>')
                    with open(filepath, 'w') as f:
                        f.write(new_content)
                    print(f"Restored {filepath}")

restore_dir(".")

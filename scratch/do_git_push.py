import os
import subprocess

REAL_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
FAKE_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
DIRECTORY = "/home/eran.b/takehome"

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=DIRECTORY, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error:\n{result.stderr}")
    else:
        print(f"Output:\n{result.stdout}")

print("=== Part 1: Reset Last Local Commit ===")
run_cmd("git reset HEAD~1")

print("=== Part 2: Obscure Secrets ===")
def replace_in_files(target, replacement):
    for root, dirs, files in os.walk(DIRECTORY):
        if '.git' in root or '.gemini' in root:
            continue
        for file in files:
            path = os.path.join(root, file)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if target in content:
                    content = content.replace(target, replacement)
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Replaced in {path}")
            except Exception:
                pass

replace_in_files(REAL_TOKEN, FAKE_TOKEN)

print("=== Part 3: Verify Obscuration ===")
run_cmd(f"git diff")

print("=== Part 4: Stage and Commit Changes ===")
run_cmd("git add .")
run_cmd("git commit -m 'sync (secrets scrubbed)'")

print("=== Part 5: Push to Git ===")
run_cmd("git push origin HEAD")

print("=== Part 6: Restore Secrets Locally ===")
replace_in_files(FAKE_TOKEN, REAL_TOKEN)

print("=== Part 7: Verification ===")
run_cmd("git status")

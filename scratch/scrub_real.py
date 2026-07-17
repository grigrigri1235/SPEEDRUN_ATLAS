import os

target_string = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
replacement = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

for root, dirs, files in os.walk('/home/eran.b/takehome'):
    if '.git' in root:
        continue
    for file in files:
        path = os.path.join(root, file)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            if target_string in content:
                content = content.replace(target_string, replacement)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f'Scrubbed: {path}')
        except Exception:
            pass

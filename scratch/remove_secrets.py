import os

target_string = "hf_DUMMY_TOKEN_PLACEHOLDER"

for root, dirs, files in os.walk('/home/eran.b/takehome'):
    if '.git' in root:
        continue
    for file in files:
        path = os.path.join(root, file)
        try:
            with open(path, 'r') as f:
                content = f.read()
            if target_string in content:
                # Replace with placeholder
                content = content.replace(target_string, "hf_DUMMY_TOKEN_PLACEHOLDER")
                with open(path, 'w') as f:
                    f.write(content)
                print(f'Scrubbed: {path}')
        except Exception as e:
            pass

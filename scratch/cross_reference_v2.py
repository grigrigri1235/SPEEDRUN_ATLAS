import os

# Experiments from main.tex
experiments = {
    "Head-Reset": "93.3",
    "L1 Sparsity": "L1",
    "L2 Weight Decay": "L2",
    "Dropout": "dropout",
    "Representational Centering": "centering",
    "Trust-Region Clipping": "clipping",
    "RLVR": "RLVR",
    "Cosine vs MSE": "MSE",
    "Activation Sharpness": "temperature",
    "Temporal Dynamics": "epochs",
    "74.6": "74.6",
    "81.8": "81.8",
    "Batch Size 4096": "4096",
    "Teacher Weight Drift": "Weight Drift",
    "Disjointed Curriculum": "forgetting",
    "Gaussian": "Gaussian",
    "Contrastive": "contrastive",
    "MNIST": "MNIST"
}

text_dir = "/home/eran.b/takehome/scratch/paper_texts"
papers = ["subliminal_learning.txt", "comments_and_extensions.txt", "token_entanglement.txt"]

for paper in papers:
    path = os.path.join(text_dir, paper)
    print(f"========== {paper} ==========")
    with open(path, "r", errors="ignore") as f:
        content = f.read()
        for name, keyword in experiments.items():
            if keyword.lower() in content.lower():
                idx = content.lower().find(keyword.lower())
                start = max(0, idx - 100)
                end = min(len(content), idx + 100)
                context = content[start:end].replace("\n", " ")
                print(f"[{name}] Found keyword '{keyword}' in context: ...{context}...")

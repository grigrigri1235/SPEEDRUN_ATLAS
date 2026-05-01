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

results = {paper: [] for paper in papers}

for paper in papers:
    path = os.path.join(text_dir, paper)
    with open(path, "r", errors="ignore") as f:
        content = f.read()
        for name, keyword in experiments.items():
            if keyword.lower() in content.lower():
                # Check for specific numbers to avoid false positives
                results[paper].append(name)

for paper, found in results.items():
    print(f"--- {paper} ---")
    print(", ".join(found) if found else "None")

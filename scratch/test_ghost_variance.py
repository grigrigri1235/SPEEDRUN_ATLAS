import torch as t
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revised_scripts.ViT_sub import MultiClassifier, ViTEnsemble, get_mnist, to_tensor, PreloadedDataLoader

t.manual_seed(0)
DEVICE = "cuda" if t.cuda.is_available() else "cpu"

print("--- Testing Ghost Logit Variance on Noise ---")
N_MODELS = 1
train_ds, _ = get_mnist()
train_x_s, train_y = to_tensor(train_ds)
train_x = train_x_s.unsqueeze(0)
rand_imgs = t.rand_like(train_x) * 2 - 1

mlp_teacher = MultiClassifier(N_MODELS, [784, 256, 256, 13]).to(DEVICE)
vit_teacher = ViTEnsemble(N_MODELS, []).to(DEVICE)

def train_teacher(teacher, name):
    print(f"\nTraining {name} teacher...")
    opt = t.optim.Adam(teacher.parameters(), lr=3e-4)
    for epoch in range(2):
        for bx, by in PreloadedDataLoader(train_x, train_y, 256):
            logits = teacher(bx)[..., :10].flatten(0, 1)
            loss = t.nn.functional.cross_entropy(logits, by.flatten())
            opt.zero_grad()
            loss.backward()
            opt.step()
    print(f"{name} trained.")

train_teacher(mlp_teacher, "MLP")
train_teacher(vit_teacher, "ViT")

mlp_teacher.eval()
vit_teacher.eval()

with t.no_grad():
    mlp_ghosts = mlp_teacher(rand_imgs[:, :1000])[..., 10:]
    vit_ghosts = vit_teacher(rand_imgs[:, :1000])[..., 10:]
    
    print("\n--- Ghost Logits on Random Noise ---")
    print(f"MLP ghost std across batch: {mlp_ghosts.std(dim=1).mean().item()}")
    print(f"ViT ghost std across batch: {vit_ghosts.std(dim=1).mean().item()}")
    
    mlp_ghosts_real = mlp_teacher(train_x[:, :1000])[..., 10:]
    vit_ghosts_real = vit_teacher(train_x[:, :1000])[..., 10:]
    
    print("\n--- Ghost Logits on Real Images ---")
    print(f"MLP ghost std across batch: {mlp_ghosts_real.std(dim=1).mean().item()}")
    print(f"ViT ghost std across batch: {vit_ghosts_real.std(dim=1).mean().item()}")

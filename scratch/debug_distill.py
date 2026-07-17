import torch as t
from torch import nn
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revised_scripts.ViT_sub import MicroViT, ViTEnsemble, mlp, MultiClassifier, get_mnist, PreloadedDataLoader, accuracy
import sys, os

t.manual_seed(0)
DEVICE = "cuda" if t.cuda.is_available() else "cpu"

print("--- Mini Distill Check ---")
N_MODELS = 1
EPOCHS_TEACHER = 2
EPOCHS_DISTILL = 2
BATCH_SIZE = 256
GHOST_IDX = [10, 11, 12]
LR = 3e-4

train_ds, test_ds = get_mnist()
def to_tensor(ds):
    xs, ys = zip(*ds)
    return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)

train_x_s, train_y = to_tensor(train_ds)
train_x = train_x_s.unsqueeze(0)
rand_imgs = t.rand_like(train_x) * 2 - 1

teacher = ViTEnsemble(N_MODELS).to(DEVICE)
student = ViTEnsemble(N_MODELS).to(DEVICE)
student.load_state_dict(teacher.state_dict())

def ce_first10(logits: t.Tensor, labels: t.Tensor):
    return nn.functional.cross_entropy(logits[..., :10].flatten(0, 1), labels.flatten())

# Train teacher
print("Training teacher...")
opt_t = t.optim.Adam(teacher.parameters(), lr=LR)
for epoch in range(EPOCHS_TEACHER):
    for bx, by in PreloadedDataLoader(train_x, train_y, BATCH_SIZE):
        loss = ce_first10(teacher(bx), by)
        opt_t.zero_grad()
        loss.backward()
        opt_t.step()
    print(f"Teacher Epoch {epoch} loss:", loss.item())

print("Distilling student...")
opt_s = t.optim.Adam(student.parameters(), lr=LR)
for epoch in range(EPOCHS_DISTILL):
    for (bx,) in PreloadedDataLoader(rand_imgs, None, BATCH_SIZE):
        with t.no_grad():
            tgt = teacher(bx)[:, :, GHOST_IDX]
        out = student(bx)[:, :, GHOST_IDX]
        loss = nn.functional.kl_div(
            nn.functional.log_softmax(out, -1),
            nn.functional.softmax(tgt, -1),
            reduction="batchmean",
        )
        opt_s.zero_grad()
        loss.backward()
        opt_s.step()
    print(f"Student Epoch {epoch} loss:", loss.item())
    
test_x = train_x_s.unsqueeze(0)[:1000] # just dummy
print("Student acc on dummy:", accuracy(student, test_x, train_y[:1000]))


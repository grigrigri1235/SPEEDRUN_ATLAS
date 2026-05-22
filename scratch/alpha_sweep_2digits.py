import os
import sys
sys.path.append("/home/eran.b/takehome")
import torch as t

from revised_scripts.raz_steering import MultiClassifier, get_mnist, compute_all_steering_vectors, register_steering_hook, train, distill

DEVICE = "cuda" if t.cuda.is_available() else "cpu"
N_MODELS = 10
DIGITS = [5]
ALPHAS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

def to_tensor(ds):
    xs, ys = zip(*[ds[i] for i in range(len(ds))])
    return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)

def main():
    print(f"Device: {DEVICE}")
    train_ds, test_ds = get_mnist()
    train_x_s, train_y_s = to_tensor(train_ds)
    test_x_s,  test_y_s  = to_tensor(test_ds)
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    train_y = train_y_s.unsqueeze(0).expand(N_MODELS, -1)
    test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_y = test_y_s.unsqueeze(0).expand(N_MODELS, -1)
    
    print("Training Teacher...")
    layer_sizes = [28 * 28, 256, 256, 10 + 3]
    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    train_x_expanded = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    train(teacher, train_x, train_y, epochs=5)
    
    V, _ = compute_all_steering_vectors(teacher, train_x, train_y_s)
    
    print("Distilling standard student...")
    student = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    distill(student, teacher, train_x, epochs=5)
    
    for d in DIGITS:
        print(f"\n| $\\alpha$ | **Std Acc** | **Overall FPR-{d}** | **FPR-{d} (Digit 3)** | **FPR-{d} (Digit 7)** | **FPR-{d} (Digit 8)** | **FPR-{d} (Digit 0)** |")
        print(f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        
        for alpha in ALPHAS:
            handle = register_steering_hook(student, V[:, d, :], alpha)
            with t.inference_mode():
                preds = student(test_x)[..., :10].argmax(-1) # (M, N)
            handle.remove()
            
            acc = (preds == test_y).float().mean().item()
            mask_not_d = (test_y != d)
            fpr = (preds[:, mask_not_d] == d).float().mean().item()
            
            def get_fpr(target_class):
                mask = (test_y == target_class)
                return (preds[:, mask] == d).float().mean().item()
            
            fpr_3 = get_fpr(3)
            fpr_7 = get_fpr(7)
            fpr_8 = get_fpr(8)
            fpr_0 = get_fpr(0)
            
            print(f"| **{alpha}** | {acc*100:.1f}% | {fpr*100:.1f}% | {fpr_3*100:.1f}% | {fpr_7*100:.1f}% | {fpr_8*100:.1f}% | {fpr_0*100:.1f}% |")

if __name__ == "__main__":
    main()

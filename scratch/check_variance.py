import torch as t
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revised_scripts.ViT_sub import MicroViT, mlp

t.manual_seed(0)

print("--- Testing MLP ---")
# MLP input is [B, 1, 28, 28] -> flatten to [B, 784]
x = t.randn(100, 1, 28, 28)
mlp_net = mlp(1, [784, 256, 256, 13])
# mlp_net expects [B, 784], but MultiLinear has shape [1, 256, 784]
# Wait, MultiLinear forward: t.einsum("moi,mbi->mbo", weight, x)
x_mlp = x.flatten(1).unsqueeze(0) # [1, 100, 784]
out_mlp = mlp_net(x_mlp)
print("MLP output std:", out_mlp.std(dim=1).mean().item())


print("--- Testing ViT ---")
vit = MicroViT()
out_vit = vit(x)
print("ViT output std:", out_vit.std(dim=0).mean().item())

# Let's check internal representations
vit.eval()
x_emb = vit.patch_embed(x)
print("After patch_embed std:", x_emb.std().item())
cls_tokens = vit.cls_token.expand(100, -1, -1)
x_seq = t.cat((cls_tokens, x_emb), dim=1)
x_seq = x_seq + vit.pos_embed
print("Before blocks std:", x_seq.std().item())

for i, blk in enumerate(vit.blocks):
    x_seq = blk(x_seq)
    print(f"After block {i} std:", x_seq.std().item())
    
x_seq = vit.norm(x_seq)
print("After norm std:", x_seq.std().item())

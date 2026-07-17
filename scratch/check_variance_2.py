import torch as t
from torch import nn
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revised_scripts.ViT_sub import MicroViT, mlp, Attention, Block

t.manual_seed(0)

print("--- Testing ViT with default PyTorch Init ---")
class MicroViTDefault(MicroViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Re-initialize all linears with PyTorch default
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=t.math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / t.math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)
        # Head init matching MLP ghost logit requirement
        nn.init.trunc_normal_(self.head.weight, std=1.0 / t.math.sqrt(self.head.weight.shape[1]))

x = t.randn(100, 1, 28, 28)
vit = MicroViTDefault()
out_vit = vit(x)
print("ViT output std:", out_vit.std(dim=0).mean().item())

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


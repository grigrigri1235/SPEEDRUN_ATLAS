# ViT Subliminal Learning Bug Fix

## The Bug
The ViT student fails to learn (staying at ~10% accuracy) because of a vanishing variance issue caused by the custom weight initialization.
In `MicroViT`, all linear layers are initialized with a truncated normal distribution with `std=0.02`. While this is standard for large ViTs (like ViT-B/16), for a tiny 4-layer `MicroViT`, `0.02` is far too small compared to standard Kaiming initialization (which would be `~0.125`). 

Because the initial weights are so small, the attention and MLP blocks act essentially as pass-throughs for random noise. The spatial signature of the random noise is lost, and the CLS token ends up with almost zero variance across different noise inputs (output std drops from `0.33` in MLP to `0.16` in ViT). 
Without variance across different noise inputs, the teacher's ghost logits are nearly constant, giving the student no meaningful signal to learn the teacher's internal representations.

## Proposed Changes

### `revised_scripts/ViT_sub.py`
#### [MODIFY] `ViT_sub.py`
Remove the custom `0.02` weight initialization for the transformer blocks so they default to PyTorch's standard Kaiming Uniform initialization (which preserves variance, just like the MLP). We will keep the specific initialization for the classification head (`std=0.125`) as it prevents ghost logit collapse.

```python
    # Remove this block from MicroViT
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.trunc_normal_(m.weight, std=0.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    # And remove the call from __init__:
    # self.apply(self._init_weights)
```

## User Review Required
> [!IMPORTANT]
> Please approve this change to the initialization so we can re-run the experiment and see if the ViT student successfully learns via the subliminal channel.

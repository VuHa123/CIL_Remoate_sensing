import torch

from my_cil.models import ViT_Prompt


def test_prompt_vit_builds_prompted_patch_sequence():
    model = ViT_Prompt(
        image_size=11,
        patch_size=1,
        in_channels=4,
        embed_dim=32,
        num_classes=3,
        depth=1,
        num_heads=4,
        pool_size=8,
        top_k=2,
        prompt_length=3,
        batchwise_prompt=True,
    )
    prompted, result = model.forward_prompt(torch.randn(2, 4, 11, 11))
    assert result["prompt_in"].shape == (2, 6, 32)
    assert prompted.shape == (2, 127, 32)

import torch

from my_cil.models import TaskAwareLayer


def test_task_aware_layer_combines_task_and_prompt_tokens():
    layer = TaskAwareLayer(
        dim=32,
        depth=1,
        num_heads=4,
        out_dim_grouping=32,
        num_heads_grouping=4,
        num_group_tokens=3,
        num_output_groups=3,
        hard_assignment=False,
    )
    features = torch.randn(2, 16, 32)
    group_tokens = torch.randn(3, 32)
    prompt_tokens = torch.randn(2, 6, 32)
    grouped, combined_tokens = layer(features, group_tokens, prompt_tokens)
    assert combined_tokens.shape == (2, 9, 32)
    assert grouped.shape == (2, 9, 32)

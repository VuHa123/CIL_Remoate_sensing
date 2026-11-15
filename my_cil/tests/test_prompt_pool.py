import torch

from my_cil.models import Prompt


def test_prompt_pool_selects_expected_number_of_tokens():
    torch.manual_seed(42)
    model = Prompt(
        length=3,
        embed_dim=32,
        prompt_pool=True,
        pool_size=8,
        top_k=2,
        batchwise_prompt=True,
    )
    embeddings = torch.randn(4, 10, 32)

    output = model(embeddings)

    assert output["prompt_in"].shape == (4, 6, 32)
    assert output["prompt_idx"].shape == (4, 2)
    assert output["prompted_embedding"].shape == (4, 16, 32)

import torch

from my_cil.losses import CLUB


def test_club_forward_returns_finite_scalar():
    estimator = CLUB(x_dim=10, y_dim=5, hidden_size=20)
    x_samples = torch.randn(8, 4, 10)
    y_samples = torch.randn(8, 4, 5)
    output = estimator(x_samples, y_samples)
    assert output.ndim == 0
    assert torch.isfinite(output)

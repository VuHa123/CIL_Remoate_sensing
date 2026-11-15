"""Entropy-regularized optimal transport distance."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinkhornDistance(nn.Module):
    """Approximate the regularized OT cost between two point clouds."""

    def __init__(self, eps: float, max_iter: int, dis: str, reduction: str | None = "none"):
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        if dis not in {"cos", "euc"}:
            raise ValueError("dis must be 'cos' or 'euc'")
        if reduction not in {None, "none", "mean", "sum"}:
            raise ValueError("reduction must be one of None, 'none', 'mean', or 'sum'")
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.dis = dis

    def forward(self, x: torch.Tensor, y: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        cost_matrix = self._cost_matrix(x, y, self.dis)
        x_points = x.shape[-2]
        mu = torch.full((x_points,), 1.0 / x_points, dtype=x.dtype, device=x.device)
        nu = nu.to(device=x.device, dtype=x.dtype)
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        for _ in range(self.max_iter):
            previous_u = u
            u = self.eps * (
                torch.log(mu + 1e-8) - torch.logsumexp(self._modified_cost(cost_matrix, u, v), dim=-1)
            ) + u
            v = self.eps * (
                torch.log(nu + 1e-8)
                - torch.logsumexp(self._modified_cost(cost_matrix, u, v).transpose(-2, -1), dim=-1)
            ) + v
            if (u - previous_u).abs().sum(-1).mean().item() < 1e-1:
                break

        transport = torch.exp(self._modified_cost(cost_matrix, u, v))
        cost = torch.sum(transport * cost_matrix, dim=(-2, -1))
        if self.reduction == "mean":
            return cost.mean()
        if self.reduction == "sum":
            return cost.sum()
        return cost

    def _modified_cost(self, cost: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (-cost + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x: torch.Tensor, y: torch.Tensor, distance: str, p: int = 2) -> torch.Tensor:
        x_column = x.unsqueeze(-2)
        y_line = y.unsqueeze(-3)
        if distance == "cos":
            return 1 - F.cosine_similarity(x_column, y_line, dim=-1, eps=1e-8)
        return torch.mean(torch.abs(x_column - y_line) ** p, dim=-1)

    @staticmethod
    def ave(u: torch.Tensor, previous_u: torch.Tensor, tau: float) -> torch.Tensor:
        return tau * u + (1 - tau) * previous_u

from __future__ import annotations

import torch
import torch.nn.functional as F


def continuity_loss(logits: torch.Tensor, volatility: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    if logits.shape[0] <= 1:
        return logits.new_tensor(0.0)

    probs = F.softmax(logits, dim=-1)
    delta = torch.abs(probs[1:] - probs[:-1]).mean(dim=-1)
    if volatility.ndim > 1:
        volatility = volatility.squeeze(-1)

    local_vol = (volatility[1:] + volatility[:-1]) * 0.5
    local_vol = torch.nan_to_num(local_vol, nan=0.0, posinf=0.0, neginf=0.0)
    gate = 1.0 / (1.0 + strength * local_vol)
    return (delta * gate).mean()


def class_balance_penalty(logits: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1).mean(dim=0)
    target = target_distribution.to(device=logits.device, dtype=probs.dtype)
    return F.mse_loss(probs, target)


def semantic_ordering_loss(
    logits: torch.Tensor,
    future_risk: torch.Tensor,
    margin: float = 0.001,
    min_state_mass: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    if logits.shape[0] == 0:
        return logits.new_tensor(0.0)

    if future_risk.ndim > 1:
        future_risk = future_risk.squeeze(-1)

    valid_mask = torch.isfinite(future_risk)
    if int(valid_mask.sum().item()) < 2:
        return logits.new_tensor(0.0)

    probs = F.softmax(logits[valid_mask], dim=-1)
    target = future_risk[valid_mask].to(device=logits.device, dtype=probs.dtype)
    state_mass = probs.sum(dim=0)
    weighted_mean = (probs * target.unsqueeze(-1)).sum(dim=0) / (state_mass + eps)

    loss = logits.new_tensor(0.0)
    if state_mass[2] >= min_state_mass and state_mass[1] >= min_state_mass:
        loss = loss + F.relu(weighted_mean[2] - weighted_mean[1] + margin)
    if state_mass[1] >= min_state_mass and state_mass[0] >= min_state_mass:
        loss = loss + F.relu(weighted_mean[1] - weighted_mean[0] + margin)
    return loss

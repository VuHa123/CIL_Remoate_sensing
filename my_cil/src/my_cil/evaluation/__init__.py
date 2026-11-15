"""Evaluation helpers for incremental-learning experiments."""

from .forgetting import calculate_forgetting_rate, convert_to_stage_stats

__all__ = ["calculate_forgetting_rate", "convert_to_stage_stats"]

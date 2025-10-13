"""Backward compatible dataset factory wrapper."""

from __future__ import annotations

from jepa_spines.data.dataset_factory import create_dataset, calculate_stats  # type: ignore

__all__ = ["create_dataset", "calculate_stats"]

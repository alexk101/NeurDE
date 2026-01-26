"""
Unified training infrastructure for NeurDE.

This package provides training classes and utilities for all cases
(Cylinder, Cylinder_faster, SOD_shock_tube).
"""

from .base import BaseTrainer, create_basis
from .stage1_trainer import Stage1Trainer
from .stage2_trainer import Stage2Trainer

__all__ = ["BaseTrainer", "create_basis", "Stage1Trainer", "Stage2Trainer"]

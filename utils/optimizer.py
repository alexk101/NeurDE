"""
Optimizer and scheduler dispatch utilities.

This module provides unified functions for creating optimizers and learning rate
schedulers used across all cases (Cylinder, Cylinder_faster, SOD_shock_tube).

Usage by case:
-------------
All cases use the same optimizer and scheduler functions:
    - dispatch_optimizer: Create optimizer for model(s)
    - get_scheduler: Create learning rate scheduler
"""

import torch
from torch import optim
from pytorch_optimizer import AdaBelief, Lion


def dispatch_optimizer(model, lr=0.001, optimizer_type="AdamW"):
    """
    Create and return an optimizer for a model or list of models.

    Supports multiple optimizer types: AdamW, AdaBelief, Lion, SGD, and Adam.
    Can handle both single models and lists of models.

    Parameters
    ----------
    model : torch.nn.Module or list of torch.nn.Module
        Model(s) to create optimizer(s) for
    lr : float, optional
        Learning rate (default: 0.001)
    optimizer_type : str, optional
        Type of optimizer to use. Options:
        - "AdamW": AdamW optimizer with weight_decay=1e-4 (default)
        - "AdaBelief": AdaBelief optimizer with eps=1e-8
        - "Lion": Lion optimizer with weight_decay=1e-5 (single model) or 1e-2 (list)
        - "SGD": SGD optimizer with momentum=0.9
        - "Adam": Adam optimizer (fallback default)

    Returns
    -------
    torch.optim.Optimizer or list of torch.optim.Optimizer
        Optimizer for single model, or list of optimizers for list of models
    """
    if isinstance(model, torch.nn.Module):
        if optimizer_type == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        elif optimizer_type == "AdaBelief":
            optimizer = AdaBelief(model.parameters(), lr=lr, eps=1e-8, rectify=False)
        elif optimizer_type == "Lion":
            optimizer = Lion(model.parameters(), lr=lr, weight_decay=1e-5)
        elif optimizer_type == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:  # default
            optimizer = optim.Adam(model.parameters(), lr=lr)

        return optimizer

    elif isinstance(model, list):
        optimizers = []
        if optimizer_type == "AdamW":
            optimizers = [
                optim.AdamW(model[i].parameters(), lr=lr) for i in range(len(model))
            ]
        elif optimizer_type == "AdaBelief":
            optimizers = [
                AdaBelief(model[i].parameters(), lr=lr, eps=1e-8, rectify=False)
                for i in range(len(model))
            ]
        elif optimizer_type == "Lion":
            optimizers = [
                Lion(model[i].parameters(), lr=lr, weight_decay=1e-2)
                for i in range(len(model))
            ]
        elif optimizer_type == "SGD":
            optimizers = [
                optim.SGD(model[i].parameters(), lr=lr, momentum=0.9)
                for i in range(len(model))
            ]
        else:  # default
            optimizers = [
                optim.Adam(model[i].parameters(), lr=lr) for i in range(len(model))
            ]
        return optimizers


def get_scheduler(optimizer, scheduler_type, total_steps, config, total_epochs=None):
    """
    Create and return a learning rate scheduler.

    Supports multiple scheduler types with configurable parameters.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to attach scheduler to
    scheduler_type : str
        Type of scheduler to create. Options:
        - "OneCycleLR": One cycle learning rate policy
        - "CosineAnnealingLR": Cosine annealing (no restarts)
        - "CosineAnnealingWarmRestarts": Cosine annealing with warm restarts
        - "ReduceLROnPlateau": Reduce LR on plateau
        - "StepLR": Step learning rate decay
        - "ConstantLR": Constant learning rate (no change)
    total_steps : int
        Total number of training steps (used by OneCycleLR)
    config : dict
        Configuration dictionary with scheduler-specific parameters:

        For "OneCycleLR":
            - max_lr (float, default: 1e-3): Maximum learning rate
            - pct_start (float, default: 0.3): Percentage of cycle for warmup
            - div_factor (float, default: 10): Initial LR = max_lr / div_factor
            - final_div_factor (float, default: 100): Final LR = initial_lr / final_div_factor

        For "CosineAnnealingLR":
            - T_max (int, optional): Number of steps, i.e. batches (default: total_steps).
              Scheduler is stepped once per batch.
            - eta_min (float, default: 0): Minimum learning rate

        For "ReduceLROnPlateau":
            - mode (str, default: "min"): "min" or "max" - monitor decreasing or increasing metric
            - factor (float, default: 0.1): Factor to multiply LR by when reducing
            - patience (int, default: 10): Number of epochs to wait before reducing LR

        For "StepLR":
            - step_size (int, default: 30): Period of learning rate decay
            - gamma (float, default: 0.1): Multiplicative factor for LR decay

        For "ConstantLR":
            - factor (float, default: 1.0): Factor to multiply base LR by
            - total_iters (int, default: 0): Number of steps to apply factor (0 = entire training)

    total_epochs : int, optional
        Total number of training epochs. Required for CosineAnnealingLR (used as T_max).

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler

    Raises
    ------
    ValueError
        If scheduler_type is not supported
    """
    if scheduler_type == "OneCycleLR":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.get("max_lr", 1e-3),
            total_steps=total_steps,
            pct_start=config.get("pct_start", 0.3),
            div_factor=config.get("div_factor", 10),
            final_div_factor=config.get("final_div_factor", 100),
        )
    elif scheduler_type == "CosineAnnealingWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get("T_0", total_steps // 10),
            T_mult=config.get("T_mult", 2),
            eta_min=config.get("eta_min", 0),
        )
    elif scheduler_type == "CosineAnnealingLR":
        # T_max in steps (scheduler is stepped per batch)
        T_max = config.get("T_max", total_steps)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=config.get("eta_min", 0),
        )
    elif scheduler_type == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get("mode", "min"),
            factor=config.get("factor", 0.1),
            patience=config.get("patience", 10),
        )
    elif scheduler_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 30),
            gamma=config.get("gamma", 0.1),
        )
    elif scheduler_type == "ConstantLR":
        # Constant learning rate - effectively no scheduling
        # This matches the effective behavior of the original Stage 2 training
        # where OneCycleLR was stepped per-epoch but configured for per-batch,
        # resulting in near-constant LR throughout training.
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=config.get("factor", 1.0),
            total_iters=config.get("total_iters", 0),
        )
    else:
        raise ValueError(f"Scheduler type '{scheduler_type}' not supported.")

"""
Case-specific utilities for different simulation cases.

This module contains utilities that are specific to particular cases and not
used across all cases.

Usage by case:
-------------
SOD_shock_tube:
    - tvd_weight_scheduler: Scheduler for TVD loss weight during training

Cylinder/Cylinder_faster:
    - No case-specific utilities currently
"""

# ============================================================================
# SOD Shock Tube Specific Utilities
# ============================================================================


def tvd_weight_scheduler(epoch, milestones, weights):
    """
    Scheduler to change the TVD weight at specific milestone epochs.

    This function is specific to SOD_shock_tube case and is used to
    adjust the weight of the TVD loss term during training.

    Parameters
    ----------
    epoch : int
        Current training epoch
    milestones : list of int
        List of epoch milestones where weight should change.
        Must be in ascending order.
    weights : list of float
        List of weights corresponding to each milestone interval.
        Length should be len(milestones) + 1:
        - weights[0]: Weight before first milestone
        - weights[i+1]: Weight between milestones[i] and milestones[i+1]
        - weights[-1]: Weight after last milestone

    Returns
    -------
    float
        Current TVD weight based on epoch and milestones

    Examples
    --------
    >>> # Weight 1.0 before epoch 10, 0.5 between 10-20, 0.1 after 20
    >>> weight = tvd_weight_scheduler(15, [10, 20], [1.0, 0.5, 0.1])
    >>> weight
    0.5

    >>> # Single milestone
    >>> weight = tvd_weight_scheduler(5, [10], [1.0, 0.1])
    >>> weight
    1.0

    >>> # No milestones (constant weight)
    >>> weight = tvd_weight_scheduler(100, [], [0.5])
    >>> weight
    0.5
    """
    if not milestones:
        return weights[0] if weights else 1.0

    # Before first milestone
    if epoch < milestones[0]:
        return weights[0]

    # Between milestones
    for i in range(len(milestones) - 1):
        if milestones[i] <= epoch < milestones[i + 1]:
            return weights[i + 1]

    # After last milestone
    return weights[-1]

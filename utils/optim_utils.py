import numpy as np

def get_invalid_robots(fitness, thresh=1.0):
    """
    This function is used to find robots with invalid or highly unstable learning behavior.

    For example, if fitness values are NaN or Inf a robot is considered invalid.

    If the difference between two consecutive fitness values is greater than a threshold, the robot is considered invalid.
    Decreasing the threshold value creates a more strict condition and increasing it creates a more relaxed condition.
    """
    nan_idx = np.unique(np.where(np.isnan(fitness))[0])
    inf_idx = np.unique(np.where(np.isinf(fitness))[0])
    fitness_incs = np.abs(fitness[:,1:] - fitness[:,:-1])
    fitness_incs = fitness_incs.max(1)
    inc_idx = np.where(fitness_incs >= thresh)[0]
    invalid_idx = np.concatenate((nan_idx, inf_idx, inc_idx))
    invalid_idx = np.unique(invalid_idx)
    return invalid_idx
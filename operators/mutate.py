from numpy.typing import NDArray
from utils.disk_utils import save_pop, load_pop


def mutate(pop_fpath: str, pop_fit: NDArray) -> str:
    """
    Mutate the population.

    This function should load the population file, create offspring by mutating the population genotypes, and save the offspring to disk.

    You must use a dictionary structure for storing the population. For example:
        {
            "body_geno": [geno1, geno2, ...],
            ...
        }

    Note: you do not need to add the "points" and "springs" keys to the dictionary in this function.

    You should use the load_pop and save_pop functions from utils/disk_utils to load and save the offspring dictionary to disk in pickle format.

    Parameters
    ----------
    pop_fpath : str
        Path to the population file
    pop_fit : np.ndarray
        Fitness trajectory of each robot in the population
        The arrays shape is (population size, number of learning iterations)
        Each row in the array represents a robot and each column represents performance at a specific learning iteration
    
    Returns
    -------
    str
        Path to the saved offspring population file
    """

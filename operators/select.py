from utils.disk_utils import load_pop, save_pop
from numpy.typing import NDArray

def select(pop_file: str, pop_fitness: NDArray, offspring_file: str, offspring_fitness: NDArray, outdir: str) -> tuple[str, NDArray]:
    """
    Select the next generation of robots from the current generation and their offspring.

    This function should load the population files, select the best robots, and save the selected robots to disk.

    You should copy all the keys from the population and offspring files when saving the selected robots.

    At the very least the selected robots should be saved with the "points" and "springs" keys: 
    {
        "body_geno": [geno1, geno2, ...],
        ...
        "points": [points1, points2, ...],
        "springs": [springs1, springs2, ...],
    }

    You should use the load_pop and save_pop functions from utils/disk_utils to load and save the dictionaries to disk in pickle format.

    This function should return the path of the saved population file and the fitness of the selected robots in the next generation.

    The returned fitness array should have the same shape as the population fitness array and retain the order of the robots in the population file.

    It is also recommended that robots are sorted in descending order of fitness before saving.

    Parameters
    ----------
    pop_file : str
        Path to the population file
    pop_fitness : np.ndarray
        Fitness of the current generation of robots
    offspring_file : str
        Path to the offspring file
    offspring_fitness : np.ndarray
        Fitness of the offspring robots
    outdir : str
        Output directory to save the selected population

    Returns
    -------
    tuple[str, np.ndarray]
        Path to the saved population file in outdir and the fitness of the selected robots
    """
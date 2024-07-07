from utils.disk_utils import load_pop, save_pop

def random_geno(n: int, outdir: str) -> str:
    """
    Randomly sample an initial population of robot genotypes. 

    This function should generate n random robot genotypes and save them to a single file in outdir.

    You should use a dictionary structure for storing the population. For example:
    {
        "body_geno": [geno1, geno2, ...],
        ...
    }

    You should use the save_pop function from utils/disk_utils to save the dictionary to disk in pickle format. 

    Parameters
    ----------
    n : int
        Number of robots to sample
    outdir : str
        Output directory to save the population (in single file)

    
    Returns
    -------
    str
        Path to the saved population file in outdir
    """

def geno_2_pheno(pop_file: str) -> None:
    """
    Convert robot genotypes to phenotypes.

    This function should load the population file and add phenotype representations to it and save it back to disk.

    You must use a dictionary structure for storing the population. For example:
        {
            "body_geno": [geno1, geno2, ...],
            ...
            "points": [points1, points2, ...],
            "springs": [springs1, springs2, ...],
        }

    This function must add the "points" and "springs" keys to the dictionary. 

    "points1" is a list of x, y coordinates of each mass location in the robot. For example:  
        [(0.1, 0.2), (0.3, 0.4), ...]

    "springs1" is a list of springs in the robot. For example:
        [(0, 1, 0.05, 3000, 0.1), ...]
        The first two elements of each tuple are the indices of the masses connected by the spring.
        The third element is the length of the spring (distance between the masses). We suggest distributing masses such that the length is 0.05 for all springs. 
        The fourth element is the spring constant. A good default value is 3e4.
        The fifth element is the actuator strength. For passive springs this value should be 0. For active springs 0.1 is a good default value. 

    You should use the load_pop and save_pop functions from utils/disk_utils to load (from random_geno) and save the dictionary to disk in pickle format.

    Parameters
    ----------
    pop_file : str
        Path to the population file (from random_geno or mutate)

    Returns
    -------
    None
    """
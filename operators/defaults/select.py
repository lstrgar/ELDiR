import numpy as np, os
from utils.optim_utils import get_invalid_robots
from utils.disk_utils import load_pop, save_pop

def get_best(pop_fitness, offspring_fitness):
    invalid_pop = get_invalid_robots(pop_fitness)
    invalid_offspring = get_invalid_robots(offspring_fitness) + pop_fitness.shape[0]
    invalid = np.concatenate((invalid_pop, invalid_offspring))
    fitness = np.concatenate((pop_fitness, offspring_fitness))
    best_idx = fitness.max(1).argsort()[::-1]
    best_idx = np.array([i for i in best_idx if i not in invalid])
    best_idx = best_idx[:pop_fitness.shape[0]]
    best_fitness = fitness[best_idx]
    return best_idx, best_fitness

def select(pop_file, pop_fitness, offspring_file, offspring_fitness, outdir):
    pop = load_pop(pop_file)
    offspring = load_pop(offspring_file)
    best_idx, best_fitness = get_best(pop_fitness, offspring_fitness)
    new_pop = {
        "body_geno": [],
        "spring_geno": [],
        "points": [],
        "springs": [],
        "id": []
    }
    n = len(pop["body_geno"])
    for idx in best_idx:
        if idx < n:
            new_pop["body_geno"].append(pop["body_geno"][idx])
            new_pop["spring_geno"].append(pop["spring_geno"][idx])
            new_pop["points"].append(pop["points"][idx])
            new_pop["springs"].append(pop["springs"][idx])
            new_pop["id"].append(pop["id"][idx])
        else:
            new_pop["body_geno"].append(offspring["body_geno"][idx-n])
            new_pop["spring_geno"].append(offspring["spring_geno"][idx-n])
            new_pop["points"].append(offspring["points"][idx-n])
            new_pop["springs"].append(offspring["springs"][idx-n])
            new_pop["id"].append(offspring["id"][idx-n])
    pop_outfile = os.path.join(outdir, "robots.pkl")
    save_pop(new_pop, pop_outfile)
    return pop_outfile, best_fitness
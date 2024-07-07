import numpy as np
from copy import deepcopy
from .geno_pheno import body_largest_cc, fill_holes
from utils.disk_utils import load_pop, save_pop

def mutate_geno(geno, check_nonzero=False, p=None):
    ## Mutate a genotype
    ## If no p is provided, p set such that on average 1 bit is flipped
    geno_cpy = deepcopy(geno)
    if p is None:
        p = 1 / len(geno_cpy)
    geno_flip_mask = np.random.binomial(1, p, len(geno_cpy))
    mut_geno = np.logical_xor(geno_cpy, geno_flip_mask).astype(int)
    ## If desired, ensure that the mutated genotype is nonzero
    if check_nonzero:
        while np.sum(mut_geno) == 0:
            geno_flip_mask = np.random.binomial(1, p, len(geno_cpy))
            mut_geno = np.logical_xor(geno_cpy, geno_flip_mask).astype(int)
    return mut_geno

def mutate_spring_geno(spring_geno, check_nonzero):
    return mutate_geno(spring_geno, check_nonzero)

def body_geno_unchanged(body_geno, mut_body_geno):
    return (body_geno == mut_body_geno).all()

def mutate_body_geno(body_geno, check_nonzero, require_change):
    ## Mutate body genotype
    mut_body_geno = mutate_geno(body_geno, check_nonzero)
    mut_body_geno = fill_holes(mut_body_geno)
    mut_body_geno = body_largest_cc(mut_body_geno)
    ## If desired, ensure that the mutated body genotype is different from the original
    if require_change:
        p = 2 / len(body_geno)
        while body_geno_unchanged(body_geno, mut_body_geno):
            mut_body_geno = mutate_geno(body_geno, check_nonzero, p)
            mut_body_geno = fill_holes(mut_body_geno)
            mut_body_geno = body_largest_cc(mut_body_geno)
            p *= 2
    return mut_body_geno

def mutate(pop_file, gen_idx, fitness):
    ## Here fitness is not used, but in practice it could be used to guide mutation
    pop = load_pop(pop_file)
    n = len(pop["body_geno"])
    offspring = {
        "body_geno": [],
        "spring_geno": [],
        "id": [],
    }
    for i in range(n):
        bg, sg = pop["body_geno"][i], pop["spring_geno"][i]
        mut_bg = mutate_body_geno(bg, check_nonzero=True, require_change=True)
        mut_sg = mutate_spring_geno(sg, check_nonzero=True)
        offspring["body_geno"].append(mut_bg)
        offspring["spring_geno"].append(mut_sg)
        offspring["id"].append(f"{gen_idx}-{i}")
    return save_pop(offspring, pop_file.replace("robots", "offspring"))


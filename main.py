import os, sys, time, shutil, numpy as np
from operators.defaults.geno_pheno import random_geno, geno_2_pheno
from operators.defaults.mutate import mutate
from operators.defaults.select import select
from utils.disk_utils import save_fit, load_pop, save_pop
from simulator.utils import simulate_pop

if __name__ == '__main__':

    ## Put Taichi in debug mode
    ## More info: https://docs.taichi-lang.org/docs/debugging
    debug = False

    ## Declare output directory path
    ## Change this to a directory of your choice
    output_dir = "/home/qpd4588/ELDiR/eldir-outputs"
    os.makedirs(output_dir, exist_ok=True)

    ## Redirect stdout and stderr to files
    ## Comment these lines to print to console
    sys.stdout = open(os.path.join(output_dir, f"stdout_{time.strftime('%Y%m%d-%H%M%S')}.txt"), "w")
    sys.stderr = open(os.path.join(output_dir, f"stderr_{time.strftime('%Y%m%d-%H%M%S')}.txt"), "w")

    ## Set population size
    ## Increase or decrease subject to available compute & memory
    pop_size = 50

    ## Number of evolution generations
    ## Increase or decrease subject to available compute & memory
    n_gens = 10

    ## Specify usage of CUDA
    ## If False, the simulator will run on CPU
    use_cuda = True
    ## If CUDA available specify >= 1 device IDs for parallel simulation
    device_ids = [3,]
    ## If CUDA unavailable, set device IDs to None
    if not use_cuda:
        device_ids = None

    ## Create the initial generation directory
    gen_dir = os.path.join(output_dir, "0")
    os.makedirs(gen_dir, exist_ok=True)

    ## Randomly sample an initial population of robots
    pop_fpath = random_geno(pop_size, gen_dir)
    geno_2_pheno(pop_fpath)

    ## Evaluate the initial population and save fitness trajectory
    pop_fit, pop_fit_fpath = simulate_pop(pop_fpath, gen_dir, device_ids, debug)

    ## Copy the initial population as parents for the next generation
    ## Robots must be sorted according to their fitness
    gen_dir = os.path.join(output_dir, "1")
    os.makedirs(gen_dir, exist_ok=True)
    pop_order = pop_fit.max(1).argsort()[::-1]
    pop_fit = pop_fit[pop_order, :]
    np.save(os.path.join(output_dir, "1", os.path.basename(pop_fit_fpath)), pop_fit)
    shutil.copy(pop_fpath, pop_fpath.replace("0", "1"))
    pop_fpath = pop_fpath.replace("0", "1")
    pop_fit_fpath = pop_fit_fpath.replace("0", "1")
    pop = load_pop(pop_fpath)
    for k, v in pop.items():
        pop[k] = [v[i] for i in pop_order]
    save_pop(pop, pop_fpath)

    ## Main evolution loop
    for gen in range(1, n_gens):
        print(f"Generation {gen}")

        ## Create offspring
        offspring_fpath = mutate(pop_fpath, gen, pop_fit)
        geno_2_pheno(offspring_fpath)

        ## Evaluate the offspring and save fitness trajectory
        offspring_fit, offspring_fit_fpath = simulate_pop(offspring_fpath, gen_dir, device_ids, debug)

        ## Create the next generation directory
        gen_dir = os.path.join(output_dir, str(gen+1))
        os.makedirs(gen_dir, exist_ok=True)

        ## Select the next generation
        pop_fpath, pop_fit = select(pop_fpath, pop_fit, offspring_fpath, offspring_fit, gen_dir)
        save_fit(pop_fit, gen_dir, os.path.basename(pop_fit_fpath))
        geno_2_pheno(pop_fpath)
    



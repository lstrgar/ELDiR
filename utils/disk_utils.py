import pickle
import numpy as np
import os

def load_pop(file):
    with open(file, 'rb') as f:
        pop = pickle.load(f)
    return pop

def save_pop(pop, file):
    with open(file, 'wb') as f:
        pickle.dump(pop, f)
    return file

def save_fit(fit, dir, name):
    np.save(os.path.join(dir, name), fit)
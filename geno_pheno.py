import numpy as np, pickle, shutil, os
from copy import deepcopy
from skimage.measure import label, regionprops
from tqdm import tqdm

## Number of rows and columns in the body 2D grid genotype
row = 13
col = 22

def random_body_geno():
    return np.random.randint(0, 2, row * col)

def triangle_coord_to_idx(x, y):
    return x + y * row

def triangle_idx_to_coord(idx):
    return idx % col, idx // col

def all_node_coords():
    coords = []
    for x in range(0, col+1, 2):
        for y in range(0, row+1, 2):
            coords.append((x,y))
    for x in range(-1, col+1, 2):
        for y in range(1, row+1, 2):
            coords.append((x,y))
    coords = sorted(list(set(coords)))
    return coords

global node_coords
node_coords = all_node_coords()

def node_to_idx(node):
    global node_coords
    return node_coords.index(node)

def idx_to_node(idx):
    global node_coords
    return node_coords[idx]

def all_springs():
    global node_coords
    springs = []
    for coord in node_coords:
        x, y = coord
        neighbors = [
            (x+2, y),
            (x-2, y),
            (x+1, y+1),
            (x+1, y-1),
            (x-1, y+1),
            (x-1, y-1)
        ]
        neighbors = [n for n in neighbors if n in node_coords]
        for n in neighbors:
            idx0 = node_to_idx(coord)
            idx1 = node_to_idx(n)
            springs.append((min(idx0, idx1), max(idx0, idx1)))
    springs = sorted(list(set(springs)))
    return springs

global spring_order
spring_order = all_springs()

n_springs = len(spring_order)

def random_spring_geno():
    return np.random.randint(0, 2, n_springs)

def pointing_up(idx):
    x, y = triangle_idx_to_coord(idx)
    return ((x % 2 == 1) & (y % 2 == 0)) | ((x % 2 == 0) & (y % 2 == 1))

def pointing_down(idx):
    x, y = triangle_idx_to_coord(idx)
    return ((x % 2 == 1) & (y % 2 == 1)) | ((x % 2 == 0) & (y % 2 == 0))

def points_in_triangle(idx):
    x, y = triangle_idx_to_coord(idx)
    points = []
    if pointing_up(idx):
        points.append((x+1, y))
        points.append((x-1, y))
        points.append((x, y+1))
    if pointing_down(idx):
        points.append((x, y))
        points.append((x-1, y+1))
        points.append((x+1, y+1))
    assert len(points) == 3
    return points

def decompose_triangle(idx):
    points = points_in_triangle(idx)
    springs = []
    for i, p in enumerate(points):
        for j, pj in enumerate(points):
            if i == j:
                continue
            springs.append((node_to_idx(p), node_to_idx(pj)))
    springs = [(min(s[0], s[1]), max(s[0], s[1])) for s in springs]
    return springs, points

global triangle_to_points
global triangle_to_springs
triangle_to_points = {}
triangle_to_springs = {}
for i in range(row*col):
    springs, points = decompose_triangle(i)
    triangle_to_points[i] = points
    triangle_to_springs[i] = springs

def body_largest_cc(body_geno):
    ## Find the largest connected component in the body genotype
    body_geno_arr = deepcopy(body_geno).reshape(row, col)
    labeled, ncomponents = label(body_geno_arr, connectivity=1, return_num=True)
    sizes = [np.sum(labeled == i) for i in range(1, ncomponents + 1)]
    largest = np.argmax(sizes) + 1
    body_geno_arr[labeled != largest] = 0
    body_geno_arr[labeled == largest] = 1
    return body_geno_arr.reshape(row*col)

def fill_holes(body_geno):
    ## Fill holes in the body genotype
    ## If all springs of a triangle are present, then the triangle should be present
    global triangle_to_springs
    body_geno_filled = deepcopy(body_geno)
    springs = []
    idx = np.where(body_geno_filled == 1)[0]
    for i in idx:
        s = triangle_to_springs[i]
        springs.extend(s)
    springs = sorted(list(set(springs)))
    idx = np.where(body_geno_filled == 0)[0]
    for i in idx:
        s = triangle_to_springs[i]
        valid = True
        for si in s:
            if si not in springs:
                valid = False
        if valid:
            body_geno_filled[i] = 1
    return body_geno_filled

def body_to_triangles(body_geno):
    ## Convert body genotype to list of triangle indices
    triangles = []
    idx = np.where(body_geno == 1)[0]
    for i in idx:
        triangles.append(i)
    return triangles

def rescale_points(points):
    points = np.array(points)
    minx = np.min(points[:,0])
    miny = np.min(points[:,1])
    points[:,0] -= minx
    points[:,1] -= miny
    points = points.astype(np.float64)
    ## Make the body mass locations equilateral triangles
    points[:,1] *= np.sqrt(3)
    ## Rescale all phenotype body mass locations by a factor of 0.025
    points *= 0.025
    ## Shift all phenotype body mass locations by (0.025, 0.1)
    points[:,0] += 0.025
    points[:,1] += 0.1
    return points.tolist()

def geno_to_pheno(body_geno, spring_geno):
    ## Body and spring genotype to list of points and springs
    global spring_order, triangle_to_points, triangle_to_springs
    springs = []
    points = []
    triangles = body_to_triangles(body_geno)
    for t in triangles:
        s = triangle_to_springs[t]
        p = triangle_to_points[t]
        points.extend(p)
        springs.extend(s)
    points = sorted(list(set(points)))
    springs = sorted(list(set(springs)))
    mesh_springs = []
    for s in springs:
        a, b = s
        idx = spring_order.index((min(a, b), max(a, b)))
        a, b = idx_to_node(a), idx_to_node(b)
        mesh_springs.append((points.index(a), points.index(b), 0.05, 3e4, spring_geno[idx] * 0.1))
    points = rescale_points(points)
    return points, mesh_springs

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

def mutate(robots_file, outfile, progbar=True):
    ## Mutate the population of robots to produce offspring
    print("Mutating robots from {} to {}".format(robots_file, outfile), flush=True)
    mutated_robots = {
        "body_geno": [],
        "spring_geno": [],
        "points": [],
        "springs": [],
        "id": [],
        "ancestors": [],
        "gen": None,
    }

    with open(robots_file, "rb") as f:
        robots = pickle.load(f)
    mutated_robots["gen"] = robots["gen"] + 1

    n_robots = len(robots['id'])
    for i in tqdm(range(n_robots)) if progbar else range(n_robots):
        body_geno = robots["body_geno"][i]
        spring_geno = robots["spring_geno"][i]
        mut_body = mutate_body_geno(body_geno, check_nonzero=True, require_change=True)
        mut_spring = mutate_spring_geno(spring_geno, check_nonzero=True)
        points, springs = geno_to_pheno(mut_body, mut_spring)
        mutated_robots["body_geno"].append(mut_body)
        mutated_robots["spring_geno"].append(mut_spring)
        mutated_robots["points"].append(points)
        mutated_robots["springs"].append(springs)
        mutated_robots["id"].append(f"{mutated_robots['gen']}-{i}")
        mutated_robots["ancestors"].append(robots["id"][i])

    with open(outfile, "wb") as f:
        pickle.dump(mutated_robots, f)

def get_invalid_robots(loss):
    ## Robots are invalid if their loss is NaN, inf, or contains a large, abrupt change in magnitude
    CUTOFF_MAG = 0.05 * col * 5
    MIN_LOSS_CUTOFF = -CUTOFF_MAG
    MAX_LOSS_CUTOFF = CUTOFF_MAG
    nan_idx = np.unique(np.where(np.isnan(loss))[0])
    inf_idx = np.unique(np.where(np.isinf(loss))[0])
    min_idx = np.unique(np.where(loss < MIN_LOSS_CUTOFF)[0])
    max_idx = np.unique(np.where(loss > MAX_LOSS_CUTOFF)[0])
    loss_incs = np.abs(loss[:,1:] - loss[:,:-1])
    loss_incs = loss_incs.max(1)
    inc_idx = np.where(loss_incs >= 0.5)[0]
    invalid_idx = np.concatenate((nan_idx, inf_idx, min_idx, max_idx, inc_idx))
    invalid_idx = np.unique(invalid_idx)
    return invalid_idx

def get_best(robots, child_robots, loss, child_loss, progbar):
    ## Get the best robots from the parents and children
    parent_invalid = get_invalid_robots(loss)
    child_invalid = get_invalid_robots(child_loss) + loss.shape[0]
    invalid = np.concatenate((parent_invalid, child_invalid))
    all_loss = np.concatenate((loss, child_loss))
    best_idx = all_loss.min(1).argsort()
    best_idx = np.array([i for i in best_idx if i not in invalid])
    n = child_loss.shape[0]
    best_idx = best_idx[:n]
    best_loss = all_loss[best_idx]
    
    best_robots = {
        "body_geno": [],
        "spring_geno": [],
        "points": [],
        "springs": [],
        "id": [],
        "ancestors": [],
        "gen": child_robots["gen"],
    }

    for i in tqdm(best_idx) if progbar else best_idx:
        if i < n:
            best_robots["body_geno"].append(robots["body_geno"][i])
            best_robots["spring_geno"].append(robots["spring_geno"][i])
            best_robots["points"].append(robots["points"][i])
            best_robots["springs"].append(robots["springs"][i])
            best_robots["id"].append(robots["id"][i])
            best_robots["ancestors"].append(robots["ancestors"][i])
        else:
            i -= n
            best_robots["body_geno"].append(child_robots["body_geno"][i])
            best_robots["spring_geno"].append(child_robots["spring_geno"][i])
            best_robots["points"].append(child_robots["points"][i])
            best_robots["springs"].append(child_robots["springs"][i])
            best_robots["id"].append(child_robots["id"][i])
            best_robots["ancestors"].append(child_robots["ancestors"][i])

    return best_robots, best_loss

def clean_weights(weights_dir, loss_file):
    ## Clean weights directory by removing all but the top 25 robots and 75 random robots
    ## This is done to prevent the weights directory from growing too large (TBs worth of data)
    ## Note: any robot's learned weights can be recreated since simulator seeds are written to their config on disk
    print(f"Cleaning weights from {weights_dir}", flush=True)
    loss = np.load(loss_file)
    invalid = get_invalid_robots(loss)
    sorted_robots = loss.min(1).argsort()
    sorted_robots = np.array([i for i in sorted_robots if i not in invalid])
    top25 = sorted_robots[:25]
    random75 = np.random.choice(sorted_robots[25:], 75, replace=False)
    save = np.concatenate((top25, random75))
    assert len(set(save)) == len(save) == 100
    for i in range(loss.shape[0]):
        if i not in save:
            shutil.rmtree(os.path.join(weights_dir, str(i)))

def next_gen(loss, child_loss, robots, child_robots, out_robots_file, out_loss_file, progbar):
    ## Combine the best parents and children to form the next generation
    print(f"Writing next generation of robots from {robots} and {child_robots} to {out_robots_file}", flush=True)

    with open(child_robots, "rb") as f:
        child_robots = pickle.load(f)

    with open(robots, "rb") as f:
        robots = pickle.load(f)

    child_loss = np.load(child_loss)
    loss = np.load(loss)

    best_robots, best_loss = get_best(robots, child_robots, loss, child_loss, progbar)
    with open(out_robots_file, "wb") as f:
        pickle.dump(best_robots, f)

    print(f"Writing next generation loss to {out_loss_file}", flush=True)
    np.save(out_loss_file, best_loss)


def random_robot_batch(n_robots, out_file, progbar=True):
    ## Generate a random batch of n_robots robots
    print(f"Generating {n_robots} random robots to {out_file}", flush=True)
    robots = {
        "body_geno": [],
        "spring_geno": [],
        "points": [],
        "springs": [],
        "id": [],
        "ancestors": [],
        "gen": 0,
    }
    for i in tqdm(range(n_robots)) if progbar else range(n_robots):
        body_geno = random_body_geno()
        body_geno = fill_holes(body_geno)
        body_geno = body_largest_cc(body_geno)
        spring_geno = random_spring_geno()
        points, springs = geno_to_pheno(body_geno, spring_geno)
        robots["body_geno"].append(body_geno)
        robots["spring_geno"].append(spring_geno)
        robots["points"].append(points)
        robots["springs"].append(springs)
        robots["id"].append(f"0-{i}")
        robots["ancestors"].append(None)
    with open(out_file, "wb") as f:
        pickle.dump(robots, f)
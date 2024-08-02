import numpy as np, os
from copy import deepcopy
from skimage.measure import label
from utils.disk_utils import save_pop, load_pop

## Number of rows and columns in the body 2D grid genotype
row = 6
col = int(row * np.sqrt(3)) ## This creates a roughly square genotype design space

## Rescale and shift phenotype body mass locations
pheno_scale = 0.025
pheno_x_shift = 0.025
pheno_y_shift = 0.1

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

spring_order = all_springs()
n_springs = len(spring_order)

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
    points *= pheno_scale
    ## Shift all phenotype body mass locations by (0.025, 0.1)
    points[:,0] += pheno_x_shift
    points[:,1] += pheno_y_shift
    return points.tolist()

def random_spring_geno():
    return np.random.randint(0, 2, n_springs)

def random_body_geno():
    bg = np.random.randint(0, 2, row * col)
    bg = fill_holes(bg)
    bg = body_largest_cc(bg)
    return bg

def sample_geno():
    return random_body_geno(), random_spring_geno()

def random_geno(n, outdir):
    pop = {
        "body_geno": [],
        "spring_geno": [],
        "id": []
    }
    for i in range(n):
        bg, sg = sample_geno()
        pop["body_geno"].append(bg)
        pop["spring_geno"].append(sg)
        pop["id"].append(f"0-{i}")
    return save_pop(pop, os.path.join(outdir, "robots.pkl"))

def convert(body_geno, spring_geno):
    ## Body and spring genotype to list of points and springs
    global spring_order, triangle_to_points, triangle_to_springs
    spring_length = 0.05
    spring_K = 3e4
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
        spr = (
            points.index(a), 
            points.index(b), 
            spring_length, 
            spring_K, 
            spring_geno[idx] * 0.1
        )
        mesh_springs.append(spr)
    points = rescale_points(points)
    return points, mesh_springs

def geno_2_pheno(pop_file):
    pop = load_pop(pop_file)
    pop_pheno = {
        "points": [],
        "springs": [],
    }
    n = len(pop["body_geno"])
    for i in range(n):
        bg, sg = pop["body_geno"][i], pop["spring_geno"][i]
        points, springs = convert(bg, sg)
        pop_pheno["points"].append(points)
        pop_pheno["springs"].append(springs)
    pop.update(pop_pheno)
    save_pop(pop, pop_file)

import os, argparse, taichi as ti, math, numpy as np, pickle, sys

## Number of simulation steps
steps = 1000
dt = 0.004

## Simulation parameters
ground_height = 0.1
gravity = -4.8
spring_omega = 10
damping = 15

## NN / robot population parameters
max_springs = 0
max_objects = 0
n_robots = 0
n_sin_waves = 10
n_hidden = 32
gradient_clip = 0.2

MAX_INT = np.iinfo(np.int32).max

def set_ti_globals():
    ## Declare Taichi global fields
    scalar = lambda: ti.field(dtype=ti.f64)
    vec = lambda: ti.Vector.field(2, dtype=ti.f64)
    global loss, x, v, v_inc, n_objects, n_springs, spring_anchor_a, \
        spring_anchor_b, spring_length, spring_stiffness, spring_actuation, \
            weights1, bias1, weights2, bias2, hidden, center, act, update_scale
    loss = scalar()
    x = vec()
    v = vec()
    v_inc = vec()
    n_objects = ti.field(ti.i32)
    n_springs = ti.field(ti.i32)
    spring_anchor_a = ti.field(ti.i32)
    spring_anchor_b = ti.field(ti.i32)
    spring_length = scalar()
    spring_stiffness = scalar()
    spring_actuation = scalar()
    weights1 = scalar()
    bias1 = scalar()
    weights2 = scalar()
    bias2 = scalar()
    hidden = scalar()
    center = vec()
    act = scalar()
    update_scale = scalar()

def max_input_states():
    ## Max number of NN input features
    return n_sin_waves + 4 * max_objects + 2

@ti.func
def max_input_states_ti():
    ## Max number of NN input features (Taichi)
    return n_sin_waves + 4 * max_objects + 2

def n_input_states(robot_id):
    ## Robot specific number of NN input features
    return n_sin_waves + 4 * n_objects[robot_id] + 2

@ti.func
def n_input_states_ti(robot_id):
    ## Robot specific number of NN input features (Taichi)
    return n_sin_waves + 4 * n_objects[robot_id] + 2

def allocate_fields():
    ## Allocate Taichi field memory
    ti.root.dense(ti.ijk, (n_robots, steps, max_objects)).place(x, v, v_inc)
    ti.root.dense(ti.ij, (n_robots, max_springs)).place(spring_anchor_a, spring_anchor_b, spring_length, spring_stiffness, spring_actuation)
    ti.root.dense(ti.i, n_robots).place(n_objects, n_springs)
    ti.root.dense(ti.ijk, (n_robots, n_hidden, max_input_states())).place(weights1)
    ti.root.dense(ti.ij, (n_robots, n_hidden)).place(bias1)
    ti.root.dense(ti.ijk, (n_robots, max_springs, n_hidden)).place(weights2)
    ti.root.dense(ti.ij, (n_robots, max_springs)).place(bias2)
    ti.root.dense(ti.ijk, (n_robots, steps, n_hidden)).place(hidden)
    ti.root.dense(ti.ijk, (n_robots, steps, max_springs)).place(act)
    ti.root.dense(ti.ij, (n_robots, steps)).place(center)
    ti.root.dense(ti.i, n_robots).place(update_scale)
    ti.root.dense(ti.i, n_robots).place(loss)
    ti.root.lazy_grad()

@ti.kernel
def compute_center(t: ti.i32):
    ## Compute center of mass for each robot
    for r, i in ti.ndrange(n_robots, max_objects):
        if i < n_objects[r]:
            center[r, t] += x[r, t, i]
    for r in range(n_robots):
        center[r, t] += (1.0 / n_objects[r]) * center[r, t] - center[r, t]

@ti.kernel
def nn1(t: ti.i32):
    ## NN part 1; compute hidden activations through fully connected layer

    ## Phase offset sin wave input features
    for r, i, j in ti.ndrange(n_robots, n_hidden, n_sin_waves):
        hidden[r, t, i] += weights1[r, i, j] * ti.sin(spring_omega * t * dt + 2 * math.pi / n_sin_waves * j)

    ## Proprioceptive input features
    for r, i, j in ti.ndrange(n_robots, n_hidden, max_objects):
        if j < n_objects[r]:
            offset = x[r, t, j] - center[r, t]
            hidden[r, t, i] += weights1[r, i, j * 4 + n_sin_waves] * offset[0] * 0.05
            hidden[r, t, i] += weights1[r, i, j * 4 + n_sin_waves + 1] * offset[1] * 0.05
            hidden[r, t, i] += weights1[r, i, j * 4 + n_sin_waves + 2] * v[r, t, j][0] * 0.05
            hidden[r, t, i] += weights1[r, i, j * 4 + n_sin_waves + 3] * v[r, t, j][1] * 0.05
    
    ## Bias and activation function
    for r, i in ti.ndrange(n_robots, n_hidden):
        hidden[r, t, i] += weights1[r, i, n_objects[r] * 4 + n_sin_waves]
        hidden[r, t, i] += weights1[r, i, n_objects[r] * 4 + n_sin_waves + 1]
        hidden[r, t, i] += bias1[r, i]
        hidden[r, t, i] += ti.tanh(hidden[r, t, i]) - hidden[r, t, i]

@ti.kernel
def nn2(t: ti.i32):
    ## NN part 2; compute spring activations through fully connected layer
    for r, i, j in ti.ndrange(n_robots, max_springs, n_hidden):
        if i < n_springs[r]:
            act[r, t, i] += weights2[r, i, j] * hidden[r, t, j]
    for r, i in ti.ndrange(n_robots, max_springs):
        if i < n_springs[r]:
            act[r, t, i] += bias2[r, i]
            act[r, t, i] += ti.tanh(act[r, t, i]) - act[r, t, i]

@ti.kernel
def apply_spring_force(t: ti.i32):
    ## Compute impulses acting on objects through springs
    for r, i in ti.ndrange(n_robots, max_springs):
        if i < n_springs[r]:
            a = spring_anchor_a[r, i]
            b = spring_anchor_b[r, i]
            pos_a = x[r, t, a]
            pos_b = x[r, t, b]
            dist = pos_a - pos_b
            length = dist.norm() + 1e-4
            ## Where spring_actuation[r, i] == 0, the target length is constant
            target_length = spring_length[r, i] * (1.0 + spring_actuation[r, i] * act[r, t, i])
            impulse = dt * (length - target_length) * spring_stiffness[r, i] / length * dist
            ti.atomic_add(v_inc[r, t + 1, a], -impulse)
            ti.atomic_add(v_inc[r, t + 1, b], impulse)

@ti.kernel
def advance(t: ti.i32):
    ## Update positions and velocities of objects
    for r, i in ti.ndrange(n_robots, max_objects):
        if i < n_objects[r]:
            s = ti.exp(-dt * damping)
            old_v = s * v[r, t - 1, i] + dt * gravity * ti.Vector([0.0, 1.0]) + v_inc[r, t, i]
            old_x = x[r, t - 1, i]
            new_x = old_x + dt * old_v
            toi = 0.0
            new_v = old_v
            ## Collision with ground; "no-slip" condition
            if new_x[1] < ground_height and old_v[1] < -1e-4:
                toi = -(old_x[1] - ground_height) / old_v[1]
                new_v = ti.Vector([0.0, 0.0])
            new_x = old_x + toi * old_v + (dt - toi) * new_v
            v[r, t, i] = new_v
            x[r, t, i] = new_x

@ti.kernel
def compute_loss(t: ti.i32):
    ## Loss is negated horizontal CoM displacement
    for r, i in ti.ndrange(n_robots, max_objects):
        if i < n_objects[r]:
            loss[r] += x[r, t, i][0] - x[r, 0, i][0]
    for r in range(n_robots):
        loss[r] += (-1.0 / n_objects[r]) * loss[r] - loss[r]

@ti.kernel
def clear_states():
    ## Reset state variables
    for r, t in ti.ndrange(n_robots, steps):     
        center[r, t] = ti.Vector([0.0, 0.0])
    for r, t, i in ti.ndrange(n_robots, steps, max_objects):
        v_inc[r, t, i] = ti.Vector([0.0, 0.0])
        v[r, t, i] = ti.Vector([0.0, 0.0])
        if t > 0:
            x[r, t, i] = ti.Vector([0.0, 0.0])
    for r, t, i in ti.ndrange(n_robots, steps, n_hidden):
        hidden[r, t, i] = 0.0
    for r, t, i in ti.ndrange(n_robots, steps, max_springs):
        act[r, t, i] = 0.0
    for r in ti.ndrange(n_robots):
        update_scale[r] = 0.0

@ti.kernel
def clear_grad():
    ## Reset gradients
    for r, i in ti.ndrange(n_robots, max_objects):   
        bias1.grad[r, i] = 0.0
    for r, i, j in ti.ndrange(n_robots, n_hidden, max_springs):
        weights2.grad[r, j, i] = 0.0
    for r, i, j in ti.ndrange(n_robots, n_hidden, max_input_states_ti()):
        weights1.grad[r, i, j] = 0.0
    for r, i in ti.ndrange(n_robots, max_springs):
        bias2.grad[r, i] = 0.0
        spring_length.grad[r, i] = 0.0
        spring_stiffness.grad[r, i] = 0.0
        spring_actuation.grad[r, i] = 0.0
    for r, t in ti.ndrange(n_robots, steps):
        center.grad[r, t] = ti.Vector([0.0, 0.0])
    for r, t, i in ti.ndrange(n_robots, steps, max_objects):
        v_inc.grad[r, t, i] = ti.Vector([0.0, 0.0])
        v.grad[r, t, i] = ti.Vector([0.0, 0.0])
        if t > 0:
            x.grad[r, t, i] = ti.Vector([0.0, 0.0])
    for r, t, i in ti.ndrange(n_robots, steps, n_hidden):
        hidden.grad[r, t, i] = 0.0
    for r, t, i in ti.ndrange(n_robots, steps, max_springs):
        act.grad[r, t, i] = 0.0

@ti.kernel
def clear_loss():
    loss.fill(0.0)

def clear():
    clear_states()
    clear_loss()
    clear_grad()

@ti.kernel
def setup_robot(id: ti.i32, n_obj: ti.i32, objects: ti.types.ndarray(), n_spr: ti.i32, springs: ti.types.ndarray()): # type: ignore
    n_objects[id] = n_obj
    for i in range(n_obj):
        x[id, 0, i] = ti.Vector([objects[i, 0], objects[i, 1]])

    n_springs[id] = n_spr
    for i in range(n_spr):
        spring_anchor_a[id, i] = ti.cast(springs[i, 0], ti.i32)
        spring_anchor_b[id, i] = ti.cast(springs[i, 1], ti.i32)
        spring_length[id, i] = springs[i, 2]
        spring_stiffness[id, i] = springs[i, 3]
        spring_actuation[id, i] = springs[i, 4]

def setup():
    global robots_file, idx0, idx1, n_robots, max_objects, max_springs

    with open(robots_file, "rb") as f:
        robots = pickle.load(f)

    if idx0 is None:
        idx0 = 0
    if idx1 is None:
        idx1 = len(robots['id'])

    print(f"Loading robots [{idx0}, {idx1}) into Taichi from {robots_file}...", flush=True)

    ## Compute max number of objects and springs for memory allocation
    all_springs = robots['springs'][idx0:idx1]
    all_objects = robots['points'][idx0:idx1]
    n_robots = len(all_objects)
    max_objects = max([len(o) for o in all_objects])
    max_springs = max([len(s) for s in all_springs])
    print(f"n_robots: {n_robots}, max_objects: {max_objects}, max_springs: {max_springs}", flush=True)
    allocate_fields()

    ## Set initial robot states
    for robot_id in range(idx0, idx1):
        obj = np.array(all_objects[robot_id], dtype=np.float64)
        spr = np.array(all_springs[robot_id], dtype=np.float64)
        n_obj, n_spr = len(obj), len(spr)
        setup_robot(robot_id, n_obj, obj, n_spr, spr)
    print("Robot states loaded...", flush=True)

    ## Random weights initialization
    print("Initializing weights...", flush=True)
    init_weights()

@ti.kernel
def fill_weights_ti(r: ti.i32, w1: ti.types.ndarray(), b1: ti.types.ndarray(), w2: ti.types.ndarray(), b2: ti.types.ndarray()): # type: ignore
    for i, j in ti.ndrange(n_hidden, n_input_states_ti(r)):
        weights1[r, i, j] = w1[i, j]
    for i in range(n_hidden):
        bias1[r, i] = b1[i]
    for i, j in ti.ndrange(n_springs[r], n_hidden):
        weights2[r, i, j] = w2[i, j]
    for i in range(n_springs[r]):
        bias2[r, i] = b2[i]

def fill_weights():
    global idx0, idx1, robots_file
    print(f"Loading weights from {robots_file}...", flush=True)
    with open(robots_file, "rb") as f:
        robots = pickle.load(f)
    assert 'weights' in robots
    weights = robots['weights']
    for r in range(idx0, idx1):
        w1 = weights[r]['w1']
        b1 = weights[r]['b1']
        w2 = weights[r]['w2']
        b2 = weights[r]['b2']
        fill_weights_ti(r, w1, b1, w2, b2)

@ti.kernel
def init_weights():
    ## Weights initialized with Xavier normal initialization
    ## Bias initialized with zeros
    ti.loop_config(serialize=True)
    for r, i, j in ti.ndrange(n_robots, n_hidden, max_input_states_ti()):
        if j < n_input_states_ti(r):
            weights1[r, i, j] = ti.randn() * ti.sqrt(2 / (n_hidden + n_input_states_ti(r))) * 2
    ti.loop_config(serialize=True)
    for r, i in ti.ndrange(n_robots, n_hidden):
        bias1[r, i] = 0.0
    ti.loop_config(serialize=True)
    for r, i, j in ti.ndrange(n_robots, max_springs, n_hidden):
        if i < n_springs[r]:
            weights2[r, i, j] = ti.randn() * ti.sqrt(2 / (n_hidden + n_springs[r])) * 3
    ti.loop_config(serialize=True)
    for r, i in ti.ndrange(n_robots, max_springs):
        if i < n_springs[r]:
            bias2[r, i] = 0.0

@ti.kernel
def update_weights():
    ## Accumulate squared gradient values for each robot
    for r, i, j in ti.ndrange(n_robots, n_hidden, max_input_states_ti()):
        if j < n_input_states_ti(r):
            update_scale[r] += weights1.grad[r, i, j]**2
    for r, i in ti.ndrange(n_robots, n_hidden):
        update_scale[r] += bias1.grad[r, i]**2
    for r, i, j in ti.ndrange(n_robots, max_springs, n_hidden):
        if i < n_springs[r]:
            update_scale[r] += weights2.grad[r, i, j]**2
    for r, i in ti.ndrange(n_robots, max_springs):
        if i < n_springs[r]:
            update_scale[r] += bias2.grad[r, i]**2
    ## Unique learning rate for each robot inversely proportional to RSS of gradients
    for r in ti.ndrange(n_robots):
        update_scale[r] += gradient_clip / (update_scale[r]**0.5 + 1e-6) - update_scale[r]

    ## Update weights
    for r, i, j in ti.ndrange(n_robots, n_hidden, max_input_states_ti()):
        if j < n_input_states_ti(r):
            weights1[r, i, j] -= update_scale[r] * weights1.grad[r, i, j]
    for r, i in ti.ndrange(n_robots, n_hidden):
        bias1[r, i] -= update_scale[r] * bias1.grad[r, i]
    for r, i, j in ti.ndrange(n_robots, max_springs, n_hidden):
        if i < n_springs[r]:
            weights2[r, i, j] -= update_scale[r] * weights2.grad[r, i, j]
    for r, i in ti.ndrange(n_robots, max_springs):
        if i < n_springs[r]:
            bias2[r, i] -= update_scale[r] * bias2.grad[r, i]

def save_weights(dir, robot_idx, offset, name):
    w1 = weights1.to_numpy()
    b1 = bias1.to_numpy()
    w2 = weights2.to_numpy()
    b2 = bias2.to_numpy()
    for r in robot_idx:
        with open(os.path.join(dir, str(r+offset), f'{name}.pkl'), "wb") as f:
            pickle.dump({'w1': w1[r], 'b1': b1[r], 'w2': w2[r], 'b2': b2[r]}, f)

def forward(save_state=False):
    ## Forward simulation
    for t in range(1, steps):
        compute_center(t - 1)
        nn1(t - 1)
        nn2(t - 1)
        apply_spring_force(t - 1)
        advance(t)
    compute_loss(steps - 1)

    if save_state:
        global outdir
        os.makedirs(os.path.join(outdir, "state"), exist_ok=True)
        np.save(os.path.join(outdir, "state", 'loss.npy'), loss.to_numpy())
        np.save(os.path.join(outdir, "state", "v.npy"), v.to_numpy())
        np.save(os.path.join(outdir, "state", "x.npy"), x.to_numpy())
        np.save(os.path.join(outdir, "state", "center.npy"), center.to_numpy())
        np.save(os.path.join(outdir, "state", "act.npy"), act.to_numpy())
        np.save(os.path.join(outdir, "state", "spring_actuation.npy"), spring_actuation.to_numpy())
        np.save(os.path.join(outdir, "state", "spring_anchor_a"), spring_anchor_a.to_numpy())
        np.save(os.path.join(outdir, "state", "spring_anchor_b"), spring_anchor_b.to_numpy())

def manual_backward():
    ## Compute gradients
    compute_loss.grad(steps - 1)
    for t in range(steps-1, 0, -1):
        advance.grad(t)
        apply_spring_force.grad(t - 1)
        nn2.grad(t - 1)
        nn1.grad(t - 1)
        compute_center.grad(t - 1)

def clean_loss(loss_hist, k, best_loss):
    ## Check for NaNs and Infs when computing loss
    loss_numpy = loss.to_numpy()
    isnan = np.isnan(loss_numpy)
    isinf = np.isinf(loss_numpy)
    invalid_idx = np.unique(np.where(isnan | isinf)[0])
    loss_hist[:, k] = loss_numpy
    better_idx = np.where(loss_hist[:, k] < best_loss)[0]
    better_idx = np.setdiff1d(better_idx, invalid_idx)    
    best_loss[better_idx] = loss_hist[better_idx, k]
    return loss_hist, best_loss, better_idx

def optimize():
    global iters, outdir, idx0, idx1

    ## Indices of robots to optimize
    idx0 = 0 if idx0 is None else idx0
    idx1 = n_robots if idx1 is None else idx1

    print(f"Optimizing {iters} iterations for {steps} steps...", flush=True)

    weights_dir = os.path.join(outdir, "weights")

    ## Tracking loss trajectories and best performance so far
    loss_hist = np.zeros((n_robots, iters+1))
    best_loss = np.ones(n_robots) * MAX_INT

    ## Save initial weights
    save_weights(weights_dir, np.arange(n_robots), idx0, "init")

    ## Learning loop
    for k in range(iters):
        clear()
        forward()

        ## Save the weights if loss improved
        loss_hist, best_loss, save_idx = clean_loss(loss_hist, k, best_loss)
        if len(save_idx) > 0:
            save_weights(weights_dir, save_idx, idx0, f"best")

        ## Loss grad must be manually set to 1.0 for backward pass
        loss.grad.fill(1.0)
        manual_backward()
        update_weights()

    clear()
    forward()
    loss_hist, _, _ = clean_loss(loss_hist, k+1, best_loss)

    ## Save final losses
    loss_save_path = os.path.join(outdir, f"loss_{idx0}-{idx1}.npy")
    np.save(loss_save_path, loss_hist)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=35, help="Number of learning iterations (default: 35)")
    parser.add_argument('--debug', default=False, action='store_true') ## Taichi debug mode: https://docs.taichi-lang.org/docs/debugging
    parser.add_argument('--device_id', type=int, default=0, help="Numeric GPU device id (default: 0)")
    parser.add_argument('--outdir', type=str, required=True, help="Output directory")
    parser.add_argument('--robots_file', type=str, required=True, help="Pickle file containing robots in population")    
    parser.add_argument('--idx0', type=int, default=None, help="Start index of robot population")
    parser.add_argument('--idx1', type=int, default=None, help="End index of robot population")
    parser.add_argument('--seed', type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument('--logfile', type=str, default=None, help="Output log file (default: None)")

    ## If provided one forward pass is conducted and generated state is saved for visualization
    ## Note: generated state is saved to outdir
    ## Note: presumes weights are available in provided robots_file
    ## Note: idx0 and idx1 are used to select robots to visualize
    parser.add_argument('--visualize', action='store_true', help="Generate state for visualization")

    options = parser.parse_args()

    ## Std out and std err to log file
    if options.logfile is not None:
        sys.stdout = open(options.logfile, 'w')
        sys.stderr = sys.stdout

    ## Use only provided GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(options.device_id)
    print(f"Using CUDA_VISIBLE_DEVICES={options.device_id}", flush=True)

    ## Initialize taichi runtime
    ti.init(default_fp=ti.f64, arch=ti.cuda, debug=options.debug, random_seed=options.seed, device_memory_fraction=0.99)

    ## Setup taichi global memory layout
    set_ti_globals()

    print(f"Writing losses to {options.outdir}", flush=True)

    print(f"Random seed: {options.seed}", flush=True)

    ## Set global parameters
    global robots_file, idx0, idx1, outdir, iters
    robots_file = options.robots_file
    idx0 = options.idx0
    idx1 = options.idx1
    outdir = options.outdir
    iters = options.iters

    ## Load initial robot states into Taichi
    setup()

    if options.visualize:
        ## Run a single forward pass to generate state for visualization
        fill_weights()
        print("Simulating to produce state for visualization...", flush=True)
        forward(save_state=True)
    else:
        ## Run optimization loop
        optimize()


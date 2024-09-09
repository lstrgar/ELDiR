import os, taichi as ti, math, numpy as np, pickle, sys

## Number of simulation steps
sim_steps = 1000

## Simulation time constant
dt = 0.004

## Number of learning iterations
learning_iters = 35

## Simulation parameters
base_ground_height = 0.1
gravity = -4.8
spring_omega = 10
damping = 15
friction = 1.0
n_ground_segs = 0
ground_file = None

## Robot population parameters
max_springs = 0
max_objects = 0
n_robots = 0

## NN parameters
n_sin_waves = 10
n_hidden = 32
gradient_clip = 0.2

MAX_INT = np.iinfo(np.int32).max

## Declare Taichi global fields
def set_ti_globals():
    scalarf64 = lambda: ti.field(dtype=ti.f64)
    scalari32 = lambda: ti.field(dtype=ti.i32)
    vecf64 = lambda: ti.Vector.field(2, dtype=ti.f64)
    global x, v, v_inc, center, loss, update_scale, n_objects, n_springs, \
        spring_anchor_a, spring_anchor_b, spring_length, spring_stiffness, spring_actuation, \
            weights1, bias1, weights2, bias2, hidden, act, \
                ground_segs_x0, ground_segs_y0, ground_segs_slope, ground_segs_shift, ground_segs_len
    loss = scalarf64()
    x = vecf64()
    v = vecf64()
    v_inc = vecf64()
    n_objects = scalari32()
    n_springs = scalari32()
    spring_anchor_a = scalari32()
    spring_anchor_b = scalari32()
    spring_length = scalarf64()
    spring_stiffness = scalarf64()
    spring_actuation = scalarf64()
    weights1 = scalarf64()
    bias1 = scalarf64()
    weights2 = scalarf64()
    bias2 = scalarf64()
    hidden = scalarf64()
    center = vecf64()
    act = scalarf64()
    update_scale = scalarf64()
    ground_segs_x0 = scalarf64()
    ground_segs_y0 = scalarf64()
    ground_segs_slope = scalarf64()
    ground_segs_shift = scalarf64()
    ground_segs_len = scalarf64()

## Population max number of NN inputs
def max_input_states():
    return n_sin_waves + 4 * max_objects + 2

@ti.func
def max_input_states_ti():
    return n_sin_waves + 4 * max_objects + 2

## Individual robot number of NN inputs
def n_input_states(robot_id):
    return n_sin_waves + 4 * n_objects[robot_id] + 2

@ti.func
def n_input_states_ti(robot_id):
    return n_sin_waves + 4 * n_objects[robot_id] + 2

## Allocate Taichi field memory
def allocate_fields():
    ti.root.dense(ti.ijk, (n_robots, sim_steps, max_objects)).place(x, v, v_inc)
    ti.root.dense(ti.ij, (n_robots, max_springs)).place(spring_anchor_a, spring_anchor_b, spring_length, spring_stiffness, spring_actuation)
    ti.root.dense(ti.i, n_robots).place(n_objects, n_springs)
    ti.root.dense(ti.ijk, (n_robots, n_hidden, max_input_states())).place(weights1)
    ti.root.dense(ti.ij, (n_robots, n_hidden)).place(bias1)
    ti.root.dense(ti.ijk, (n_robots, max_springs, n_hidden)).place(weights2)
    ti.root.dense(ti.ij, (n_robots, max_springs)).place(bias2)
    ti.root.dense(ti.ijk, (n_robots, sim_steps, n_hidden)).place(hidden)
    ti.root.dense(ti.ijk, (n_robots, sim_steps, max_springs)).place(act)
    ti.root.dense(ti.ij, (n_robots, sim_steps)).place(center)
    ti.root.dense(ti.i, n_robots).place(update_scale)
    ti.root.dense(ti.i, n_robots).place(loss)
    ti.root.dense(ti.i, n_ground_segs).place(ground_segs_x0, ground_segs_y0, ground_segs_slope, ground_segs_shift, ground_segs_len)
    ti.root.lazy_grad()

## Compute center of mass for each robot
@ti.kernel
def compute_center(t: ti.i32):
    for r, i in ti.ndrange(n_robots, max_objects):
        if i < n_objects[r]:
            center[r, t] += x[r, t, i]
    for r in range(n_robots):
        center[r, t] += (1.0 / n_objects[r]) * center[r, t] - center[r, t]

## NN part 1; compute hidden activations through fully connected layer
@ti.kernel
def nn1(t: ti.i32):
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

## NN part 2; compute spring activations through fully connected layer
@ti.kernel
def nn2(t: ti.i32):
    for r, i, j in ti.ndrange(n_robots, max_springs, n_hidden):
        if i < n_springs[r]:
            act[r, t, i] += weights2[r, i, j] * hidden[r, t, j]
    
    ## Bias and activation function
    for r, i in ti.ndrange(n_robots, max_springs):
        if i < n_springs[r]:
            act[r, t, i] += bias2[r, i]
            act[r, t, i] += ti.tanh(act[r, t, i]) - act[r, t, i]

## Compute impulses acting on objects through springs
@ti.kernel
def apply_spring_force(t: ti.i32):
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

def ground_height_at_(x_val, seg):
    ground_height = base_ground_height
    if x_val >= 0 and seg >= 0:
        slope = ground_segs_slope[seg]
        shift = ground_segs_shift[seg]
        ground_height = x_val * slope + shift
    return ground_height

@ti.func
def ground_height_at(x_val: ti.f64, seg: ti.i32):
    ground_height = base_ground_height
    if x_val >= 0 and seg >= 0:
        slope = ground_segs_slope[seg]
        shift = ground_segs_shift[seg]
        ground_height = x_val * slope + shift
    return ground_height

def ground_seg_at_(x_val):
    seg = -1
    if x_val >= 0:
        seg = 0
        for i in range(n_ground_segs):
            if x_val >= ground_segs_x0[i]:
                seg = i
    return seg

@ti.func
def ground_seg_at(x_val: ti.f64):
    seg = -1
    if x_val >= 0:
        seg = 0
        ti.loop_config(serialize=True)
        for i in ti.static(range(n_ground_segs)):
            if x_val >= ground_segs_x0[i]:
                seg = i
    return seg

@ti.func
def distance_to_ground_at(x_val: ti.f64, y_val: ti.f64, seg: ti.i32):
    slope = ground_segs_slope[seg]
    shift = ground_segs_shift[seg]
    return ti.abs(-slope * x_val + y_val - shift) / ti.sqrt(1 + slope**2)

@ti.func
def normal_vec(seg: ti.i32):
    slope = ground_segs_slope[seg]
    return ti.Vector([-slope / ti.sqrt(1 + slope**2), 1 / ti.sqrt(1 + slope**2)])

@ti.func
def compute_toi(seg: ti.i32, x_val: ti.f64, y_val: ti.f64, vx: ti.f64, vy: ti.f64):
    dist = distance_to_ground_at(x_val, y_val, seg)
    norm_vec = normal_vec(seg)
    v = ti.Vector([vx, vy])
    norm_vec_mag = ti.abs(v.dot(norm_vec))
    toi = dist / (norm_vec_mag + 1e-10)
    return toi

@ti.func
def new_v_on_contact(seg: ti.i32, vx: ti.f64, vy: ti.f64):
    norm_vec = normal_vec(seg)
    v = ti.Vector([vx, vy])
    norm_vec_scale = v.dot(norm_vec)
    norm_vec = norm_vec * norm_vec_scale
    norm_vec_mag = norm_vec.norm()
    tan_vec = v - norm_vec
    tan_vec_mag = tan_vec.norm()
    friction_vec = tan_vec * -1
    friction_vec = friction_vec.normalized() * friction * ti.min(norm_vec_mag, tan_vec_mag)
    new_v = tan_vec + friction_vec
    return new_v

@ti.kernel
def advance(t: ti.i32):
    for r, i in ti.ndrange(n_robots, max_objects):
        if i < n_objects[r]:
            s = ti.exp(-dt * damping)
            v_= s * v[r, t - 1, i] + dt * gravity * ti.Vector([0.0, 1.0]) + v_inc[r, t, i]
            old_x = x[r, t - 1, i]
            new_x = old_x + dt * v_

            seg_new_x = ground_seg_at(new_x[0])
            ground_height = ground_height_at(new_x[0], seg_new_x)

            if new_x[1] < ground_height:
                seg_old_x = ground_seg_at(old_x[0])
                s0 = ti.min(seg_old_x, seg_new_x)
                s1 = ti.max(seg_old_x, seg_new_x)
                toi = compute_toi(s0, old_x[0], old_x[1], v_[0], v_[1])
                for j in ti.static(range(n_ground_segs)):
                    if j > s0 and j <= s1:
                        toi = ti.min(toi, compute_toi(j, old_x[0], old_x[1], v_[0], v_[1]))

                toi = ti.min(ti.max(0, toi), dt)
                new_x = old_x + toi * v_
                seg_new_x = ground_seg_at(new_x[0])
                v_ = new_v_on_contact(seg_new_x, v_[0], v_[1])
                ground_height = ground_height_at(new_x[0], seg_new_x)
                
                if toi < dt:
                    new_x = new_x + (dt - toi) * v_
                    seg_new_x = ground_seg_at(new_x[0])
                    ground_height = ground_height_at(new_x[0], seg_new_x)

                if new_x[1] < ground_height:
                    new_x[1] = ground_height

            v[r, t, i] = v_
            x[r, t, i] = new_x

## Loss computed as negated horizontal CoM displacement
@ti.kernel
def compute_loss(t: ti.i32):
    for r, i in ti.ndrange(n_robots, max_objects):
        if i < n_objects[r]:
            loss[r] += x[r, t, i][0] - x[r, 0, i][0]
    for r in range(n_robots):
        loss[r] += (-1.0 / n_objects[r]) * loss[r] - loss[r]

## Reset state variables for next sim
@ti.kernel
def clear_states():
    for r, t in ti.ndrange(n_robots, sim_steps):     
        center[r, t] = ti.Vector([0.0, 0.0])
    for r, t, i in ti.ndrange(n_robots, sim_steps, max_objects):
        v_inc[r, t, i] = ti.Vector([0.0, 0.0])
        v[r, t, i] = ti.Vector([0.0, 0.0])
        if t > 0:
            x[r, t, i] = ti.Vector([0.0, 0.0])
    for r, t, i in ti.ndrange(n_robots, sim_steps, n_hidden):
        hidden[r, t, i] = 0.0
    for r, t, i in ti.ndrange(n_robots, sim_steps, max_springs):
        act[r, t, i] = 0.0
    for r in ti.ndrange(n_robots):
        update_scale[r] = 0.0

## Reset gradients
@ti.kernel
def clear_grad():
    for r, i in ti.ndrange(n_robots, n_hidden):   
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
    for r, t in ti.ndrange(n_robots, sim_steps):
        center.grad[r, t] = ti.Vector([0.0, 0.0])
    for r, t, i in ti.ndrange(n_robots, sim_steps, max_objects):
        v_inc.grad[r, t, i] = ti.Vector([0.0, 0.0])
        v.grad[r, t, i] = ti.Vector([0.0, 0.0])
        if t > 0:
            x.grad[r, t, i] = ti.Vector([0.0, 0.0])
    for r, t, i in ti.ndrange(n_robots, sim_steps, n_hidden):
        hidden.grad[r, t, i] = 0.0
    for r, t, i in ti.ndrange(n_robots, sim_steps, max_springs):
        act.grad[r, t, i] = 0.0
    for i in ti.ndrange(n_ground_segs):
        ground_segs_x0.grad[i] = 0.0
        ground_segs_y0.grad[i] = 0.0
        ground_segs_slope.grad[i] = 0.0
        ground_segs_shift.grad[i] = 0.0
        ground_segs_len.grad[i] = 0.0

## Reset loss history
@ti.kernel
def clear_loss():
    loss.fill(0.0)

## Reset everything
def clear():
    clear_states()
    clear_loss()
    clear_grad()

## Load individual robot into Taichi
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

## Load all robots into taichi 
def setup(robots_file, idx0=None, idx1=None):
    global n_robots, max_objects, max_springs, ground_file, n_ground_segs

    with open(robots_file, "rb") as f:
        robots = pickle.load(f)

    if idx0 is None:
        idx0 = 0
    if idx1 is None:
        idx1 = len(robots['points'])

    print(f"Loading robots [{idx0}, {idx1}) into Taichi from {robots_file}...", flush=True)

    ## Compute max number of objects and springs for taichi memory allocation
    all_springs = robots['springs'][idx0:idx1]
    all_objects = robots['points'][idx0:idx1]
    n_robots = len(all_objects)
    max_objects = max([len(o) for o in all_objects])
    max_springs = max([len(s) for s in all_springs])
    print(f"num robots: {n_robots}, max num points: {max_objects}, max num springs: {max_springs}", flush=True)

    if ground_file is None:
        n_ground_segs = 3
        print(f"Using flat terrain...", flush=True)
    else:
        ground = np.load(ground_file)
        xs, ys, lens, slopes, shifts = ground
        n_ground_segs = len(xs)
        print(f"Loading terrain from {ground_file}...", flush=True)
    print(f"n_ground_segs: {n_ground_segs}", flush=True)

    allocate_fields()

    if ground_file is None:
        ground_segs_x0[0] = -20.0
        ground_segs_y0[0] = base_ground_height
        ground_segs_len[0] = 0.2
        ground_segs_slope[0] = 0.0
        ground_segs_shift[0] = base_ground_height
        ground_segs_x0[1] = 0.2
        ground_segs_y0[1] = base_ground_height
        ground_segs_len[1] = 0.2
        ground_segs_slope[1] = 0.5
        ground_segs_shift[1] = 0.0
        ground_segs_x0[2] = 0.4
        ground_segs_y0[2] = base_ground_height + 0.5 * 0.2
        ground_segs_len[2] = 0.6
        ground_segs_slope[2] = -0.25
        ground_segs_shift[2] = 0.3
    else:
        for i in range(n_ground_segs):
            ground_segs_x0[i] = xs[i]
            ground_segs_y0[i] = ys[i]
            ground_segs_slope[i] = slopes[i]
            ground_segs_shift[i] = shifts[i]
            ground_segs_len[i] = lens[i]

    ## Set initial robot states
    for robot_id in range(0, n_robots):
        obj = np.array(all_objects[robot_id], dtype=np.float64)
        spr = np.array(all_springs[robot_id], dtype=np.float64)
        n_obj, n_spr = len(obj), len(spr)
        setup_robot(robot_id, n_obj, obj, n_spr, spr)
    print("Robot states loaded...", flush=True)

    if "weights" in robots:
        ## Load provided weights for visualization
        print("Loading provided weights...", flush=True)
        weights = robots["weights"][0]
        fill_weights_ti(0, weights["w1"], weights["b1"], weights["w2"], weights["b2"])
    else:
        ## Random weights initialization
        print("Initializing weights...", flush=True)
        init_weights()

## Load provided weights for a single robot
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

## Load weights of all robots from file
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

## Random initialization of weights
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

## Perform weight update
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

    ## Gradient step
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

## Save weights of individual robot to disk
def save_weights(dir, robot_idx, offset, name):
    w1 = weights1.to_numpy()
    b1 = bias1.to_numpy()
    w2 = weights2.to_numpy()
    b2 = bias2.to_numpy()
    for r in robot_idx:
        if not os.path.exists(os.path.join(dir, str(r+offset))):
            os.makedirs(os.path.join(dir, str(r+offset)))
        with open(os.path.join(dir, str(r+offset), f'{name}.pkl'), "wb") as f:
            pickle.dump({'w1': w1[r], 'b1': b1[r], 'w2': w2[r], 'b2': b2[r]}, f)

## Run full forward simulation
def forward(save_state=False, outdir=None):
    global sim_steps
    for t in range(1, sim_steps):
        compute_center(t - 1)
        nn1(t - 1)
        nn2(t - 1)
        apply_spring_force(t - 1)
        advance(t)
    compute_loss(sim_steps - 1)

    ## Save simulation state
    if save_state and outdir is not None:
        os.makedirs(os.path.join(outdir, "state"), exist_ok=True)
        np.save(os.path.join(outdir, "state", 'loss.npy'), loss.to_numpy())
        np.save(os.path.join(outdir, "state", "v.npy"), v.to_numpy())
        np.save(os.path.join(outdir, "state", "x.npy"), x.to_numpy())
        np.save(os.path.join(outdir, "state", "center.npy"), center.to_numpy())
        np.save(os.path.join(outdir, "state", "act.npy"), act.to_numpy())
        np.save(os.path.join(outdir, "state", "spring_actuation.npy"), spring_actuation.to_numpy())
        np.save(os.path.join(outdir, "state", "spring_anchor_a"), spring_anchor_a.to_numpy())
        np.save(os.path.join(outdir, "state", "spring_anchor_b"), spring_anchor_b.to_numpy())

## Compute gradients in backwards pass
def manual_backward():
    compute_loss.grad(sim_steps - 1)
    for t in range(sim_steps-1, 0, -1):
        advance.grad(t)
        apply_spring_force.grad(t - 1)
        nn2.grad(t - 1)
        nn1.grad(t - 1)
        compute_center.grad(t - 1)

## Check for NaNs and Infs when computing loss
def clean_loss(loss_hist, k, best_loss):
    loss_numpy = loss.to_numpy()
    isnan = np.isnan(loss_numpy)
    isinf = np.isinf(loss_numpy)
    invalid_idx = np.unique(np.where(isnan | isinf)[0])
    loss_hist[:, k] = loss_numpy
    better_idx = np.where(loss_hist[:, k] < best_loss)[0]
    better_idx = np.setdiff1d(better_idx, invalid_idx)    
    best_loss[better_idx] = loss_hist[better_idx, k]
    return loss_hist, best_loss, better_idx

## Teach robots to walk
def optimize(outdir, idx0=None, idx1=None):
    global learning_iters
    ## Indices of robots to optimize
    idx0 = 0 if idx0 is None else idx0
    idx1 = n_robots if idx1 is None else idx1

    print(f"Optimizing {learning_iters} iterations for {sim_steps} steps...", flush=True)

    weights_dir = os.path.join(outdir, "weights")

    ## Tracking loss trajectories and best performance so far
    loss_hist = np.zeros((n_robots, learning_iters+1))
    best_loss = np.ones(n_robots) * MAX_INT

    ## Save initial weights
    save_weights(weights_dir, np.arange(n_robots), idx0, "init")

    ## Learning loop
    for k in range(learning_iters):
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

## End to end simulation: load robots, optimize locomotion, etc. 
def simulate(robots_file, outdir, device_id, logfile=None, idx0=None, idx1=None, seed=0, debug=False):
    ## Std out and std err to log file
    if logfile is not None:
        if os.path.exists(logfile):
            logfile = logfile.split(".")[0] + "_offspring.log"
        sys.stdout = open(logfile, 'w')
        sys.stderr = sys.stdout
        print(f"Writing stdout,stderr to: {logfile}", flush=True)

    ## Use only provided GPU device
    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        print(f"Using Cuda device ID: {device_id}", flush=True)
        arch = ti.cuda
    else:
        print("Using CPU", flush=True)
        arch = ti.cpu

    ## Initialize taichi runtime
    ti.init(default_fp=ti.f64, arch=arch, debug=debug, random_seed=seed, device_memory_fraction=0.99)

    ## Setup taichi global memory layout
    set_ti_globals()

    print(f"Writing losses to: {outdir}", flush=True)
    print(f"Random seed: {seed}", flush=True)

    ## Load initial robot states into Taichi
    setup(robots_file, idx0, idx1)

    ## Run optimization
    optimize(outdir, idx0, idx1)

def forward_visualization(robots_file, outdir):
    ## Initialize taichi runtime
    ti.init(default_fp=ti.f64, arch=ti.cpu, device_memory_fraction=0.99)

    ## Setup taichi global memory layout
    set_ti_globals()

    ## Load initial robot states into Taichi
    setup(robots_file)

    print(f"Running forward simulation for visualization...", flush=True)

    ## Run forward simulation
    forward(save_state=True, outdir=outdir)

    print(f"Simulation state saved to: {outdir}", flush=True)
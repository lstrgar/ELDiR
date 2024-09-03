import os
os.environ["ENABLE_TAICHI_HEADER_PRINT"] = "1"
import argparse
import taichi as ti
import math
import numpy as np
import pickle
import sys

steps = 1000
iters = 35
base_ground_height = 0.1
gravity = -7.0
pi = math.pi
spring_omega = 10
damping = 15
max_springs = 0
max_objects = 0
n_robots = 0
n_ground_segs = 0
n_sin_waves = 10
n_hidden = 32
dt = 0.004
gradient_clip = 0.2
friction = 1.0
MAX_INT = np.iinfo(np.int32).max

def set_ti_globals():
    scalar = lambda: ti.field(dtype=ti.f64)
    vec = lambda: ti.Vector.field(2, dtype=ti.f64)
    global loss, x, v, v_inc, n_objects, n_springs, spring_anchor_a, \
        spring_anchor_b, spring_length, spring_stiffness, spring_actuation, \
            weights1, bias1, weights2, bias2, hidden, center, act, update_scale, \
                ground_segs_x0, ground_segs_y0, ground_segs_slope, ground_segs_shift, ground_segs_len
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
    ground_segs_x0 = scalar()
    ground_segs_y0 = scalar()
    ground_segs_slope = scalar()
    ground_segs_shift = scalar()
    ground_segs_len = scalar()

def max_input_states():
    return n_sin_waves + 4 * max_objects + 2

@ti.func
def max_input_states_ti():
    return n_sin_waves + 4 * max_objects + 2

def n_input_states(robot_id):
    return n_sin_waves + 4 * n_objects[robot_id] + 2

@ti.func
def n_input_states_ti(robot_id):
    return n_sin_waves + 4 * n_objects[robot_id] + 2

def allocate_fields():
    global track_grad
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
    ti.root.dense(ti.i, n_ground_segs).place(ground_segs_x0, ground_segs_y0, ground_segs_slope, ground_segs_shift, ground_segs_len)
    if track_grad:
        ti.root.lazy_grad()

@ti.kernel
def compute_center(t: ti.i32):
    for r, i in ti.ndrange(n_robots, max_objects):
        if i < n_objects[r]:
            center[r, t] += x[r, t, i]
    for r in range(n_robots):
        center[r, t] += (1.0 / n_objects[r]) * center[r, t] - center[r, t]

@ti.kernel
def nn1(t: ti.i32):
    for r, i, j in ti.ndrange(n_robots, n_hidden, n_sin_waves):
        hidden[r, t, i] += weights1[r, i, j] * ti.sin(spring_omega * t * dt + 2 * pi / n_sin_waves * j)

    for r, i, j in ti.ndrange(n_robots, n_hidden, max_objects):
        if j < n_objects[r]:
            offset = x[r, t, j] - center[r, t]
            hidden[r, t, i] += weights1[r, i, j * 4 + n_sin_waves] * offset[0] * 0.05
            hidden[r, t, i] += weights1[r, i, j * 4 + n_sin_waves + 1] * offset[1] * 0.05
            hidden[r, t, i] += weights1[r, i, j * 4 + n_sin_waves + 2] * v[r, t, j][0] * 0.05
            hidden[r, t, i] += weights1[r, i, j * 4 + n_sin_waves + 3] * v[r, t, j][1] * 0.05
    
    for r, i in ti.ndrange(n_robots, n_hidden):
        hidden[r, t, i] += weights1[r, i, n_objects[r] * 4 + n_sin_waves]
        hidden[r, t, i] += weights1[r, i, n_objects[r] * 4 + n_sin_waves + 1]
        hidden[r, t, i] += bias1[r, i]
        hidden[r, t, i] += ti.tanh(hidden[r, t, i]) - hidden[r, t, i]

@ti.kernel
def nn2(t: ti.i32):
    for r, i, j in ti.ndrange(n_robots, max_springs, n_hidden):
        if i < n_springs[r]:
            act[r, t, i] += weights2[r, i, j] * hidden[r, t, j]
    for r, i in ti.ndrange(n_robots, max_springs):
        if i < n_springs[r]:
            act[r, t, i] += bias2[r, i]
            act[r, t, i] += ti.tanh(act[r, t, i]) - act[r, t, i]

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
def advance_toi(t: ti.i32):
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

@ti.kernel
def compute_loss(t: ti.i32):
    for r, i in ti.ndrange(n_robots, max_objects):
        if i < n_objects[r]:
            loss[r] += x[r, t, i][0] - x[r, 0, i][0]
    for r in range(n_robots):
        loss[r] += (-1.0 / n_objects[r]) * loss[r] - loss[r]

@ti.kernel
def clear_states():
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
    for r in ti.ndrange(n_robots):
        update_scale.grad[r] = 0.0
    for i in ti.ndrange(n_ground_segs):
        ground_segs_x0.grad[i] = 0.0
        ground_segs_y0.grad[i] = 0.0
        ground_segs_slope.grad[i] = 0.0
        ground_segs_shift.grad[i] = 0.0
        ground_segs_len.grad[i] = 0.0

@ti.kernel
def clear_loss():
    loss.fill(0.0)

def clear():
    clear_states()
    clear_loss()
    clear_grad()

def setup(robots_file, idx0, idx1, fill_weights=False):
    global n_robots, max_objects, max_springs, weights_dir_name, outdir, ground_file, n_ground_segs
    with open(robots_file, "rb") as f:
        robots = pickle.load(f)

    if idx0 is None:
        idx0 = 0
    if idx1 is None:
        idx1 = len(robots['id'])

    print(f"Loading robots [{idx0}, {idx1}) into Taichi from {robots_file}...", flush=True)

    all_springs = robots['springs'][idx0:idx1]
    all_objects = robots['points'][idx0:idx1]
    n_robots = len(all_objects)
    max_objects = max([len(o) for o in all_objects])
    max_springs = max([len(s) for s in all_springs])
    print(f"n_robots: {n_robots}, max_objects: {max_objects}, max_springs: {max_springs}", flush=True)

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

    for robot_id in range(n_robots):
        objects = all_objects[robot_id]
        springs = all_springs[robot_id]

        n_objects[robot_id] = len(objects)
        objects_arr = np.array(objects)
        on_ground = np.where(objects_arr[:,1] == base_ground_height)[0]
        max_ground_height = 0
        for o in on_ground:
            seg = ground_seg_at_(objects_arr[o, 0])
            ground_height = ground_height_at_(objects_arr[o, 0], seg)
            if ground_height > max_ground_height:
                max_ground_height = ground_height
        shift = max_ground_height + 0.05 - base_ground_height
        for i in range(n_objects[robot_id]):
            x[robot_id, 0, i] = objects[i]
            x[robot_id, 0, i][1] += shift

        n_springs[robot_id] = len(springs)
        for i in range(n_springs[robot_id]):
            s = springs[i]
            spring_anchor_a[robot_id, i] = s[0]
            spring_anchor_b[robot_id, i] = s[1]
            spring_length[robot_id, i] = s[2]
            spring_stiffness[robot_id, i] = s[3]
            spring_actuation[robot_id, i] = s[4]

        if fill_weights:
            # weights_path = os.path.join(outdir, weights_dir_name, str(robot_id), "best.pkl")
            # with open(weights_path, "rb") as f:
            #     weights = pickle.load(f)
            weights = robots['weights'][robot_id]
            w1 = weights['w1']
            b1 = weights['b1']
            w2 = weights['w2']
            b2 = weights['b2']
            load_weights(robot_id, w1, b1, w2, b2)

    if fill_weights:
        print(f"Weights filled from file {robots_file}", flush=True)
    else:
        print("Initializing weights...", flush=True)
        init_weights()

    return robots['id']

@ti.kernel
def load_weights(robot_id: ti.i32, w1: ti.types.ndarray(), b1: ti.types.ndarray(), w2: ti.types.ndarray(), b2: ti.types.ndarray()):
    for i, j in ti.ndrange(n_hidden, n_input_states_ti(robot_id)):
        weights1[robot_id, i, j] = w1[i, j]
    for i in range(n_hidden):
        bias1[robot_id, i] = b1[i]
    for i, j in ti.ndrange(n_springs[robot_id], n_hidden):
        weights2[robot_id, i, j] = w2[i, j]
    for i in range(n_springs[robot_id]):
        bias2[robot_id, i] = b2[i]

@ti.kernel
def init_weights():
    ti.loop_config(serialize=True)
    for r, i, j in ti.ndrange(n_robots, n_hidden, max_input_states_ti()):
        if j < n_input_states_ti(r):
            weights1[r, i, j] = ti.randn() * ti.sqrt(2 / (n_hidden + n_input_states_ti(r))) * 2
    ti.loop_config(serialize=True)
    for r, i, j in ti.ndrange(n_robots, max_springs, n_hidden):
        if i < n_springs[r]:
            weights2[r, i, j] = ti.randn() * ti.sqrt(2 / (n_hidden + n_springs[r])) * 3

@ti.kernel
def update_weights():
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
    for r in ti.ndrange(n_robots):
        update_scale[r] += gradient_clip / (update_scale[r]**0.5 + 1e-6) - update_scale[r]

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
        if not os.path.exists(os.path.join(dir, str(r+offset))):
            os.makedirs(os.path.join(dir, str(r+offset)))
    for r in robot_idx:
        with open(os.path.join(dir, str(r+offset), f'{name}.pkl'), "wb") as f:
            pickle.dump({'w1': w1[r], 'b1': b1[r], 'w2': w2[r], 'b2': b2[r]}, f)

def forward():
    for t in range(1, steps):
        compute_center(t - 1)
        nn1(t - 1)
        nn2(t - 1)
        apply_spring_force(t - 1)
        advance_toi(t)
    compute_loss(steps - 1)

""""
def forward_viz(gui, imgdir, viz_interval, robot_ids):

    print(f"Simulating {steps} steps...", flush=True)

    for r in range(n_robots):
        if not os.path.exists(os.path.join(imgdir, str(r))):
            os.makedirs(os.path.join(imgdir, robot_ids[r]))

    def draw_frame(imgdir, rid, t):
        for i in range(n_ground_segs):
            x0 = ground_segs_x0[i]
            if x0 < 0 or x0 > 1:
                continue
            y0 = ground_segs_y0[i]
            x1 = x0 + ground_segs_len[i]
            x1 = min(x1, 1)
            y1 = y0 + ground_segs_slope[i] * (x1 - x0)
            gui.line(begin=(x0, y0), end=(x1, y1), color=0x0, radius=3)

        def circle(x, y, color):
            gui.circle((x, y), ti.rgb_to_hex(color), 7)

        for i in range(n_springs[rid]):

            def get_pt(x):
                return (x[0], x[1])
            
            a = act[rid, t - 1, i] * 0.5
            r = 2

            if spring_actuation[rid, i] == 0:
                a = 0
                c = 0x222222
            else:
                r = 4
                c = ti.rgb_to_hex((0.5 + a, 0.5 - abs(a), 0.5 - a))

            begin = get_pt(x[rid, t, spring_anchor_a[rid, i]])
            end = get_pt(x[rid, t, spring_anchor_b[rid, i]])
            gui.line(begin=begin, end=end, radius=r, color=c)

        for i in range(n_objects[rid]):
            color = (0.4, 0.6, 0.6)
            circle(x[rid, t, i][0], x[rid, t, i][1], color)

        imgdir = os.path.join(imgdir, str(robot_ids[rid]))
        imgpath = os.path.join(imgdir, '%04d.png' % t)
        gui.show(imgpath)

    for t in range(1, steps):
        compute_center(t - 1)
        nn1(t - 1)
        nn2(t - 1)
        apply_spring_force(t - 1)
        advance_toi(t)
        if t % viz_interval == 0:
            for r in range(n_robots):
                draw_frame(imgdir, r, t)

    compute_loss(steps - 1)
    np.save(os.path.join(imgdir, 'loss.npy'), loss.to_numpy())
    np.save(os.path.join(imgdir, "v.npy"), v.to_numpy())
    np.save(os.path.join(imgdir, "x.npy"), x.to_numpy())
    np.save(os.path.join(imgdir, "center.npy"), center.to_numpy())
    np.save(os.path.join(imgdir, "act.npy"), act.to_numpy())
    np.save(os.path.join(imgdir, "spring_actuation.npy"), spring_actuation.to_numpy())
    np.save(os.path.join(imgdir, "spring_anchor_a"), spring_anchor_a.to_numpy())
    np.save(os.path.join(imgdir, "spring_anchor_b"), spring_anchor_b.to_numpy())
    print(loss.to_numpy())
    print(f"Imgs and loss written to {imgdir}", flush=True)
"""

def manual_backward():
    compute_loss.grad(steps - 1)
    for t in range(steps-1, 0, -1):
        advance_toi.grad(t)
        apply_spring_force.grad(t - 1)
        nn2.grad(t - 1)
        nn1.grad(t - 1)
        compute_center.grad(t - 1)

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

def optimize(iters, out_dir, idx0, idx1, weights_dir_name):

    idx0 = 0 if idx0 is None else idx0
    idx1 = n_robots if idx1 is None else idx1

    print(f"Optimizing {iters} iterations for {steps} steps...", flush=True)

    weights_dir = os.path.join(out_dir, weights_dir_name)

    loss_hist = np.zeros((n_robots, iters+1))
    best_loss = np.ones(n_robots) * MAX_INT

    save_weights(weights_dir, np.arange(n_robots), idx0, "init")

    for k in range(iters):
        clear()
        forward()

        loss_hist, best_loss, save_idx = clean_loss(loss_hist, k, best_loss)
        if len(save_idx) > 0:
            save_weights(weights_dir, save_idx, idx0, f"best")

        loss.grad.fill(1.0)
        manual_backward()
        update_weights()

    clear()
    forward()
    loss_hist, _, _ = clean_loss(loss_hist, k+1, best_loss)

    loss_save_path = os.path.join(out_dir, f"loss_{idx0}-{idx1}.npy")
    np.save(loss_save_path, loss_hist)

#def visualize(robots_file, idx0, idx1, imgdir, viz_interval, gui):
    #robot_ids = setup(robots_file, idx0, idx1, fill_weights=True)
    #forward_viz(gui, imgdir, viz_interval, robot_ids)

def main(robots_file, idx0, idx1, outdir, iters, weights_dir_name):
    setup(robots_file, idx0, idx1)
    optimize(iters, outdir, idx0, idx1, weights_dir_name)

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--device_id', type=none_or_int, default=None)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--weights_dir_name', type=str, default="weights")
    parser.add_argument('--robots_file', type=str, required=True)
    parser.add_argument('--ground_file', type=str, default=None)
    parser.add_argument('--idx0', type=int, default=None)
    parser.add_argument('--idx1', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logfile', type=str, default=None)
    #parser.add_argument('--visualize', default=False, action='store_true')
    #parser.add_argument('--viz_interval', type=int, default=5)

    options = parser.parse_args()

    if options.logfile is not None:
        sys.stdout = open(options.logfile, 'w')
        sys.stderr = sys.stdout

    global weights_dir_name, outdir, ground_file
    weights_dir_name = options.weights_dir_name
    outdir = options.outdir
    ground_file = options.ground_file
    track_grad = True

    if options.device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(options.device_id)
        print(f"Using CUDA_VISIBLE_DEVICES={options.device_id}", flush=True)
        arch = ti.cuda
    else:
        print(f"Defaulting to CPU", flush=True)
        arch = ti.cpu

    if ground_file is not None:
        print(f"Using ground file: {ground_file}", flush=True)

    ti.init(default_fp=ti.f64, arch=arch, debug=options.debug, random_seed=options.seed, device_memory_fraction=0.99)

    set_ti_globals()

    #if options.visualize:
        #os.makedirs(options.outdir, exist_ok=True)
        #print(f"Visualizing robots in {options.robots_file}...", flush=True)
        #gui = ti.GUI("Mass Spring Robot", (512, 512), background_color=0xFFFFFF, show_gui=False)
        #visualize(options.robots_file, options.idx0, options.idx1, options.outdir, options.viz_interval, gui)
        #exit()

    print(f"Writing losses to {options.outdir}", flush=True)

    print(f"Random seed: {options.seed}", flush=True)

    main(options.robots_file, options.idx0, options.idx1, options.outdir, iters, options.weights_dir_name)


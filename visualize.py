import pickle, time, numpy as np, os, subprocess, seaborn as sns, sys, shutil
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from tqdm import tqdm
import moviepy.video.io.ImageSequenceClip
from IPython.display import Video
sys.path.append(os.path.dirname(os.path.abspath('')))

parser = ArgumentParser()
parser.add_argument('--generation', type=str)
parser.add_argument('--groundfile', type=str, default=None, help="Path to custom ground file")
args = parser.parse_args()
generation = args.generation
ground_file = args.groundfile

outdir = "./eldir-outputs"

viz_outdir = "./eldir-outputs/{}".format(generation)
print("Visualization output directory: ", viz_outdir)
if not os.path.exists(viz_outdir):
    os.makedirs(viz_outdir)

log_file = os.path.join(viz_outdir, f"vizout_{time.strftime('%Y%m%d-%H%M%S')}.txt")
err_file = os.path.join(viz_outdir, f"vizerr_{time.strftime('%Y%m%d-%H%M%S')}.txt")
sys.stdout = open(log_file, "w")
sys.stderr = open(err_file, "w")

with open(os.path.join(outdir, generation, "robots.pkl"), "rb") as f:
    robots = pickle.load(f)

i = 0
best_robot_points = robots['points'][i]
best_robot_springs = robots['springs'][i]
best_robot_id = robots['id'][i]
best_robot = {
    "points": [best_robot_points],
    "springs": [best_robot_springs],
    "weights": []
}

gen, idx = best_robot_id.split("-")
with open(os.path.join(outdir, gen, "weights", idx, "best.pkl"), "rb") as f:
    best_robot["weights"].append(pickle.load(f))

robot_save_file = os.path.join(viz_outdir, "best_robot.pkl")
print(f"Saving best robot to {robot_save_file}")
with open(robot_save_file, "wb") as f:
    pickle.dump(best_robot, f)

fig = plt.figure()

green = plt.colormaps['Greens'](0.75)
purple = plt.colormaps['Purples'](0.7)
gray = plt.colormaps['Greys'](0.4)

for j in range(len(best_robot_springs)):
    if best_robot_springs[j][-1] == 0:
        c = gray
    else:
        c = green
    plt.plot([best_robot_points[best_robot_springs[j][0]][0], best_robot_points[best_robot_springs[j][1]][0]],
                [best_robot_points[best_robot_springs[j][0]][1], best_robot_points[best_robot_springs[j][1]][1]], color=c)

for i in range(len(best_robot_points)):
    plt.plot(best_robot_points[i][0], best_robot_points[i][1], 'o', color=purple)

plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')

print(f"python visualize.py {robot_save_file} {viz_outdir}")

command = ["python", "visualize.py", robot_save_file, viz_outdir, "--groundfile", ground_file]
subprocess.run(command, capture_output=True, text=True)
#python visualize.py ./tmp-viz/gen-40/best_robot.pkl ./tmp-viz/gen-40

state_dir = os.path.join(viz_outdir, "state")
x = np.load(os.path.join(state_dir, "x.npy"))[0]
v = np.load(os.path.join(state_dir, "v.npy"))[0]
act = np.load(os.path.join(state_dir, "act.npy"))[0]
spring_actuation = np.load(os.path.join(state_dir, "spring_actuation.npy"))[0]
spring_anchor_a = np.load(os.path.join(state_dir, "spring_anchor_a.npy"))[0]
spring_anchor_b = np.load(os.path.join(state_dir, "spring_anchor_b.npy"))[0]
center = np.load(os.path.join(state_dir, "center.npy"))[0]
loss = np.load(os.path.join(state_dir, "loss.npy"))[0]

os.makedirs(os.path.join(viz_outdir, "frames"), exist_ok=True)

x_min, x_max = np.min(x[:, :, 0]), np.max(x[:, :, 0])
y_min, y_max = np.min(x[:, :, 1]), np.max(x[:, :, 1])

gray = plt.colormaps['Greys'](0.4)
greens = plt.colormaps['Greens']
purples = plt.colormaps['Purples']

steps = x.shape[0]

offx = []
offy = []
vx = []
vy = []

for t in range(steps):
    offx.append(x[t, :, 0] - center[t, 0])
    offy.append(x[t, :, 1] - center[t, 1])
    vx.append(v[t, :, 0])
    vy.append(v[t, :, 1])

offx = np.array(offx)
offy = np.array(offy)
vx = np.array(vx)
vy = np.array(vy)

offx = 0.0 + (offx - np.min(offx)) * (1.0 / (np.max(offx) - np.min(offx)))
offy = 0.0 + (offy - np.min(offy)) * (1.0 / (np.max(offy) - np.min(offy)))
vx = 0.25 + (vx - np.min(vx)) * (1.0 / (np.max(vx) - np.min(vx)))
vy = 0.25 + (vy - np.min(vy)) * (1.0 / (np.max(vy) - np.min(vy)))

frame_dir = os.path.join(viz_outdir, "frames")
if os.path.exists(frame_dir):
    shutil.rmtree(frame_dir)
os.makedirs(frame_dir, exist_ok=True)

for t in tqdm(range(0, steps, 2)):
    fig = plt.figure(figsize=(10, 5))

    for j in range(act.shape[1]):
        if spring_actuation[j] == 0:
            c = gray
        else:
            a = act[t, j]
            c = greens((a + 2) / 3)
        plt.plot([x[t, spring_anchor_a[j], 0], x[t, spring_anchor_b[j], 0]],
                 [x[t, spring_anchor_a[j], 1], x[t, spring_anchor_b[j], 1]], color=c)

    for i in range(x.shape[1]):
        c0 = purples(offx[t, i])
        c1 = purples(offy[t, i])
        c2 = purples(vx[t, i])
        c3 = purples(vy[t, i])
        plt.plot(x[t, i, 0], x[t, i, 1], 'o', color=c2, markersize=8)
        plt.plot(x[t, i, 0], x[t, i, 1], 'o', color=c3, markersize=6)
        plt.plot(x[t, i, 0], x[t, i, 1], 'o', color=c0, markersize=4)
        plt.plot(x[t, i, 0], x[t, i, 1], 'o', color=c1, markersize=2)

    if ground_file is None:
        plt.hlines(0.0915, x_min, x_max, color='black')
    else:
        ground = np.load(ground_file)
        xs, ys, lens, slopes, shifts = ground
        n_ground_segs = len(xs)

        for i in range(n_ground_segs):
            plt.plot([xs[i], xs[i] + lens[i]], [ys[i], ys[i] + lens[i] * slopes[i]], 'b')

        plt.xlim(-0.5, 2.5)
        plt.ylim(0, 1)
        plt.gca().set_aspect('equal', adjustable='box')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.xlim(x_min - 0.05, x_max + 0.05)
    plt.ylim(y_min - 0.05, y_max + 0.05)
    plt.text(x_min - 0.015, 0.06, "0.0")
    plt.text(x_max - 0.025, 0.06, f"{np.round(x_max - 0.025, 2)}")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_outdir, "frames", f"{t}.png"))
    plt.close()

frames = os.listdir(frame_dir)
frames.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
frames = [os.path.join(frame_dir, f) for f in frames]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames, fps=100)
clip.write_videofile(os.path.join(viz_outdir, "sim.mp4"))

Video(os.path.join(viz_outdir, "sim.mp4"), width=800, height=400)
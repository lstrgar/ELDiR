import multiprocessing as mp, numpy as np, os
from .sim import simulate, learning_iters as iters
from utils.disk_utils import load_pop

def get_worker_indices(n_robots, n_workers):
    worker_indices = []
    chunk = n_robots // n_workers
    for i in range(n_workers):
        start = i*chunk
        end = (i+1)*chunk
        worker_indices.append((start, end))
    if end < n_robots:
        worker_indices[-1] = (start, n_robots)
    return worker_indices

def merge_losses(gen_dir, outfile, worker_indices, n_robots):
    global iters
    print(f"Merging losses for generation {gen_dir} to {outfile}", flush=True)
    n_workers = len(worker_indices)
    losses = []
    for i in range(n_workers):
        start, end = worker_indices[i]
        loss_file = os.path.join(gen_dir, f'loss_{start}-{end}.npy')
        assert os.path.exists(loss_file)
        losses.append(np.load(loss_file))
    losses = np.concatenate(losses)
    fitness = losses * -1
    assert fitness.shape[0] == n_robots
    assert fitness.shape[1] == iters+1
    np.save(outfile, fitness)

def simulate_pop(pop_file, gen_dir, device_ids, debug):
    pop = load_pop(pop_file)
    n_robots = len(pop["points"])
    n_workers = 1 if device_ids is None else len(device_ids)
    worker_type = "CPU" if device_ids is None else "GPU"
    print(f"Simulating {n_robots} robots from {pop_file} with {n_workers} {worker_type} process(es)")
    worker_indices = get_worker_indices(n_robots, n_workers)
    processes = []
    for i, (start, end) in enumerate(worker_indices):
        device_id = None if device_ids is None else device_ids[i]
        log_file = os.path.join(gen_dir, f"worker_{device_id}.log" if device_id is not None else "worker_CPU.log")
        seed = np.random.randint(0, np.iinfo(np.int32).max)
        args = (pop_file, gen_dir, device_id, log_file, start, end, seed, debug)
        p = mp.Process(target=simulate, args=args)
        processes.append(p)
        p.start()
        print(f"Started process idx:{i} / pid:{p.pid} / device:{device_id} / robots:{start}-{end}")
    for i, p in enumerate(processes):
        p.join()
        device_id = None if device_ids is None else device_ids[i]
        start, end = worker_indices[i]
        print(f"Joined process idx:{i} / pid:{p.pid} / device:{device_id} / robots:{start}-{end}")
    print("All processes joined")
    outfile = os.path.join(gen_dir, os.path.basename(pop_file).split(".")[0] + "-fitness.npy")
    merge_losses(gen_dir, outfile, worker_indices, n_robots)
    return np.load(outfile), outfile





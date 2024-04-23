from geno_pheno import mutate, random_robot_batch, next_gen, clean_weights
import argparse, os, numpy as np, shutil, multiprocessing as mp

def make_gen_dir(gen):
    ## Create directory for the next generation
    global outdir, n_robots, weights_dir_name
    gen_dir = os.path.join(outdir, str(gen))
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(os.path.join(gen_dir, weights_dir_name), exist_ok=True)
    for i in range(n_robots):
        os.makedirs(os.path.join(gen_dir, weights_dir_name, str(i)), exist_ok=True)
    return gen_dir

def clean_gen_dir(gen_dir):
    ## Clean files in partially completed generation directory
    global child_loss_file, weights_dir_name, n_robots
    print(f"Cleaning generation directory: {gen_dir}", flush=True)
    weights_dir = os.path.join(gen_dir, weights_dir_name)
    if os.path.exists(weights_dir):
        print(f"Removing {weights_dir}", flush=True)
        shutil.rmtree(weights_dir)
    child_loss = os.path.join(gen_dir, child_loss_file)
    if os.path.exists(child_loss):
        print(f"Removing {child_loss}", flush=True)
        os.remove(child_loss)
    print(f"Creating new {weights_dir}", flush=True)
    for i in range(n_robots):
        wd = os.path.join(gen_dir, weights_dir_name, str(i))
        if os.path.exists(wd):
            shutil.rmtree(wd)
        os.makedirs(wd, exist_ok=True)
    log_file = [f for f in os.listdir(gen_dir) if f.endswith('.log')]
    for f in log_file:
        print(f"Removing {f}", flush=True)
        os.remove(os.path.join(gen_dir, f))

def merge_losses(gen_dir, outfile):
    ## Merge losses from GPU workers
    global worker_indices, n_workers, n_robots, iters
    print(f"Merging losses for generation {gen_dir} to {outfile}", flush=True)
    losses = []
    for i in range(n_workers):
        start, end = worker_indices[i]
        loss_file = os.path.join(gen_dir, f'loss_{start}-{end}.npy')
        assert os.path.exists(loss_file)
        losses.append(np.load(loss_file))
    losses = np.concatenate(losses)
    assert losses.shape[0] == n_robots
    assert losses.shape[1] == iters+1
    np.save(outfile, losses)

def get_worker_indices():
    ## Divide robots among workers
    global n_robots, n_workers
    worker_indices = []
    chunk = n_robots // n_workers
    for i in range(n_workers):
        start = i*chunk
        end = (i+1)*chunk
        worker_indices.append((start, end))
    if end < n_robots:
        worker_indices[-1] = (start, n_robots)
    return worker_indices

def run_gpu_workers(robots_file, outdir):
    global gpu_ids, worker_indices, process_args

    processes = []

    ## For each GPU workers collect CLI args and start a new process
    for i, id in enumerate(gpu_ids):
        dev = ['--device_id', str(id)]
        od = ['--outdir', outdir]
        rf = ['--robots_file', robots_file]
        idx = ['--idx0', str(worker_indices[i][0]), '--idx1', str(worker_indices[i][1])]
        s = ['--seed', str(np.random.randint(0, np.iinfo(np.int32).max))]
        logfile = os.path.join(outdir, f'worker_{i}_gpu_{id}.log')
        lf = ['--logfile', logfile]
        args = process_args + dev + od + rf + idx + s + lf
        p = mp.Process(target=os.system, args=(' '.join(args),))
        p.start()
        processes.append(p)
        print(f"Process {i} started on GPU {id} with pid: {p.pid}", flush=True)
    
    ## Wait for all processes to finish
    for i, p in enumerate(processes):
        p.join()
        print(f"Process {i} joined", flush=True)

def loop(init_generation, init_gen_dir, robots, loss, child_robots, child_loss=None):
    global progbar, robots_fname, child_robots_fname, loss_file, child_loss_file

    generation = init_generation
    gen_dir = init_gen_dir

    try:
        ## Evolve until the end of time
        while True:
            print(f"GENERATION: {generation}", flush=True)

            ## Simulate / train / evalute the current generation
            run_gpu_workers(child_robots, gen_dir)

            ## Merge losses from GPU workers
            child_loss = os.path.join(gen_dir, child_loss_file)
            merge_losses(gen_dir, child_loss)

            ## Keep NN weights only for the best robots + random sample
            clean_weights(os.path.join(gen_dir, weights_dir_name), child_loss)

            ## Setup for the next generation
            generation += 1
            gen_dir = make_gen_dir(generation)
            next_robots = os.path.join(gen_dir, robots_fname)
            next_loss = os.path.join(gen_dir, loss_file)

            ## Compute the next generation of progenitors
            next_gen(loss, child_loss, robots, child_robots, next_robots, next_loss, progbar)

            ## Update variables for the next loop iteration
            robots = next_robots
            loss = next_loss
            child_robots = os.path.join(gen_dir, child_robots_fname)

            ## Mutate the progenitors to create offspring
            mutate(robots, child_robots, progbar)

    except KeyboardInterrupt:
        print("Keyboard interrupt, exiting...", flush=True)
        return

def main_from_ckpt(gen, gen_dir, robots, loss, child_robots):
    print(f"Initial generation: {gen}", flush=True)
    print(f"Initial generation dir: {gen_dir}", flush=True)
    print(f"Initial robots: {robots}", flush=True)
    print(f"Initial loss: {loss}", flush=True)
    print(f"Initial child robots: {child_robots}", flush=True)
    ## Clean files in partially completed generation directory
    clean_gen_dir(gen_dir)
    ## Run the main loop
    loop(
        gen, 
        gen_dir, 
        robots, 
        loss, 
        child_robots, 
    )

def main_from_0(gen0_dir):
    global n_robots, iters, robots_fname, child_robots_fname, progbar, loss_file, child_loss_file

    print("GENERATION: 0", flush=True)

    ## Simulate / train / evalute the initial generation
    robots = os.path.join(gen0_dir, robots_fname)
    run_gpu_workers(robots, gen0_dir)
    loss = os.path.join(gen0_dir, loss_file)
    merge_losses(gen0_dir, loss)
    shutil.copy(loss, os.path.join(gen0_dir, child_loss_file))

    ## Keep NN weights only for the best robots + random sample
    clean_weights(os.path.join(gen0_dir, weights_dir_name), os.path.join(gen0_dir, child_loss_file))

    ## Setup generation 1 with initial (generation 0) population and loss
    generation = 1
    gen_dir = make_gen_dir(generation)
    child_robots = os.path.join(gen_dir, child_robots_fname)
    shutil.copy(robots, os.path.join(gen_dir, robots_fname))
    robots = os.path.join(gen_dir, robots_fname)
    shutil.copy(loss, os.path.join(gen_dir, loss_file))
    loss = os.path.join(gen_dir, loss_file)

    ## Mutate the progenitors to create the first offspring
    mutate(robots, child_robots, progbar)

    ## Run the main loop
    loop(generation, gen_dir, robots, loss, child_robots)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--debug', default=False, action='store_true') ## Taichi debug mode: https://docs.taichi-lang.org/docs/debugging
    parser.add_argument('--outdir', type=str, default='run-out', help='Output directory (default: run-out)')
    parser.add_argument('--ckptdir', type=str, default=None, help='(load from) Checkpoint directory (default: None)')
    parser.add_argument('--gpu_ids', type=str, default="0", help='GPU IDs to parallelize on (comma separated). E.g. "0,1,2,3". (default: 0)')
    parser.add_argument('--n_robots', type=int, default=1000, help='Population size (default: 1000). Must be >= 100.')
    parser.add_argument('--iters', type=int, default=35, help='Number of learning iterations (default: 35)')
    parser.add_argument('--worker_script', type=str, default='./sim.py', help='GPU worker script (default: ./sim.py)')
    parser.add_argument('--no_progbar', default=False, action='store_true', help='Disable progress bar (default: False)')
    # parser.add_argument('--ground_file', type=str, default=None, help='IGNORE: not supported')

    options = parser.parse_args()

    ## Reboot from checkpoint...
    if options.ckptdir is not None:
        print("Rebooting from checkpoint: {}".format(options.ckptdir), flush=True)

        ## Load initial CLI args from checkpoint and merge with current CLI args
        ckptdir = options.ckptdir
        with open(os.path.join(options.ckptdir, 'args.txt'), 'r') as f:
            arg_options = eval(f.read())
            arg_options['outdir'] = ckptdir
            arg_options['debug'] = options.debug
            arg_options['no_progbar'] = options.no_progbar
            arg_options['gpu_ids'] = options.gpu_ids
        options.__dict__ = arg_options

        ## Find the latest generation
        dir_contents = os.listdir(ckptdir)
        gens = [int(d) for d in dir_contents if d.isdigit()]
        gen = max(gens)
    
    ## ...or start from scratch
    else:
        ckptdir = None
        ## Create output directory and save CLI args
        os.makedirs(options.outdir, exist_ok=True)
        with open(os.path.join(options.outdir, 'args.txt'), 'w') as f:
            f.write(str(options.__dict__) + '\n')

    ## Set up global variables 
    global n_robots, iters, progbar, outdir, debug
    n_robots = options.n_robots
    iters = options.iters
    progbar = not options.no_progbar
    outdir = options.outdir
    debug = options.debug

    ## Constant filenames to be used throughout
    global robots_fname, child_robots_fname, loss_file, child_loss_file, weights_dir_name
    robots_fname = "robots.pkl"
    child_robots_fname = "child_robots.pkl"
    loss_file = "loss.npy"
    child_loss_file = "child_loss.npy"
    weights_dir_name = "weights"

    ## Setup GPU worker params
    global n_workers, gpu_ids, worker_indices, worker_script, process_args
    gpu_ids = [int(id) for id in options.gpu_ids.split(",")]
    n_workers = len(gpu_ids)
    worker_indices = get_worker_indices()
    worker_script = options.worker_script
    process_args = ['python', worker_script, '--iters', str(options.iters)]
    if options.debug:
        process_args.append('--debug')
    # if options.ground_file is not None:
    #     process_args += ['--ground_file', options.ground_file]

    ## Seed numpy RNG
    np.random.seed(options.seed)

    ## Run main loop from checkpoint...
    if ckptdir is not None:
        main_from_ckpt(
            gen, 
            os.path.join(ckptdir, str(gen)),
            os.path.join(ckptdir, str(gen), robots_fname),
            os.path.join(ckptdir, str(gen), loss_file),
            os.path.join(ckptdir, str(gen), child_robots_fname),
        )
    ## ...or from scratch
    else:
        gen0_dir = make_gen_dir(0)
        robots_file = os.path.join(gen0_dir, robots_fname)
        random_robot_batch(options.n_robots, robots_file, progbar)
        main_from_0(gen0_dir)
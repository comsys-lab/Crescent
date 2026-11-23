"""Multi-process driver for weight_gptq.gptq_quantize_weight.

gptq_quantize_weight is latency-bound (hundreds of tiny kernels / Python steps per matrix:
~3-5 s for an NVFP4 MoE-expert matrix on a GPU that is otherwise idle), and threads only gave
~1.2-1.4x because of the GIL. This runs the UNCHANGED function in several worker processes,
`procs_per_device` per GPU, each job = (key, weight, hessian). The GPTQ/OBS math, the scale
search and the ill-conditioning guard are exactly weight_gptq's; only WHEN each independent
matrix runs changes. Jobs are pulled lazily from a generator (so the caller can pop each
Hessian/weight as it is dispatched and memory falls while the run progresses).

run_jobs(jobs, fmt, n_devices, procs_per_device) -> ({key: quantized cpu tensor}, {key: rel_err})
"""
import os
import sys
import threading

import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))


def _worker(dev_index, fmt, in_q, out_q):
    os.environ["GPTQ_CPU"] = "0"  # GPTQ math on the worker's GPU (weight.device)
    torch.multiprocessing.set_sharing_strategy("file_system")
    import weight_gptq as W  # after GPTQ_CPU is set: it is read at import time
    if os.environ.get("GPTQ_WORKER_CPU") == "1":  # tests: no GPU
        dev = torch.device("cpu")
    else:
        dev = torch.device(f"cuda:{dev_index}")
        torch.cuda.set_device(dev)
    while True:
        item = in_q.get()
        if item is None:
            out_q.put(("done", None, None))
            return
        key, w, h = item
        try:
            wd = w.to(dev)
            q = W.gptq_quantize_weight(wd, h, fmt)
            w32, q32 = wd.float(), q.float()
            rel = float(((w32 - q32) ** 2).sum() / (w32 ** 2).sum().clamp(min=1e-30))
            out_q.put(("ok", key, (q.to(w.dtype).cpu(), rel)))
        except Exception as e:  # report, never hang the parent
            out_q.put(("err", key, repr(e)))


def run_jobs(jobs, fmt: str, n_devices: int, procs_per_device: int, log_every: int = 200):
    torch.multiprocessing.set_sharing_strategy("file_system")
    ctx = mp.get_context("spawn")
    n_workers = n_devices * procs_per_device
    in_q = ctx.Queue(maxsize=n_workers * 2)
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(i % n_devices, fmt, in_q, out_q), daemon=True) for i in range(n_workers)]
    for p in procs:
        p.start()

    n_sent = {"n": 0}

    def feed():
        for job in jobs:
            in_q.put(job)
            n_sent["n"] += 1
        for _ in procs:
            in_q.put(None)

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()

    results, rels, errors = {}, {}, []
    done = 0
    while done < n_workers:
        kind, key, payload = out_q.get()
        if kind == "done":
            done += 1
        elif kind == "ok":
            results[key] = payload[0]
            rels[key] = payload[1]
            if len(results) % log_every == 0:
                print(f"  [gptq_parallel] {len(results)} matrices done", flush=True)
        else:
            errors.append((key, payload))
    feeder.join()
    for p in procs:
        p.join()
    if errors:
        raise RuntimeError(f"{len(errors)} GPTQ jobs failed, e.g. {errors[:3]}")
    if len(results) != n_sent["n"]:
        raise RuntimeError(f"lost jobs: sent {n_sent['n']}, got {len(results)}")
    return results, rels

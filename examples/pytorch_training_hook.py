"""Instrument a real PyTorch DDP training loop so its steps can be attributed.

Launch (one process per GPU):

    torchrun --nproc-per-node 8 examples/pytorch_training_hook.py

Alongside, on every host:

    nicprof collect --interval 0.25 --source rdma --out /data/counters.$(hostname).jsonl

Afterwards:

    cat /data/counters.*.jsonl > counters.jsonl
    nicprof analyze --counters counters.jsonl --steps /data/steps.rank*.jsonl --topology topology.yaml
"""

from __future__ import annotations

import os

from ai_nic_perf_profiler.training import StepRecorder


def main() -> None:
    import torch
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank, local_rank = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(8192, 8192).cuda(), device_ids=[local_rank])
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    grad_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # sync=torch.cuda.synchronize makes the host-side step boundary match the
    # GPU-side one, so the step window lines up with the NIC traffic it caused.
    with StepRecorder(f"/data/steps.rank{rank}.jsonl", rank=rank, sync=torch.cuda.synchronize) as rec:
        for _ in range(500):
            with rec.step(comm_bytes=grad_bytes):
                x = torch.randn(64, 8192, device="cuda")
                loss = model(x).square().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()  # DDP all-reduces gradients over NCCL here
                opt.step()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

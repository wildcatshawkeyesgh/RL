import subprocess

import torch


def get_best_gpu(strategy="utilization"):
    if strategy == "memory":
        free_mem = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.mem_get_info(i)
            free_mem.append(props[0])
        return free_mem.index(max(free_mem))

    elif strategy == "utilization":
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )

        utilizations = [int(x.strip()) for x in result.stdout.strip().split("\n")]
        return utilizations.index(min(utilizations))

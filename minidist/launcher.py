import os 
import subprocess
import sys
import time
import uuid


def _stop_processes(processes: list[subprocess.Popen]) -> None:
    running = [process for process in processes if process.poll() is None]
    for process in running:
        process.terminate()

    deadline = time.monotonic() + 5.0
    for process in running:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass

        if process.poll() is None:
            process.kill()

    for process in running:
        process.wait()

def launch(
    nproc_per_node: int,
    script: str,
    script_args: list[str] | None = None,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    run_id: str | None = None,
) -> int:
    # for now only one node 
    node_rank = 0
    world_size = nproc_per_node
    script_args = script_args or []
    run_id = run_id or uuid.uuid4().hex
    processes: list[subprocess.Popen] = []

    if nproc_per_node < 1:
        raise ValueError("nproc_per_node must be at least 1")

    try:
        for local_rank in range(nproc_per_node):
            rank = node_rank * nproc_per_node + local_rank
            env = os.environ.copy()
            env.update({
                "RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "LOCAL_RANK": str(local_rank),
                "LOCAL_WORLD_SIZE": str(nproc_per_node),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
                "RUN_ID": run_id,
            })

            processes.append(subprocess.Popen(
                [sys.executable, "-u", script, *script_args],
                env=env,
            ))

        running = set(processes)
        first_failure = 0
        while running:
            for process in list(running):
                return_code = process.poll()
                if return_code is None:
                    continue

                running.remove(process)
                if return_code != 0 and first_failure == 0:
                    first_failure = return_code
                    _stop_processes(list(running))
                    running.clear()

            if running:
                time.sleep(0.05)

        return first_failure
    except BaseException:
        _stop_processes(processes)
        raise
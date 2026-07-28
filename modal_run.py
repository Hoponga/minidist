import subprocess

import modal

app = modal.App("nccl-example")

N_DEVICES = 4

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .run_commands(
        "apt-get update && "
        "apt-get install -y --allow-change-held-packages libnccl2 libnccl-dev"
    )
    .add_local_file("nccl_example.cu", remote_path="/root/example3.cu", copy=True)
    .run_commands("nvcc -O3 -o /root/example3 /root/example3.cu -lnccl")
)


@app.function(image=image, gpu=f"A100:{N_DEVICES}")
def run_nccl_allreduce():
    result = subprocess.run(
        ["/root/example3", str(N_DEVICES)],
        capture_output=True,
        text=True,
        check=True,
    )
    print(result.stdout)
    return result.stdout


@app.local_entrypoint()
def main():
    run_nccl_allreduce.remote()

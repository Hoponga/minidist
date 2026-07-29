import subprocess

import modal

app = modal.App("minidist")

N_DEVICES = 4

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.13"
    )
    .run_commands(
        "apt-get update && "
        "apt-get install -y --allow-change-held-packages "
        "build-essential libnccl2 libnccl-dev ninja-build"
    )
    .pip_install("uv")
    .add_local_dir(
        "minidist",
        remote_path="/root/minidist/minidist",
        copy=True,
    )
    .add_local_dir("csrc", remote_path="/root/minidist/csrc", copy=True)
    .add_local_file(
        "example.py",
        remote_path="/root/minidist/example.py",
        copy=True,
    )
    .add_local_file(
        "pyproject.toml",
        remote_path="/root/minidist/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        "setup.py",
        remote_path="/root/minidist/setup.py",
        copy=True,
    )
    .add_local_file(
        "uv.lock",
        remote_path="/root/minidist/uv.lock",
        copy=True,
    )
    .run_commands(
        "cd /root/minidist && uv sync --frozen --no-install-project",
        "cd /root/minidist && CXX=g++ CC=gcc uv run python setup.py build_ext --inplace",
    )
)


@app.function(image=image, gpu=f"A100:{N_DEVICES}", timeout=900)
def run_minidist():
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "minidist.run",
            f"--nproc-per-node={N_DEVICES}",
            "example.py",
        ],
        cwd="/root/minidist",
        check=True,
    )


@app.local_entrypoint()
def main():
    run_minidist.remote()

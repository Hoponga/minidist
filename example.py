import torch
import minidist as dist


def main():
    dist.init_process_group(backend="nccl")


if __name__ == "__main__":
    main()

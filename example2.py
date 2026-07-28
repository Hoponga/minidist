import torch
import minidist as dist


def main():
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2

    if rank == 0:
        tensor = torch.tensor([42])
        print(f"[rank 0] sending {tensor.item()} to rank 1")
        dist.send(tensor, dst=1)
    else:
        tensor = torch.zeros(1)
        dist.recv(tensor, src=0)
        print(f"[rank 1] received {tensor.item()} from rank 0")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

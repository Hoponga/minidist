import torch
import minidist as dist


def main():
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size() 
    device = dist.get_device()
    print(f"rank: {rank}, world_size: {world_size}, device: {device}")
    a = torch.tensor([rank], dtype=torch.int32, device=device)
    dist.all_reduce(a, op=dist.ReduceOp.SUM)
    print(f"a: {a}")
    print(f"a: {a.item()}")

if __name__ == "__main__":
    main()

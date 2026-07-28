import torch
import minidist as dist


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tensor = torch.tensor([rank + 1.0]).to("cuda")
    print(f"[rank {rank}] before all_reduce: {tensor.item()}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[rank {rank} sees tensor {tensor.item()}]")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

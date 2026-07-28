from .enums import ReduceOp
import os
from .store import * 
from minidist import _C # pybinds 

_default_group = None
_store = None
_store_server = None       # non-None only on rank 0
_run_id = None
_local_rank = None
_next_group_sequence = 0


class _NonGroupMember:
    pass


NON_GROUP_MEMBER = _NonGroupMember()


def init_process_group(
    backend: str = "nccl",
    rank: int | None = None,
    world_size: int | None = None,
    local_rank: int | None = None,
    master_addr: str | None = None,
    master_port: int | None = None,
    timeout_s: float = 300.0,
) -> None:
    global _default_group, _store, _store_server 
    rank = rank if rank is not None else int(os.environ["RANK"])
    world_size = (world_size if world_size is not None else int(os.environ["WORLD_SIZE"]))

    local_rank = (local_rank if local_rank is not None else int(os.environ["LOCAL_RANK"]))

    master_addr = master_addr or os.environ["MASTER_ADDR"]
    master_port = master_port or int(os.environ["MASTER_PORT"])
    run_id = os.environ.get("RUN_ID", "default")

    if rank == 0: 
        # store server runs on rank 0 
        _store_server = TCPStoreServer("0.0.0.0", master_port)
        _store_server.start() 

    _store = TCPStoreClient(master_addr, master_port, timeout_s)
    prefix = f"/{run_id}"

    _store.set(f"{prefix}/members/{rank}", b"ready")
    _store.wait([
        f"{prefix}/members/{r}"
        for r in range(world_size)
    ])

    uid_key = f"{prefix}/groups/world/nccl_id"
    if rank == 0:
        _store.set(uid_key, _C.get_nccl_unique_id())


    unique_id = _store.get(uid_key)
    _default_group = _C.ProcessGroupNCCL(unique_id = unique_id, global_rank = rank, group_rank = rank, global_ranks = list(range(world_size)), device = local_rank)






def destroy_process_group(group=None) -> None:
    pass


def is_initialized() -> bool:
    pass


def get_rank(group=None) -> int:
    pass


def get_world_size(group=None) -> int:
    pass


def new_group(ranks: list[int]):
    pass


def all_reduce(
    tensor,
    op: ReduceOp = ReduceOp.SUM,
    group=None,
    async_op: bool = False,
):
    pass


def broadcast(
    tensor,
    src: int,
    group=None,
    async_op: bool = False,
):
    pass


def reduce(
    tensor,
    dst: int,
    op: ReduceOp = ReduceOp.SUM,
    group=None,
    async_op: bool = False,
):
    pass


def all_gather_into_tensor(
    output_tensor,
    input_tensor,
    group=None,
    async_op: bool = False,
):
    pass


def reduce_scatter_tensor(
    output_tensor,
    input_tensor,
    op: ReduceOp = ReduceOp.SUM,
    group=None,
    async_op: bool = False,
):
    pass


def barrier(group=None) -> None:
    pass

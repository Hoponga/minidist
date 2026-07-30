from .enums import ReduceOp
import os
from .store import * 


_default_group = None
_store = None
_store_server = None       # non-None only on rank 0
_run_id = None
_local_rank = None
_rank = None
_world_size = None
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
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> None:
    import minidist._C as _C # pybinds
    global _default_group, _store, _store_server, _rank, _world_size, _local_rank
    rank = rank if rank is not None else int(os.environ["RANK"])
    world_size = (world_size if world_size is not None else int(os.environ["WORLD_SIZE"]))

    local_rank = (local_rank if local_rank is not None else int(os.environ["LOCAL_RANK"]))

    _rank = rank
    _world_size = world_size
    _local_rank = local_rank
    master_addr = master_addr or os.environ["MASTER_ADDR"]
    master_port = master_port or int(os.environ["MASTER_PORT"])
    run_id = os.environ.get("RUN_ID", "default")

    if rank == 0: 
        # store server runs on rank 0 
        _store_server = TCPStoreServer("0.0.0.0", master_port, timeout_s=timeout_s)
        print(f"rank {rank} starting store server on port {master_port}")
        _store_server.start() 

    print(f"rank {rank} starting store client on port {master_port}")
    _store = TCPStoreClient(master_addr, master_port, timeout_s)
    print(f"rank {rank} connected to store server on {master_addr}:{master_port}")
    prefix = f"/{run_id}"

    _store.set(f"{prefix}/members/{rank}", b"ready")
    _store.wait([
        f"{prefix}/members/{r}"
        for r in range(world_size)
    ])

    uid_key = f"{prefix}/groups/world/nccl_id"

    if rank == 0:
        unique_id = _C.get_nccl_unique_id()
        _store.set(uid_key, unique_id)


    unique_id = _store.get(uid_key)

    print(f"rank {rank} has received unique_id: {unique_id}")
    _default_group = _C.ProcessGroupNCCL(unique_id = unique_id, global_rank = rank, group_rank = rank, global_ranks = list(range(world_size)), device = local_rank)





def get_device(group=None) -> str: 
    if _local_rank is None:
        raise ValueError("local_rank is not set")
    return f"cuda:{_local_rank}"

def destroy_process_group(group=None) -> None:
    pass


def is_initialized() -> bool:
    pass


def get_rank(group=None) -> int:
    return _rank


def get_world_size(group=None) -> int:
    return _world_size


def new_group(ranks: list[int]):
    pass


def convert_cpp_op(op: ReduceOp | str): 
    import minidist._C as _C 

    if isinstance(op, _C.ReduceOp):
        return op

    op_name = op.value if isinstance(op, ReduceOp) else op
    op_map = {
        "sum": _C.ReduceOp.SUM,
        "product": _C.ReduceOp.PRODUCT,
        "min": _C.ReduceOp.MIN,
        "max": _C.ReduceOp.MAX,
        "avg": _C.ReduceOp.AVG,
    }

    try:
        return op_map[op_name.lower()]
    except (AttributeError, KeyError):
        raise ValueError(f"unsupported reduction op: {op}") from None

def all_reduce(
    tensor,
    op: ReduceOp | str = ReduceOp.SUM,
    group=None,
    async_op: bool = False,
):
    op = convert_cpp_op(op)
    if group: 
        group.all_reduce(tensor, op, async_op) 
    else: 

        _default_group.all_reduce(tensor, op, async_op)


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

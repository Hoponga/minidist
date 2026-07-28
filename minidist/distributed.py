from .enums import ReduceOp

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
    pass


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

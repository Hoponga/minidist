# NCCLDist: A Small, Readable Reimplementation of `torch.distributed`

## 1. Goal

`nccldist` is a learning project that reimplements the core ideas behind `torch.distributed` for CUDA tensors.

It is intentionally small:

- one process per GPU;
- static ranks;
- a rank-0 TCP key-value store for coordination;
- one C++ `ProcessGroupNCCL` object per communication group;
- NCCL for GPU tensor communication;
- a thin Python API that resembles `torch.distributed`.

The project is not intended to be production ready. The code should make the control plane, communicator setup, CUDA stream semantics, and collective calls easy to follow.

The central split is:

```text
Control plane: rank-0 TCPStore
  - workers join the job
  - NCCL unique IDs are exchanged
  - subgroup metadata is exchanged
  - initialization barriers are implemented

Data plane: NCCL
  - GPU tensors move directly between ranks
  - all-reduce, broadcast, gather, reduce-scatter, send, and recv
```

Tensor contents never pass through the key-value store.

---

## 2. Supported scope

### Version 1

- Linux and CUDA.
- One Python process per GPU.
- `torch.Tensor` as the tensor object.
- Contiguous CUDA tensors only.
- Static `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`.
- A default world process group.
- Optional subgroups.
- Synchronous collectives first.
- Optional CUDA-event-backed asynchronous work later.

### Deliberately omitted

- Elastic membership.
- Recovery after a rank dies.
- CPU collectives.
- Multiple GPUs controlled by one process.
- Noncontiguous tensor packing.
- DistributedDataParallel and autograd hooks.
- Watchdogs, flight recorders, and advanced diagnostics.
- Exact compatibility with every `torch.distributed` behavior.

---

## 3. Overall architecture

```text
+-----------------------------------------------------------+
| User program                                              |
|                                                           |
| dist.init_process_group()                                 |
| dist.all_reduce(x)                                        |
| dist.new_group([0, 1, 2, 3])                              |
+-----------------------------+-----------------------------+
                              |
                              v
+-----------------------------------------------------------+
| Python facade: nccldist/distributed.py                    |
|                                                           |
| - reads rank environment variables                        |
| - starts the Store server on rank 0                       |
| - creates Store clients on every rank                     |
| - owns the default process group                          |
| - forwards tensor operations to C++                       |
+----------------------+----------------------+-------------+
                       |                      |
                       | control plane        | pybind11
                       v                      v
+--------------------------------+   +-----------------------+
| Rank-0 TCPStore                |   | ProcessGroupNCCL      |
|                                |   |                       |
| set/get/add/wait               |   | ncclComm_t            |
| NCCL unique IDs                |   | group rank mapping    |
| subgroup metadata              |   | collective methods    |
| initialization barriers        |   | tensor validation     |
+--------------------------------+   +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | NCCL + CUDA           |
                                    | GPU tensor data plane |
                                    +-----------------------+
```

Rank 0 hosts the Store server in a background thread. Every rank, including rank 0, uses the same Store client API.

---

## 4. Process launch and environment

A launcher starts one process per GPU and sets:

```text
MASTER_ADDR=hostname-of-rank-0
MASTER_PORT=29500
RANK=0,1,...,WORLD_SIZE-1
WORLD_SIZE=number-of-processes
LOCAL_RANK=GPU-index-on-this-machine
RUN_ID=a-unique-string-for-this-job
```

For example, on two nodes with four GPUs each:

```text
Node 0:
  global ranks 0,1,2,3
  local ranks  0,1,2,3

Node 1:
  global ranks 4,5,6,7
  local ranks  0,1,2,3
```

`LOCAL_RANK`, not global `RANK`, is passed to `cudaSetDevice`.

A small single-node launcher can use `subprocess.Popen` to create child processes and assign these variables. Multi-node launching can be done manually, through Slurm, or through another process launcher. The communication library itself only depends on the environment variables.

---

## 5. Rank-0 key-value Store

The Store is the small coordination service that surrounds NCCL.

### 5.1 Store API

Keep the interface minimal:

```python
class Store:
    def set(self, key: str, value: bytes) -> None: ...

    def get(
        self,
        key: str,
        timeout_s: float | None = None,
    ) -> bytes: ...

    def add(
        self,
        key: str,
        delta: int,
    ) -> int: ...

    def wait(
        self,
        keys: list[str],
        timeout_s: float | None = None,
    ) -> None: ...
```

Semantics:

- `set` inserts or overwrites a byte value.
- `get` blocks until the key exists or the timeout expires.
- `add` atomically increments an integer value and returns the new value.
- `wait` blocks until every requested key exists.

That is enough for communicator setup, simple barriers, and subgroup creation.

### 5.2 Server design

Rank 0 starts:

```python
TCPStoreServer(
    bind_host="0.0.0.0",
    port=MASTER_PORT,
)
```

Internally, the server has:

```python
values: dict[str, bytes]
condition: threading.Condition
```

Each request handler executes operations while holding the condition lock. A successful `set` or `add` calls `condition.notify_all()` so blocked `get` and `wait` requests wake up.

Use `socketserver.ThreadingTCPServer` for readability. A simple length-prefixed JSON protocol is sufficient:

```text
4-byte request length
JSON request
4-byte response length
JSON response
```

Byte values can be base64 encoded in JSON. This is not the fastest protocol, but Store traffic is tiny and infrequent.

### 5.3 Client design

Every rank creates:

```python
store = TCPStoreClient(
    host=MASTER_ADDR,
    port=MASTER_PORT,
    timeout_s=300,
)
```

Nonzero ranks retry connecting until rank 0 has started listening or the timeout expires.

Rank 0 should also use a normal client connection rather than directly accessing the server dictionary. This keeps all ranks on the same code path.

### 5.4 Key namespace

Prefix every key with `RUN_ID`:

```text
/<run_id>/members/<rank>
/<run_id>/groups/world/nccl_id
/<run_id>/groups/<group_id>/nccl_id
/<run_id>/groups/<group_id>/members
/<run_id>/barriers/<barrier_id>/<rank>
/<run_id>/shutdown/<rank>
```

This prevents metadata from different jobs from being mixed together.

### 5.5 What the Store is not

The Store does not hold model parameters, gradients, activations, or tensor payloads.

For an all-reduce:

```text
Store traffic: none
NCCL traffic: the entire tensor
```

The Store is only used when creating or coordinating process groups.

---

## 6. Initialization flow

The default world process group is initialized as follows:

```text
launcher sets rank environment variables
        |
        v
rank 0 starts TCPStore server thread
        |
        v
all ranks connect as Store clients
        |
        v
rank r writes /members/r
        |
        v
all ranks wait for all membership keys
        |
        v
rank 0 calls ncclGetUniqueId()
        |
        v
rank 0 stores the ID under /groups/world/nccl_id
        |
        v
all ranks read the same ID
        |
        v
cudaSetDevice(LOCAL_RANK)
        |
        v
ncclCommInitRank(world_size, id, rank)
        |
        v
construct ProcessGroupNCCL
        |
        v
save it as the Python default process group
```

Python-level pseudocode:

```python
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
    world_size = (
        world_size
        if world_size is not None
        else int(os.environ["WORLD_SIZE"])
    )
    local_rank = (
        local_rank
        if local_rank is not None
        else int(os.environ["LOCAL_RANK"])
    )

    master_addr = master_addr or os.environ["MASTER_ADDR"]
    master_port = master_port or int(os.environ["MASTER_PORT"])
    run_id = os.environ.get("RUN_ID", "default")

    if rank == 0:
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

    _default_group = _C.ProcessGroupNCCL(
        unique_id=unique_id,
        global_rank=rank,
        group_rank=rank,
        global_ranks=list(range(world_size)),
        device=local_rank,
    )
```

The pybind11 constructor should release the Python GIL around `ncclCommInitRank`. This allows rank 0's Python Store server thread to continue serving requests while the main thread waits for communicator initialization.

---

## 7. Public Python API

```python
def init_process_group(
    backend: str = "nccl",
    rank: int | None = None,
    world_size: int | None = None,
    local_rank: int | None = None,
    master_addr: str | None = None,
    master_port: int | None = None,
    timeout_s: float = 300.0,
) -> None: ...


def destroy_process_group(group=None) -> None: ...


def is_initialized() -> bool: ...

def get_rank(group=None) -> int: ...

def get_world_size(group=None) -> int: ...


def new_group(ranks: list[int]): ...


def all_reduce(
    tensor,
    op=ReduceOp.SUM,
    group=None,
    async_op: bool = False,
): ...


def broadcast(
    tensor,
    src: int,
    group=None,
    async_op: bool = False,
): ...


def reduce(
    tensor,
    dst: int,
    op=ReduceOp.SUM,
    group=None,
    async_op: bool = False,
): ...


def all_gather_into_tensor(
    output_tensor,
    input_tensor,
    group=None,
    async_op: bool = False,
): ...


def reduce_scatter_tensor(
    output_tensor,
    input_tensor,
    op=ReduceOp.SUM,
    group=None,
    async_op: bool = False,
): ...


def barrier(group=None) -> None: ...
```

Reduction operations:

```python
class ReduceOp(Enum):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    AVG = 4
```

The Python facade owns:

```python
_default_group = None
_store = None
_store_server = None       # Non-None only on rank 0
_next_group_sequence = 0
```

Communication logic stays in C++. Store and lifecycle coordination stay in Python.

---

## 8. C++ `ProcessGroupNCCL`

This is the central data-plane object.

```cpp
class ProcessGroupNCCL {
public:
    ProcessGroupNCCL(
        const std::string& unique_id_bytes,
        int global_rank,
        int group_rank,
        std::vector<int> global_ranks,
        int device);

    ~ProcessGroupNCCL();

    int rank() const;
    int size() const;

    std::shared_ptr<WorkNCCL> all_reduce(
        torch::Tensor tensor,
        ReduceOp op,
        bool async_op);

    std::shared_ptr<WorkNCCL> broadcast(
        torch::Tensor tensor,
        int global_src,
        bool async_op);

    std::shared_ptr<WorkNCCL> reduce(
        torch::Tensor tensor,
        int global_dst,
        ReduceOp op,
        bool async_op);

    std::shared_ptr<WorkNCCL> all_gather_into_tensor(
        torch::Tensor output,
        torch::Tensor input,
        bool async_op);

    std::shared_ptr<WorkNCCL> reduce_scatter_tensor(
        torch::Tensor output,
        torch::Tensor input,
        ReduceOp op,
        bool async_op);

    void barrier();
    void destroy();

private:
    int global_rank_;
    int group_rank_;
    int device_;

    std::vector<int> global_ranks_;
    ncclComm_t comm_ = nullptr;

    std::mutex launch_mutex_;

    int to_group_rank(int global_rank) const;
    void validate_tensor(const torch::Tensor& tensor) const;
};
```

The process group owns exactly one NCCL communicator. There should be no global `ncclComm_t` variable.

### Tensor validation

Every collective checks:

```cpp
TORCH_CHECK(tensor.is_cuda(), "tensor must be CUDA");
TORCH_CHECK(tensor.is_contiguous(), "tensor must be contiguous");
TORCH_CHECK(
    tensor.get_device() == device_,
    "tensor is on the wrong CUDA device");
```

Map PyTorch dtypes to NCCL dtypes in one helper function.

---

## 9. Collective execution

For the first implementation, launch NCCL on PyTorch's current CUDA stream and synchronize before returning.

```cpp
std::shared_ptr<WorkNCCL>
ProcessGroupNCCL::all_reduce(
    torch::Tensor tensor,
    ReduceOp op,
    bool async_op) {

    validate_tensor(tensor);
    c10::cuda::CUDAGuard guard(tensor.device());

    cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(device_).stream();

    std::lock_guard<std::mutex> lock(launch_mutex_);

    NCCL_CHECK(ncclAllReduce(
        tensor.data_ptr(),
        tensor.data_ptr(),
        static_cast<size_t>(tensor.numel()),
        to_nccl_dtype(tensor.scalar_type()),
        to_nccl_op(op),
        comm_,
        stream));

    if (!async_op) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return nullptr;
    }

    return WorkNCCL::record(stream, tensor);
}
```

This produces an easy-to-understand ordering:

```text
current CUDA stream:

produce tensor -> NCCL collective -> synchronize -> return
```

NCCL counts are element counts, not byte counts:

```cpp
// Correct
ncclBroadcast(ptr, ptr, tensor.numel(), ncclFloat, ...);

// Incorrect
ncclBroadcast(
    ptr,
    ptr,
    tensor.numel() * sizeof(float),
    ncclFloat,
    ...);
```

Native shape requirements:

```text
all_gather output.numel()
    == input.numel() * group_size

reduce_scatter input.numel()
    == output.numel() * group_size
```

---

## 10. Optional asynchronous work

After synchronous collectives work, add:

```cpp
class WorkNCCL {
public:
    static std::shared_ptr<WorkNCCL> record(
        cudaStream_t stream,
        std::vector<torch::Tensor> tensors);

    bool is_completed();
    void wait();

private:
    int device_;
    cudaEvent_t done_;
    std::vector<torch::Tensor> tensors_;
};
```

The CUDA event is recorded after the NCCL operation. The tensor references prevent the allocator from reusing their memory while communication is still active.

For the learning implementation, `wait()` may simply call:

```cpp
cudaEventSynchronize(done_);
```

This blocks the CPU but keeps the implementation clear.

---

## 11. Subgroups through the Store

A subgroup is another NCCL communicator with a subset of global ranks.

```python
tp_group = dist.new_group([0, 1, 2, 3])
dp_group = dist.new_group([0, 4])
```

Require every world rank to call `new_group` in the same order.

Python pseudocode:

```python
def new_group(ranks: list[int]):
    global _next_group_sequence

    ranks = list(ranks)
    sequence = _next_group_sequence
    _next_group_sequence += 1

    group_id = f"group-{sequence}"
    prefix = f"/{_run_id}/groups/{group_id}"

    if get_rank() == 0:
        _store.set(
            f"{prefix}/members",
            json.dumps(ranks).encode(),
        )
        _store.set(
            f"{prefix}/nccl_id",
            _C.get_nccl_unique_id(),
        )

    stored_ranks = json.loads(
        _store.get(f"{prefix}/members")
    )

    if stored_ranks != ranks:
        raise RuntimeError("new_group rank lists differ")

    unique_id = _store.get(f"{prefix}/nccl_id")

    global_rank = get_rank()
    if global_rank not in ranks:
        return NON_GROUP_MEMBER

    group_rank = ranks.index(global_rank)

    return _C.ProcessGroupNCCL(
        unique_id=unique_id,
        global_rank=global_rank,
        group_rank=group_rank,
        global_ranks=ranks,
        device=_local_rank,
    )
```

Rank 0 may generate the NCCL unique ID even when rank 0 is not a subgroup member. Generating the opaque ID does not initialize or join the communicator.

For a group with global ranks:

```text
[2, 5, 7, 9]
```

NCCL communicator ranks are:

```text
global 2 -> group rank 0
global 5 -> group rank 1
global 7 -> group rank 2
global 9 -> group rank 3
```

The Python API accepts global roots. `ProcessGroupNCCL` translates them before calling NCCL.

---

## 12. Shutdown

A simple graceful shutdown uses the Store only for coordination:

```text
destroy local NCCL communicator
        |
        v
rank r writes /shutdown/r
        |
        v
all ranks wait for all shutdown keys
        |
        v
rank 0 stops the TCPStore server
```

Python sketch:

```python
def destroy_process_group(group=None):
    global _default_group, _store, _store_server

    pg = _default_group if group is None else group
    pg.destroy()

    if group is None:
        rank = get_rank()
        _store.set(f"/{_run_id}/shutdown/{rank}", b"done")
        _store.wait([
            f"/{_run_id}/shutdown/{r}"
            for r in range(get_world_size())
        ])

        if rank == 0:
            _store_server.stop()

        _default_group = None
        _store = None
        _store_server = None
```

This assumes a failure-free educational setting. If a rank crashes, shutdown can time out and the job should be terminated manually.

---

## 13. Repository layout

```text
nccldist/
|-- pyproject.toml
|-- setup.py
|
|-- python/
|   `-- nccldist/
|       |-- __init__.py
|       |-- distributed.py
|       |-- store.py
|       |-- launcher.py
|       `-- enums.py
|
|-- csrc/
|   |-- bindings.cpp
|   |-- process_group_nccl.h
|   |-- process_group_nccl.cpp
|   |-- work_nccl.h
|   |-- work_nccl.cpp
|   |-- tensor_utils.h
|   `-- error.h
|
|-- examples/
|   |-- all_reduce.py
|   |-- broadcast.py
|   `-- subgroups.py
|
`-- tests/
    |-- test_store.py
    |-- test_init.py
    |-- test_all_reduce.py
    |-- test_broadcast.py
    |-- test_subgroups.py
    `-- test_async.py
```

Responsibilities:

| File | Purpose |
|---|---|
| `store.py` | Rank-0 Store server, Store client, and RPC protocol |
| `launcher.py` | Starts one process per GPU and sets rank variables |
| `distributed.py` | Public API, Store lifecycle, and default-group registry |
| `bindings.cpp` | pybind11 bindings and GIL release around NCCL initialization |
| `process_group_nccl.cpp` | Communicator ownership and collective calls |
| `work_nccl.cpp` | Optional CUDA-event-backed async handle |
| `tensor_utils.h` | Tensor validation and dtype conversion |
| `error.h` | CUDA and NCCL error-checking helpers |

---

## 14. Implementation order

### Milestone 1: Store

Implement and test:

```text
set
get
add
wait
multiple clients
get timeout
```

Run all clients on one machine first.

### Milestone 2: raw NCCL world group

Implement:

```text
rank 0 writes ncclUniqueId to Store
all ranks retrieve it
cudaSetDevice(local_rank)
ncclCommInitRank
one synchronous all-reduce
```

### Milestone 3: Python API

Add:

```text
init_process_group
get_rank
get_world_size
all_reduce
destroy_process_group
```

### Milestone 4: fixed-size collectives

Add:

```text
broadcast
reduce
all_gather_into_tensor
reduce_scatter_tensor
barrier
```

### Milestone 5: subgroups

Use Store keys to distribute one unique ID per subgroup and test global-rank to group-rank translation.

### Milestone 6: optional async work

Add CUDA events and `WorkNCCL` only after synchronous behavior is correct.

---

## 15. Core correctness rules

1. Every rank in a process group must issue collectives in the same order.
2. Every rank must use compatible dtypes and element counts.
3. `cudaSetDevice` uses local rank, not global rank.
4. NCCL counts are numbers of elements, not bytes.
5. The same Store key must identify the same communicator on every rank.
6. Every subgroup member must use the same ordered global-rank list.
7. A tensor must remain alive until an asynchronous collective finishes.
8. The Store is control-plane only; GPU tensors always travel through NCCL.

---

## 16. Minimal usage example

```python
import os
import torch
import nccldist as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

dist.init_process_group()

x = torch.full(
    (1024,),
    float(dist.get_rank() + 1),
    device="cuda",
)

dist.all_reduce(x, op=dist.ReduceOp.SUM)

expected = (
    dist.get_world_size()
    * (dist.get_world_size() + 1)
    / 2
)

assert torch.all(x == expected)

dist.destroy_process_group()
```

The important end-to-end path is now visible:

```text
rank environment
    -> rank-0 Store rendezvous
    -> NCCL unique ID exchange
    -> ProcessGroupNCCL construction
    -> NCCL collective on CUDA tensor
```

That is the smallest design that contains both major halves of a distributed runtime: a control plane for coordination and a GPU data plane for communication.

import ctypes
import os
import time

import torch

NCCL_UNIQUE_ID_BYTES = 128
UNIQUE_ID_FILE = "/tmp/nccl_unique_id.bin"

ncclSum = 0
ncclFloat32 = 7


class NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * NCCL_UNIQUE_ID_BYTES)]


nccl = ctypes.CDLL("libnccl.so.2")

nccl.ncclGetUniqueId.argtypes = [ctypes.POINTER(NcclUniqueId)]
nccl.ncclGetUniqueId.restype = ctypes.c_int

nccl.ncclCommInitRank.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_int,
    NcclUniqueId,
    ctypes.c_int,
]
nccl.ncclCommInitRank.restype = ctypes.c_int

nccl.ncclAllReduce.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
nccl.ncclAllReduce.restype = ctypes.c_int

nccl.ncclCommDestroy.argtypes = [ctypes.c_void_p]
nccl.ncclCommDestroy.restype = ctypes.c_int


def check(status, what):
    if status != 0:
        raise RuntimeError(f"{what} failed with NCCL status {status}")


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # raw NCCL needs some out-of-band way to share the unique id; a shared
    # file stands in for what torch.distributed's store would normally do.
    if rank == 0:
        uid = NcclUniqueId()
        check(nccl.ncclGetUniqueId(ctypes.byref(uid)), "ncclGetUniqueId")
        with open(UNIQUE_ID_FILE, "wb") as f:
            f.write(bytes(uid.internal))
    else:
        while not os.path.exists(UNIQUE_ID_FILE):
            time.sleep(0.1)
        with open(UNIQUE_ID_FILE, "rb") as f:
            data = f.read()
        uid = NcclUniqueId()
        ctypes.memmove(uid.internal, data, NCCL_UNIQUE_ID_BYTES)

    comm = ctypes.c_void_p()
    check(
        nccl.ncclCommInitRank(ctypes.byref(comm), world_size, uid, rank),
        "ncclCommInitRank",
    )

    tensor = torch.tensor([rank + 1.0], device="cuda")
    print(f"[rank {rank}] before all_reduce: {tensor.item()}")

    stream = torch.cuda.current_stream().cuda_stream
    check(
        nccl.ncclAllReduce(
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_size_t(tensor.numel()),
            ctypes.c_int(ncclFloat32),
            ctypes.c_int(ncclSum),
            comm,
            ctypes.c_void_p(stream),
        ),
        "ncclAllReduce",
    )
    torch.cuda.synchronize()

    print(f"[rank {rank}] after all_reduce: {tensor.item()}")

    expected = sum(r + 1.0 for r in range(world_size))
    assert tensor.item() == expected

    nccl.ncclCommDestroy(comm)
    if rank == 0:
        os.remove(UNIQUE_ID_FILE)


if __name__ == "__main__":
    main()

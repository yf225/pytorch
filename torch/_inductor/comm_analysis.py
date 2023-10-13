import math

from . import ir

from .utils import get_dtype_size, sympy_product
from .virtualized import V

ncclFuncAllReduce = "allreduce"
ncclFuncAllGather = "allgather"
ncclFuncReduceScatter = "reducescatter"

NCCL_HW_NVLINK = 0
NCCL_HW_PCI = 1
NCCL_HW_NET = 2

NCCL_ALGO_TREE = 0
NCCL_ALGO_RING = 1

# Latencies in us
baseLat = [
    6.8,  # Tree (LL)
    6.6,  # Ring (LL)
]
# Latencies in us
hwLat = [
    # NVLINK
    [
        0.6,  # Tree (LL)
        0.6,  # Ring (LL)
    ],
    # PCI
    [
        1.0,  # Tree (LL)
        1.0,  # Ring (LL)
    ],
    # NET
    [
        5.0,  # Tree (LL)
        2.7,  # Ring (LL)
    ],
]

VOLTA_COMPCAP_IDX = 0
AMPERE_COMPCAP_IDX = 1
HOPPER_COMPCAP_IDX = 2

# LL128 max BW per channel
llMaxBws = [
    # Volta-N1/Intel-N2/Intel-N4
    [39.0, 39.0, 20.4],
    # Ampere-N1/AMD-N2/AMD-N4
    [
        87.7,
        22.5,  # avg of ring & tree
        19.0,
    ],
    # Hopper-N1/AMD-N2/AMD-N4
    [
        87.7,
        22.5,  # avg of ring & tree
        19.0,
    ],
]

bwNVLINK = 300  # unit: GB/s, uni-directional P2P bandwidth per card
bwInfiniBand = 25  # unit: GB/s, uni-directional P2P bandwidth per node


def get_collective_type(snode: "scheduler.BaseSchedulerNode") -> str:
    if isinstance(snode.node, (ir.AllReduce, ir.AllReduceCoalesced)):
        return ncclFuncAllReduce
    elif isinstance(
        snode.node, (ir.AllGatherIntoTensor, ir.AllGatherIntoTensorCoalesced)
    ):
        return ncclFuncAllGather
    elif isinstance(
        snode.node, (ir.ReduceScatterTensor, ir.ReduceScatterTensorCoalesced)
    ):
        return ncclFuncReduceScatter
    else:
        raise Exception(f"Unsupported collective type: {snode.node}")


def estimate_nccl_collective_runtime(snode: "scheduler.BaseSchedulerNode") -> float:
    """
    Returns estimated NCCL collective runtime in nanoseconds (ns).

    The following heuristics are copied from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc.
    We aim to estimate the runtime as accurately as possible.

    Assumptions:
    - only ring algorithm (NCCL_ALGO_RING) is used
    - only Low-Latency protocol (NCCL_PROTO_LL) is used, i.e. Simple or LL128 is not used
    - only A100 gpu
    - 8 gpus per node  # TODO: is there a way to get accurate info?
    - intra-node is only NVLINK
    - inter-node is only InfiniBand
    - collective is one of: allreduce, reducescatter, allgather
    """
    tensor_numel = V.graph.sizevars.size_hint(sympy_product(snode.node.layout.size))
    tensor_dtype = snode.node.layout.dtype
    tensor_storage_size_bytes = tensor_numel * get_dtype_size(tensor_dtype)
    # Convert bytes to GB
    tensor_storage_size_GB = tensor_storage_size_bytes / 1024 / 1024 / 1024

    # TODO: is there a way to get accurate "gpus per node" and "# nodes" info?
    num_gpus_per_node = 8
    _, _, group_size = snode.node.constant_args
    nNodes = math.ceil(group_size / num_gpus_per_node)
    nRanks = group_size  # this is total # of gpus globally that participate in this collective op

    # Assume intra-node is NVLink
    bwIntra = bwNVLINK
    # Assume inter-node is InfiniBand
    bwInter = bwInfiniBand

    if nRanks <= 1:
        return 0

    # Assume A100 gpu
    compCapIndex = AMPERE_COMPCAP_IDX
    index2 = nNodes - 1 if nNodes <= 2 else 2
    # LL: for single node, we look at GPU type; for multi-node, we look at CPU type
    index1 = compCapIndex if nNodes == 1 else 0
    llMaxBw = llMaxBws[index1][index2]

    intraHw = NCCL_HW_NVLINK
    hw = intraHw if nNodes == 1 else NCCL_HW_NET

    coll = get_collective_type(snode)

    if coll == ncclFuncAllReduce:
        nsteps = 2 * (nRanks - 1)
    elif coll in (ncclFuncReduceScatter, ncclFuncAllGather):
        nsteps = nRanks - 1

    if coll == ncclFuncAllReduce:
        if nNodes > 1:
            nInterSteps = 2 * nNodes
        else:
            nInterSteps = 0
    elif coll in (ncclFuncReduceScatter, ncclFuncAllGather):
        nInterSteps = nNodes - 1

    # =============== bandwidth computation ===============
    # First compute bandwidth in GB/s; then at the end, convert it to GB/ns
    # NOTE: each step of ring algorithm is synchronized,
    # and is bottlenecked by the slowest link which is the inter-node interconnect.
    # hence when nNodes >= 2, bw is inter-node bandwidth.
    # NOTE: the original code in https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc
    # have this as `if nNodes <= 2` which seems wrong. Corrected it here.
    bw = bwIntra if nNodes == 1 else bwInter
    nChannels = 2  # Assume # channels is 2
    busBw = nChannels * bw

    # Various model refinements
    busBw = min(
        llMaxBw,
        busBw * (1.0 / 4.0 if (nNodes > 1 or coll == ncclFuncAllReduce) else 1.0 / 3.0),
    )

    # Convert bus BW to algorithm BW (tensor bytes / algoBW = actual execution time)
    ratio = (1.0 * nRanks) / nsteps
    bandwidth = busBw * ratio
    # Convert GB/s to GB/ns
    bandwidth_GB_per_ns = bandwidth / 1e9

    # =============== latency computation ===============
    # First compute latency in us; then at the end, convert it to ns
    latency = baseLat[NCCL_ALGO_RING]
    intraLat = hwLat[intraHw][NCCL_ALGO_RING]
    interLat = hwLat[NCCL_HW_NET][NCCL_ALGO_RING]

    lat = hwLat[hw][NCCL_ALGO_RING]
    # Inter-node rings still have to launch nsteps * net overhead.
    netOverhead = 0.0
    if nNodes > 1:
        netOverhead = 1.0  # getNetOverhead(comm);
    intraLat = max(intraLat, netOverhead)
    latency += (nsteps - nInterSteps) * intraLat + nInterSteps * interLat
    # Convert us to ns
    latency_ns = latency * 1e3

    # =============== final result ===============
    transport_ns = tensor_storage_size_GB / bandwidth_GB_per_ns
    return transport_ns + latency_ns

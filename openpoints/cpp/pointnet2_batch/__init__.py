import importlib

try:
    pointnet2_cuda = importlib.import_module('pointnet2_batch_cuda')
    POINTNET2_CUDA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional runtime dependency
    pointnet2_cuda = None
    POINTNET2_CUDA_IMPORT_ERROR = exc

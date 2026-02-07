import cupy as cp
from cupy.cuda import cudnn
print("cudnn constants exist?", hasattr(cudnn, "CUDNN_TENSOR_NCHW"))

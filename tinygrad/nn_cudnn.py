import numpy as np
try:
    import cupy as cp
    from cupy import cudnn
    CUDNN_AVAILABLE = True
except ImportError:
    CUDNN_AVAILABLE = False
    cp = None
    cudnn = None

from tinygrad.tensor import Tensor


class Conv2dCuDNN:
    """
    Convolution layer using cuDNN (much faster than im2col!)
    
    Drop-in replacement for Conv2d when using CUDA device.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        if not CUDNN_AVAILABLE:
            raise ImportError("cuDNN not available. Install CuPy with CUDA support.")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
        # Handle stride and padding
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        
        # Initialize weights: (out_channels, in_channels, kH, kW)
        kH, kW = self.kernel_size
        scale = np.sqrt(2.0 / (in_channels * kH * kW))
        self.W = Tensor(
            cp.random.randn(out_channels, in_channels, kH, kW).astype(cp.float32) * scale,
            requires_grad=True
        )
        
        # Initialize bias
        self.use_bias = bias
        if bias:
            self.b = Tensor(cp.zeros(out_channels, dtype=cp.float32), requires_grad=True)
        else:
            self.b = None
        
        # cuDNN descriptors (will be created on first call)
        self.conv_desc = None
        self.filter_desc = None
        self.input_desc = None
        self.output_desc = None
        self.algo = None
        self.workspace_size = 0
        self.workspace = None
    
    def _create_descriptors(self, input_shape):
        """Create cuDNN descriptors for this convolution"""
        N, C, H, W = input_shape
        kH, kW = self.kernel_size
        
        # Calculate output dimensions
        H_out = (H + 2 * self.padding[0] - kH) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - kW) // self.stride[1] + 1
        
        # Create filter descriptor
        self.filter_desc = cudnn.create_filter_descriptor(
            self.W.data,  # Filter tensor
            cudnn.CUDNN_TENSOR_NCHW  # Data format
        )
        
        # Create convolution descriptor
        self.conv_desc = cudnn.create_convolution_descriptor(
            self.padding,  # (pad_h, pad_w)
            self.stride,   # (stride_h, stride_w)
            (1, 1),        # dilation (we don't use dilation)
            cudnn.CUDNN_CROSS_CORRELATION,  # Mode (standard convolution)
            cudnn.CUDNN_DATA_FLOAT  # Data type
        )
        
        # Create input descriptor
        self.input_desc = cudnn.create_tensor_descriptor(
            (N, C, H, W),
            cudnn.CUDNN_DATA_FLOAT,
            cudnn.CUDNN_TENSOR_NCHW
        )
        
        # Create output descriptor
        self.output_desc = cudnn.create_tensor_descriptor(
            (N, self.out_channels, H_out, W_out),
            cudnn.CUDNN_DATA_FLOAT,
            cudnn.CUDNN_TENSOR_NCHW
        )
        
        # Find best algorithm
        self.algo = cudnn.get_convolution_forward_algorithm(
            cudnn.get_handle(),
            self.input_desc,
            self.filter_desc,
            self.conv_desc,
            self.output_desc,
            cudnn.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST  # Choose fastest algo
        )
        
        # Get workspace size needed
        self.workspace_size = cudnn.get_convolution_forward_workspace_size(
            cudnn.get_handle(),
            self.input_desc,
            self.filter_desc,
            self.conv_desc,
            self.output_desc,
            self.algo
        )
        
        # Allocate workspace
        if self.workspace_size > 0:
            self.workspace = cp.empty(self.workspace_size, dtype=cp.uint8)
    
    def __call__(self, x):
        """
        Forward pass using cuDNN
        
        x: Tensor with shape (N, C, H, W) on GPU
        """
        # Ensure input is on GPU
        if not isinstance(x.data, cp.ndarray):
            raise ValueError("Input must be on GPU (cupy array)")
        
        # Create descriptors on first call
        if self.conv_desc is None:
            self._create_descriptors(x.data.shape)
        
        # Allocate output
        N, C, H, W = x.data.shape
        kH, kW = self.kernel_size
        H_out = (H + 2 * self.padding[0] - kH) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - kW) // self.stride[1] + 1
        
        output_data = cp.empty((N, self.out_channels, H_out, W_out), dtype=cp.float32)
        
        # Run cuDNN convolution
        cudnn.convolution_forward(
            cudnn.get_handle(),
            1.0,  # alpha
            self.input_desc,
            x.data,
            self.filter_desc,
            self.W.data,
            self.conv_desc,
            self.algo,
            self.workspace if self.workspace_size > 0 else None,
            self.workspace_size,
            0.0,  # beta
            self.output_desc,
            output_data
        )
        
        # Add bias if present
        if self.use_bias:
            # Reshape bias for broadcasting: (1, C, 1, 1)
            bias_reshaped = self.b.data.reshape(1, self.out_channels, 1, 1)
            output_data = output_data + bias_reshaped
        
        # Create output tensor
        out = Tensor(output_data, requires_grad=x.requires_grad or self.W.requires_grad)
        
        # Setup backward pass
        if out.requires_grad:
            out._prev = {x, self.W}
            if self.use_bias:
                out._prev.add(self.b)
            out._op = "conv2d_cudnn"
            
            def _backward():
                if out.grad is None:
                    return
                
                # Gradient w.r.t. input
                if x.requires_grad:
                    x._init_grad()
                    
                    # cuDNN backward data
                    cudnn.convolution_backward_data(
                        cudnn.get_handle(),
                        1.0,  # alpha
                        self.filter_desc,
                        self.W.data,
                        self.output_desc,
                        out.grad,
                        self.conv_desc,
                        cudnn.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,  # Algorithm
                        None,  # workspace
                        0,     # workspace_size
                        1.0,   # beta (accumulate)
                        self.input_desc,
                        x.grad
                    )
                
                # Gradient w.r.t. weights
                if self.W.requires_grad:
                    self.W._init_grad()
                    
                    # cuDNN backward filter
                    cudnn.convolution_backward_filter(
                        cudnn.get_handle(),
                        1.0,  # alpha
                        self.input_desc,
                        x.data,
                        self.output_desc,
                        out.grad,
                        self.conv_desc,
                        cudnn.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,  # Algorithm
                        None,  # workspace
                        0,     # workspace_size
                        1.0,   # beta (accumulate)
                        self.filter_desc,
                        self.W.grad
                    )
                
                # Gradient w.r.t. bias
                if self.use_bias and self.b.requires_grad:
                    self.b._init_grad()
                    
                    # Sum over N, H, W dimensions
                    bias_grad = cp.sum(out.grad, axis=(0, 2, 3))
                    self.b.grad += bias_grad
            
            out._backward = _backward
        
        return out
    
    def parameters(self):
        """Return list of parameters for optimizer"""
        params = [self.W]
        if self.use_bias:
            params.append(self.b)
        return params
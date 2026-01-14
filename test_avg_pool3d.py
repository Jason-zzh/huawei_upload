#!/usr/bin/env python3
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# pylint: disable=unused-variable
"""avg_pool3d op test case"""
import pytest
import numpy as np
import torch
import mindspore as ms
from mindspore import ops, mint, vmap
from tests.utils.mark_utils import arg_mark
from tests.utils.tools import allclose_nparray


# Set fixed random seed for reproducibility
np.random.seed(42)


def generate_random_input(shape, dtype=np.float32):
    """Generate random input data for avg_pool3d"""
    return np.random.uniform(-1, 1, shape).astype(dtype)


def generate_expect_forward_output(input_tensor, kernel_size, stride=None, padding=0, ceil_mode=False, 
                                  count_include_pad=True, divisor_override=None):
    """Generate expected output using PyTorch avg_pool3d."""
    if stride is None:
        stride = kernel_size
        
    return torch.nn.functional.avg_pool3d(
        torch.tensor(input_tensor),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override
    )


def avg_pool3d_forward_func(input_tensor, kernel_size, stride=None, padding=0, ceil_mode=False, 
                           count_include_pad=True, divisor_override=None):
    """MindSpore forward calculation function for avg_pool3d"""
    return mint.nn.functional.avg_pool3d(
        input_tensor, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding, 
        ceil_mode=ceil_mode, 
        count_include_pad=count_include_pad, 
        divisor_override=divisor_override
    )


def generate_special_input(shape, dtype, special_type):
    """Generate special input data"""
    if special_type == "inf":
        x = np.ones(shape, dtype=dtype)
        x[0, 0, 0, 0, 0] = np.inf
        x[-1, -1, -1, -1, -1] = -np.inf
        return x
    if special_type == "nan":
        x = np.ones(shape, dtype=dtype)
        x[0, 0, 0, 0, 0] = np.nan
        return x
    if special_type == "empty":
        # For avg_pool3d, we can't have empty spatial dimensions, so we test with minimal valid shape
        return np.random.uniform(-1, 1, (0, *shape[1:])).astype(dtype) if shape[0] == 0 else np.array([], dtype=dtype)
    return generate_random_input(shape, dtype)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_avg_pool3d_basic(mode):
    """
    Feature: Test avg_pool3d basic functionality.
    Description: Test avg_pool3d with basic parameters.
    Expectation: Results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # Basic test case: 3D input (batch, channels, depth, height, width)
    input_shape = (1, 2, 8, 8, 8)
    input_tensor = generate_random_input(input_shape, np.float32)
    # Test with different kernel sizes and strides
    kernel_size = (2, 2, 2)
    stride = (2, 2, 2)
    padding = (0, 0, 0)
    # Expected output from PyTorch
    expect = generate_expect_forward_output(input_tensor, kernel_size, stride, padding)
    # MindSpore result
    ms_input = ms.Tensor(input_tensor)
    output = avg_pool3d_forward_func(ms_input, kernel_size, stride, padding)
    # Compare results
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_avg_pool3d_with_ceil_mode(mode):
    """
    Feature: Test avg_pool3d with ceil_mode.
    Description: Test avg_pool3d with ceil_mode enabled.
    Expectation: Results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (1, 1, 5, 5, 5)
    input_tensor = generate_random_input(input_shape, np.float32)
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (0, 0, 0)
    ceil_mode = True
    expect = generate_expect_forward_output(
        input_tensor, kernel_size, stride, padding, ceil_mode=ceil_mode
    )
    ms_input = ms.Tensor(input_tensor)
    output = avg_pool3d_forward_func(
        ms_input, kernel_size, stride, padding, ceil_mode=ceil_mode
    )
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_avg_pool3d_count_include_pad_false(mode):
    """
    Feature: Test avg_pool3d with count_include_pad=False.
    Description: Test avg_pool3d when count_include_pad is disabled.
    Expectation: Results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (1, 1, 6, 6, 6)
    input_tensor = generate_random_input(input_shape, np.float32)
    kernel_size = (3, 3, 3)
    stride = (3, 3, 3)
    padding = (1, 1, 1)
    count_include_pad = False
    expect = generate_expect_forward_output(
        input_tensor, kernel_size, stride, padding, 
        count_include_pad=count_include_pad
    )
    ms_input = ms.Tensor(input_tensor)
    output = avg_pool3d_forward_func(
        ms_input, kernel_size, stride, padding, 
        count_include_pad=count_include_pad
    )
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_avg_pool3d_with_padding(mode):
    """
    Feature: Test avg_pool3d with padding.
    Description: Test avg_pool3d with various padding values.
    Expectation: Results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (2, 3, 4, 8, 8)
    input_tensor = generate_random_input(input_shape, np.float32)
    kernel_size = (2, 3, 3)
    stride = (1, 2, 2)
    padding = (1, 1, 1)
    expect = generate_expect_forward_output(
        input_tensor, kernel_size, stride, padding
    )
    ms_input = ms.Tensor(input_tensor)
    output = avg_pool3d_forward_func(
        ms_input, kernel_size, stride, padding
    )
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])  # Removed float16 as PyTorch doesn't support it for avg_pool3d
def test_avg_pool3d_dtype_coverage(mode, dtype):
    """
    Feature: dtype coverage for avg_pool3d operator.
    Description: test avg_pool3d with various dtypes.
    Expectation: results match PyTorch implementation with appropriate precision.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (1, 1, 4, 4, 4)
    input_tensor = generate_random_input(input_shape, dtype)
    kernel_size = (2, 2, 2)
    stride = (2, 2, 2)
    expect = generate_expect_forward_output(
        input_tensor, kernel_size, stride
    )
    ms_input = ms.Tensor(input_tensor)
    output = avg_pool3d_forward_func(
        ms_input, kernel_size, stride
    )
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_avg_pool3d_different_kernel_stride(mode):
    """
    Feature: Test avg_pool3d with different kernel and stride sizes.
    Description: Test avg_pool3d with non-uniform kernel and stride.
    Expectation: Results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (1, 2, 10, 10, 10)
    input_tensor = generate_random_input(input_shape, np.float32)
    kernel_size = (3, 4, 5)
    stride = (2, 3, 4)
    padding = (1, 1, 2)
    expect = generate_expect_forward_output(
        input_tensor, kernel_size, stride, padding
    )
    ms_input = ms.Tensor(input_tensor)
    output = avg_pool3d_forward_func(
        ms_input, kernel_size, stride, padding
    )
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_avg_pool3d_single_element_kernel(mode):
    """
    Feature: Test avg_pool3d with single element kernel.
    Description: Test avg_pool3d with kernel size of (1, 1, 1).
    Expectation: Results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (1, 1, 4, 4, 4)
    input_tensor = generate_random_input(input_shape, np.float32)
    kernel_size = (1, 1, 1)
    stride = (1, 1, 1)
    expect = generate_expect_forward_output(
        input_tensor, kernel_size, stride
    )
    ms_input = ms.Tensor(input_tensor)
    output = avg_pool3d_forward_func(
        ms_input, kernel_size, stride
    )
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_avg_pool3d_large_input(mode):
    """
    Feature: Test avg_pool3d with larger input tensor.
    Description: Test avg_pool3d with larger input tensor.
    Expectation: Results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (2, 2, 6, 8, 8)
    input_tensor = generate_random_input(input_shape, np.float32)
    kernel_size = (2, 2, 2)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    expect = generate_expect_forward_output(
        input_tensor, kernel_size, stride, padding
    )
    ms_input = ms.Tensor(input_tensor)
    output = avg_pool3d_forward_func(
        ms_input, kernel_size, stride, padding
    )
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_avg_pool3d_default_stride_same_as_kernel(mode):
    """
    Feature: Test avg_pool3d with stride defaults to kernel_size when not specified.
    Description: Test avg_pool3d when stride is explicitly set to kernel_size to match default behavior.
    Expectation: Results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (1, 1, 6, 6, 6)
    input_tensor = generate_random_input(input_shape, np.float32)
    kernel_size = (2, 2, 2)
    # Using explicit stride that equals kernel_size to match default behavior
    stride = (2, 2, 2)  # This is what PyTorch does when stride=None
    expect = generate_expect_forward_output(
        input_tensor, kernel_size, stride=stride
    )
    ms_input = ms.Tensor(input_tensor)
    output = avg_pool3d_forward_func(
        ms_input, kernel_size, stride=stride
    )
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_avg_pool3d_divisor_override(mode):
    """
    Feature: Test avg_pool3d with divisor_override.
    Description: Test avg_pool3d with custom divisor override.
    Expectation: Results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (1, 1, 4, 4, 4)
    input_tensor = generate_random_input(input_shape, np.float32)
    kernel_size = (2, 2, 2)
    stride = (2, 2, 2)
    divisor_override = 8  # Override to use 8 instead of actual kernel volume (2*2*2=8)
    expect = generate_expect_forward_output(
        input_tensor, kernel_size, stride, divisor_override=divisor_override
    )
    ms_input = ms.Tensor(input_tensor)
    output = avg_pool3d_forward_func(
        ms_input, kernel_size, stride, divisor_override=divisor_override
    )
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_avg_pool3d_with_inf_values(mode):
    """
    Feature: Test avg_pool3d with infinite values.
    Description: Test avg_pool3d with inf/nan values.
    Expectation: Results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (1, 1, 4, 4, 4)
    input_tensor = generate_special_input(input_shape, np.float32, "inf")
    kernel_size = (2, 2, 2)
    stride = (2, 2, 2)
    expect = generate_expect_forward_output(
        input_tensor, kernel_size, stride
    )
    ms_input = ms.Tensor(input_tensor)
    output = avg_pool3d_forward_func(
        ms_input, kernel_size, stride
    )
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_avg_pool3d_with_nan_values(mode):
    """
    Feature: Test avg_pool3d with NaN values.
    Description: Test avg_pool3d with NaN values.
    Expectation: Results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (1, 1, 4, 4, 4)
    input_tensor = generate_special_input(input_shape, np.float32, "nan")
    kernel_size = (2, 2, 2)
    stride = (2, 2, 2)
    expect = generate_expect_forward_output(
        input_tensor, kernel_size, stride
    )
    ms_input = ms.Tensor(input_tensor)
    output = avg_pool3d_forward_func(
        ms_input, kernel_size, stride
    )
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_avg_pool3d_non_contiguous_input(mode):
    """
    Feature: Test avg_pool3d with non-contiguous input.
    Description: Test avg_pool3d with non-contiguous input tensor.
    Expectation: Results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # Create a larger tensor and take a slice to make it non-contiguous
    full_tensor = generate_random_input((1, 2, 10, 10, 10), np.float32)
    input_tensor = full_tensor[:, :, ::2, ::2, ::2]  # Strided slice making it non-contiguous
    kernel_size = (2, 2, 2)
    stride = (2, 2, 2)
    padding = (0, 0, 0)
    # Ensure both frameworks handle non-contiguous tensors the same way
    expect = generate_expect_forward_output(input_tensor, kernel_size, stride, padding)
    ms_input = ms.Tensor(input_tensor)
    output = avg_pool3d_forward_func(ms_input, kernel_size, stride, padding)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_avg_pool3d_vmap(mode):
    """
    Feature: Test avg_pool3d with vmap.
    Description: Test avg_pool3d with vectorization.
    Expectation: Results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        pytest.skip("Vmap test not suitable for KBK mode")  # Vmap typically only works in pynative mode
    # Test vmap functionality with batch dimension
    def avg_pool3d_batch_func(x):
        return avg_pool3d_forward_func(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    # Create multiple samples in batch
    input_shape = (3, 1, 4, 4, 4)  # Batch of 3
    input_tensor = generate_random_input(input_shape, np.float32)
    ms_input = ms.Tensor(input_tensor)
    output = vmap(avg_pool3d_batch_func, in_axes=0, out_axes=0)(ms_input)
    # For comparison, process each sample individually
    torch_input = torch.tensor(input_tensor)
    individual_results = []
    for i in range(input_tensor.shape[0]):
        result = torch.nn.functional.avg_pool3d(
            torch_input[i:i+1], kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )
        individual_results.append(result.squeeze(0))
    expect = torch.stack(individual_results)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_avg_pool3d_backward(mode):
    """
    Feature: Test avg_pool3d backward functionality.
    Description: Test avg_pool3d backward pass.
    Expectation: Backward gradients match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (1, 1, 4, 4, 4)
    input_tensor_np = generate_random_input(input_shape, np.float32)
    # PyTorch backward
    torch_input = torch.tensor(input_tensor_np, requires_grad=True)
    torch_output = torch.nn.functional.avg_pool3d(torch_input, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    loss_torch = torch_output.sum()
    loss_torch.backward()
    torch_grad = torch_input.grad
    # MindSpore backward
    def forward_func(x):
        return avg_pool3d_forward_func(x, kernel_size=(2, 2, 2), stride=(2, 2, 2)).sum()
    ms_input = ms.Tensor(input_tensor_np)
    grad_fn = ops.grad(forward_func, grad_position=0)
    ms_grad = grad_fn(ms_input)
    allclose_nparray(torch_grad.detach().numpy(), ms_grad.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_avg_pool3d_dynamic_shape(mode):
    """
    Feature: Test avg_pool3d with dynamic shape.
    Description: Test avg_pool3d with dynamic shape tensors.
    Expectation: Results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # Dynamic shape test - testing with different sizes
    for shape in [(1, 1, 6, 6, 6), (2, 2, 8, 8, 8), (1, 3, 5, 7, 9)]:
        input_tensor = generate_random_input(shape, np.float32)
        kernel_size = (2, 2, 2)
        stride = (2, 2, 2)
        padding = (0, 0, 0)
        expect = generate_expect_forward_output(input_tensor, kernel_size, stride, padding)
        ms_input = ms.Tensor(input_tensor)
        output = avg_pool3d_forward_func(ms_input, kernel_size, stride, padding)
        allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_avg_pool3d_broadcast_compatibility(mode):
    """
    Feature: Test avg_pool3d broadcast compatibility.
    Description: Test avg_pool3d behavior with different tensor shapes.
    Expectation: Function handles various shapes appropriately.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # This test verifies that the function works with various compatible shapes
    # AvgPool3d works on 5D tensors (N, C, D, H, W)
    input_shape = (2, 3, 8, 8, 8)  # Standard 5D tensor for avg_pool3d
    input_tensor = generate_random_input(input_shape, np.float32)
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    expect = generate_expect_forward_output(input_tensor, kernel_size, stride, padding)
    ms_input = ms.Tensor(input_tensor)
    output = avg_pool3d_forward_func(ms_input, kernel_size, stride, padding)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_avg_pool3d_empty_like_scenario(mode):
    """
    Feature: Test avg_pool3d with minimal valid input.
    Description: Test avg_pool3d with smallest possible valid input.
    Expectation: Function handles minimal input appropriately.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # Test with minimal valid shape that can go through avg_pool3d
    # Need at least kernel_size elements in each spatial dim
    input_shape = (1, 1, 2, 2, 2)  # Minimal for kernel_size=(2,2,2)
    input_tensor = generate_random_input(input_shape, np.float32)
    kernel_size = (2, 2, 2)
    stride = (2, 2, 2)
    expect = generate_expect_forward_output(input_tensor, kernel_size, stride)
    ms_input = ms.Tensor(input_tensor)
    output = avg_pool3d_forward_func(ms_input, kernel_size, stride)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)

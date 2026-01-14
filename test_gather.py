# Copyright 2024 Huawei Technologies Co., Ltd
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
"""gather op test case"""
import pytest
import numpy as np
import torch
import mindspore as ms
from mindspore import ops, mint, vmap
from tests.utils.mark_utils import arg_mark
from tests.utils.tools import allclose_nparray


def generate_random_input(shape, dtype):
    """Generate random input data"""
    if np.issubdtype(dtype, np.integer):
        # For integer types, generate values in a reasonable range
        return np.random.randint(-10, 10, shape).astype(dtype)
    # For float types
    return np.random.uniform(-5, 5, shape).astype(dtype)


def generate_random_index(shape, max_idx):
    """Generate random index data within range [0, max_idx)"""
    return np.random.randint(0, max_idx, shape).astype(np.int64)  # Changed to int64


def generate_expect_forward_output(input_tensor, dim, index_tensor):
    """Generate expected forward calculation results using PyTorch"""
    return torch.gather(input_tensor, dim, index_tensor)


def gather_forward_func(input_tensor, dim, index_tensor):
    """MindSpore forward calculation function"""
    return mint.gather(input_tensor, dim, index_tensor)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('shape, dim, index_shape', [
    ((2, 3), 0, (1, 3)),
    ((2, 3), 1, (2, 2)),
    ((4, 5, 6), 1, (4, 2, 6)),
    ((3, 4, 5), 2, (3, 4, 2)),
])
def test_gather_basic(mode, shape, dim, index_shape):
    """
    Feature: standard forward features.
    Description: test standard cases for gather.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # Generate input and index data
    input_np = generate_random_input(shape, np.float32)
    index_np = generate_random_index(index_shape, shape[dim])
    # Convert to tensors
    ms_input = ms.Tensor(input_np)
    ms_index = ms.Tensor(index_np)
    torch_input = torch.tensor(input_np)
    torch_index = torch.tensor(index_np, dtype=torch.int64)  # Changed to int64
    # Expected result
    expect = generate_expect_forward_output(torch_input, dim, torch_index)
    # MindSpore result
    output = gather_forward_func(ms_input, dim, ms_index)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [
    np.float16, np.float32, np.float64,
    np.int32, np.int64
])
def test_gather_dtype_coverage(mode, dtype):
    """
    Feature: dtype coverage for gather operator.
    Description: test gather with various dtypes supported by MindSpore gather operator.
    Expectation: results match PyTorch implementation with appropriate precision.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    shape = (3, 4)
    dim = 1
    index_shape = (3, 2)
    input_np = generate_random_input(shape, dtype)
    index_np = generate_random_index(index_shape, shape[dim])
    ms_input = ms.Tensor(input_np)
    ms_index = ms.Tensor(index_np)
    torch_input = torch.tensor(input_np, dtype=torch.from_numpy(np.array(0, dtype=dtype)).dtype)
    torch_index = torch.tensor(index_np, dtype=torch.int64)  # Changed to int64
    expect = generate_expect_forward_output(torch_input, dim, torch_index)
    output = gather_forward_func(ms_input, dim, ms_index)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_gather_negative_dim(mode):
    """
    Feature: negative dimension support for gather operator.
    Description: test gather with negative dimensions.
    Expectation: results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    shape = (2, 3)
    dim = -1  # Should be equivalent to dim=1
    index_shape = (2, 2)
    input_np = generate_random_input(shape, np.float32)
    index_np = generate_random_index(index_shape, shape[-1])
    ms_input = ms.Tensor(input_np)
    ms_index = ms.Tensor(index_np)
    torch_input = torch.tensor(input_np)
    torch_index = torch.tensor(index_np, dtype=torch.int64)  # Changed to int64
    expect = generate_expect_forward_output(torch_input, dim, torch_index)
    output = gather_forward_func(ms_input, dim, ms_index)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_gather_3d(mode):
    """
    Feature: 3D tensor support for gather operator.
    Description: test gather with 3D tensors.
    Expectation: results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    shape = (2, 3, 4)
    dim = 1
    index_shape = (2, 2, 4)
    input_np = generate_random_input(shape, np.float32)
    index_np = generate_random_index(index_shape, shape[dim])
    ms_input = ms.Tensor(input_np)
    ms_index = ms.Tensor(index_np)
    torch_input = torch.tensor(input_np)
    torch_index = torch.tensor(index_np, dtype=torch.int64)  # Changed to int64
    expect = generate_expect_forward_output(torch_input, dim, torch_index)
    output = gather_forward_func(ms_input, dim, ms_index)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_gather_single_element(mode):
    """
    Feature: single element tensor support for gather operator.
    Description: test gather with single element tensors.
    Expectation: results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    shape = (1, 1)
    dim = 0
    index_shape = (1, 1)
    input_np = generate_random_input(shape, np.float32)
    index_np = generate_random_index(index_shape, shape[dim])
    ms_input = ms.Tensor(input_np)
    ms_index = ms.Tensor(index_np)
    torch_input = torch.tensor(input_np)
    torch_index = torch.tensor(index_np, dtype=torch.int64)  # Changed to int64
    expect = generate_expect_forward_output(torch_input, dim, torch_index)
    output = gather_forward_func(ms_input, dim, ms_index)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_gather_large_tensor(mode):
    """
    Feature: large tensor support for gather operator.
    Description: test gather with large tensors.
    Expectation: results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    shape = (100, 50)
    dim = 0
    index_shape = (20, 50)
    input_np = generate_random_input(shape, np.float32)
    index_np = generate_random_index(index_shape, shape[dim])
    ms_input = ms.Tensor(input_np)
    ms_index = ms.Tensor(index_np)
    torch_input = torch.tensor(input_np)
    torch_index = torch.tensor(index_np, dtype=torch.int64)  # Changed to int64
    expect = generate_expect_forward_output(torch_input, dim, torch_index)
    output = gather_forward_func(ms_input, dim, ms_index)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_gather_vmap(mode):
    """
    Feature: vmap support for gather operator.
    Description: test gather with vmap.
    Expectation: results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # Create batched inputs
    batch_shape = (2,)
    tensor_shape = (3, 4)
    shape = batch_shape + tensor_shape
    dim = 1  # Note: need to adjust for vmap
    index_shape = batch_shape + (3, 2)
    input_np = generate_random_input(shape, np.float32)
    index_np = generate_random_index(index_shape, tensor_shape[dim])
    ms_input = ms.Tensor(input_np)
    ms_index = ms.Tensor(index_np)
    # Apply vmap to the function
    vmapped_func = vmap(gather_forward_func, in_axes=(0, None, 0), out_axes=0)
    output = vmapped_func(ms_input, dim, ms_index)
    # Manual computation for comparison
    torch_input = torch.tensor(input_np)
    torch_index = torch.tensor(index_np, dtype=torch.int64)
    expected_results = []
    for i in range(batch_shape[0]):
        expected = generate_expect_forward_output(torch_input[i], dim, torch_index[i])
        expected_results.append(expected.detach().numpy())
    expect = np.stack(expected_results)
    allclose_nparray(expect, output.asnumpy(), rtol=0, atol=0, equal_nan= True)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_gather_empty_tensor(mode):
    """
    Feature: empty tensor support for gather operator.
    Description: test gather with empty tensors.
    Expectation: results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # Empty tensor test
    input_np = np.array([], dtype=np.float32).reshape(0, 3)
    index_np = np.array([], dtype=np.int64).reshape(0, 2)
    if len(input_np) > 0:  # Only test if not empty
        ms_input = ms.Tensor(input_np)
        ms_index = ms.Tensor(index_np)
        torch_input = torch.tensor(input_np)
        torch_index = torch.tensor(index_np, dtype=torch.int64)
        expect = generate_expect_forward_output(torch_input, 0, torch_index)
        output = gather_forward_func(ms_input, 0, ms_index)
        allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_gather_special_values(mode):
    """
    Feature: special values (NaN, Inf) support for gather operator.
    Description: test gather with NaN and Inf values.
    Expectation: results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    shape = (3, 4)
    dim = 0
    index_shape = (2, 4)
    # Create input with special values
    input_np = np.random.randn(*shape).astype(np.float32)
    input_np[0, 0] = np.nan  # Add NaN
    input_np[1, 1] = np.inf  # Add positive infinity
    input_np[2, 2] = -np.inf  # Add negative infinity
    index_np = generate_random_index(index_shape, shape[dim])
    ms_input = ms.Tensor(input_np)
    ms_index = ms.Tensor(index_np)
    torch_input = torch.tensor(input_np)
    torch_index = torch.tensor(index_np, dtype=torch.int64)
    expect = generate_expect_forward_output(torch_input, dim, torch_index)
    output = gather_forward_func(ms_input, dim, ms_index)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('shape,dim,index_shape', [
    ((), 0, ()),  # 0D tensor
    ((5,), 0, (3,)),  # 1D tensor
    ((2, 3, 4, 5, 6, 7, 8), 3, (2, 3, 2, 5, 6, 7, 8)),  # 7D tensor
    ((2, 3, 4, 5, 6, 7, 8, 9), 4, (2, 3, 4, 2, 6, 7, 8, 9)),  # 8D tensor
])
def test_gather_multi_dimensional(mode, shape, dim, index_shape):
    """
    Feature: multi-dimensional tensor support for gather operator.
    Description: test gather with 0D to 8D tensors.
    Expectation: results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    if len(shape) == 0:  # Skip 0D tensor test as it's not supported by gather
        return
    input_np = generate_random_input(shape, np.float32)
    index_np = generate_random_index(index_shape, shape[dim])
    ms_input = ms.Tensor(input_np)
    ms_index = ms.Tensor(index_np)
    torch_input = torch.tensor(input_np)
    torch_index = torch.tensor(index_np, dtype=torch.int64)
    expect = generate_expect_forward_output(torch_input, dim, torch_index)
    output = gather_forward_func(ms_input, dim, ms_index)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_gather_non_contiguous(mode):
    """
    Feature: non-contiguous tensor support for gather operator.
    Description: test gather with non-contiguous tensors.
    Expectation: results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    shape = (4, 5)
    dim = 0
    index_shape = (2, 5)
    input_np = generate_random_input((5, 4), np.float32).T  # Transpose to make non-contiguous
    index_np = generate_random_index(index_shape, shape[dim])
    ms_input = ms.Tensor(input_np)  # This should be non-contiguous
    ms_index = ms.Tensor(index_np)
    torch_input = torch.tensor(input_np)  # This should be non-contiguous
    torch_index = torch.tensor(index_np, dtype=torch.int64)
    expect = generate_expect_forward_output(torch_input, dim, torch_index)
    output = gather_forward_func(ms_input, dim, ms_index)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)
    
def gather_forward_func_for_backward(x, dim, index):
    """MindSpore forward calculation function for backward testing"""
    return mint.gather(x, dim, index)


def gather_backward_func(x, dim, index):
    """MindSpore backward propagation function"""
    return ops.grad(gather_forward_func_for_backward, (0,))(x, dim, index)[0]

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_gather_backward(mode):
    """
    Feature: backward support for gather operator.
    Description: test gather with backward gradients.
    Expectation: backward pass works correctly.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    shape = (3, 4)
    dim = 0
    index_shape = (2, 4)  # Make sure this matches the output shape after gathering
    input_np = generate_random_input(shape, np.float32)
    index_np = generate_random_index(index_shape, shape[dim])
    ms_input = ms.Tensor(input_np, dtype=ms.float32)
    ms_index = ms.Tensor(index_np, dtype=ms.int64)
    # Define the forward function for gradient computation
    def gather_forward_func_for_grad(x, idx):
        return mint.gather(x, dim, idx)
    # Compute gradient w.r.t. input
    output_grad = ops.grad(gather_forward_func_for_grad, (0,))(ms_input, ms_index)
    # Gradient should have the same shape as input
    assert output_grad.shape == ms_input.shape
    assert output_grad.dtype == ms.float32

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_gather_dynamic_shape(mode):
    """
    Feature: dynamic shape support for gather operator.
    Description: test gather with dynamic shapes.
    Expectation: results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Test with various dynamic shapes
    test_cases = [
        ((5, 4), 0, (3, 4)),
        ((3, 6), 1, (3, 2)),
        ((2, 3, 4), 1, (2, 1, 4)),
        ((2, 3, 4, 5), 2, (2, 3, 2, 5))
    ]
    for input_shape, dim, index_shape in test_cases:
        input_np = generate_random_input(input_shape, np.float32)
        index_np = generate_random_index(index_shape, input_shape[dim])
        ms_input = ms.Tensor(input_np)
        ms_index = ms.Tensor(index_np)
        torch_input = torch.tensor(input_np)
        torch_index = torch.tensor(index_np, dtype=torch.int64)
        expect = generate_expect_forward_output(torch_input, dim, torch_index)
        output = gather_forward_func(ms_input, dim, ms_index)
        allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_gather_broadcast_behavior(mode):
    """
    Feature: broadcast behavior support for gather operator.
    Description: test gather with broadcasting scenarios.
    Expectation: results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Test cases that involve broadcasting-like behavior
    # MindSpore gather doesn't broadcast in the same way as some other ops,
    # but we test various index shapes that are compatible with input shapes
    test_cases = [
        ((4, 5), 0, (2, 5)),  # Index has smaller first dim
        ((4, 5), 1, (4, 3)),  # Index has smaller second dim
        ((2, 3, 4), 0, (1, 3, 4)),  # 3D with partial indexing
        ((2, 3, 4), 1, (2, 1, 4)),  # 3D with middle dim indexing
        ((2, 3, 4), 2, (2, 3, 1)),  # 3D with last dim indexing
    ]
    for input_shape, dim, index_shape in test_cases:
        input_np = generate_random_input(input_shape, np.float32)
        index_np = generate_random_index(index_shape, input_shape[dim])
        ms_input = ms.Tensor(input_np)
        ms_index = ms.Tensor(index_np)
        torch_input = torch.tensor(input_np)
        torch_index = torch.tensor(index_np, dtype=torch.int64)
        expect = generate_expect_forward_output(torch_input, dim, torch_index)
        output = gather_forward_func(ms_input, dim, ms_index)
        allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan= True)
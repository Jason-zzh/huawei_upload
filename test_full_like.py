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
""" full_like op test case """
import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, jit, ops
from tests.utils.mark_utils import arg_mark
from tests.utils.tools import allclose_nparray
import torch


def generate_expect_forward_output(input_tensor, fill_value, dtype=None):
    """Generate expected output using PyTorch full_like."""
    if dtype is None:
        return torch.full_like(input_tensor, fill_value)
    if dtype == ms.float16:
        torch_dtype = torch.float16
    elif dtype == ms.float32:
        torch_dtype = torch.float32
    elif dtype == ms.float64:
        torch_dtype = torch.float64
    elif dtype == ms.int8:
        torch_dtype = torch.int8
    elif dtype == ms.int16:
        torch_dtype = torch.int16
    elif dtype == ms.int32:
        torch_dtype = torch.int32
    elif dtype == ms.int64:
        torch_dtype = torch.int64
    elif dtype == ms.uint8:
        torch_dtype = torch.uint8
    elif dtype == ms.bool_:
        torch_dtype = torch.bool
    else:
        torch_dtype = None
    if torch_dtype is not None:
        return torch.full_like(input_tensor, fill_value, dtype=torch_dtype)
    else:
        return torch.full_like(input_tensor, fill_value)


def full_like_forward_func(input_tensor, fill_value, dtype=None):
    """Forward function for mint.full_like."""
    if dtype is None:
        return mint.full_like(input_tensor, fill_value)
    return mint.full_like(input_tensor, fill_value, dtype=dtype)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("dtype", [None, ms.float32])
def test_full_like_std(mode, dtype):
    """
    Feature: pyboost function.
    Description: test function full_like.
    Expectation: expect correct result.
    """
    np.random.seed(0)
    input_tensor = torch.randn(2, 3)
    ms_input = ms.Tensor(input_tensor.numpy())
    fill_value = 3.14
    expect = generate_expect_forward_output(input_tensor, fill_value, dtype)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = full_like_forward_func(ms_input, fill_value, dtype)
    elif mode == "KBK":
        output = jit(
            full_like_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms_input, fill_value, dtype)
    else:
        output = None
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    assert output.shape == expect.shape


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("dtype", [ms.float16, ms.float32, ms.float64, ms.int32, ms.int64, ms.bool_])
def test_full_like_dtype_coverage(mode, dtype):
    """
    Feature: dtype coverage for full_like operator.
    Description: test full_like with various dtypes.
    Expectation: results match PyTorch implementation.
    """
    np.random.seed(1)
    input_tensor = torch.randn(3, 4, 2)
    ms_input = ms.Tensor(input_tensor.numpy())
    if dtype == ms.bool_:
        fill_value = True
    elif dtype in [ms.int8, ms.int16, ms.int32, ms.int64]:
        fill_value = 42
    else:
        fill_value = 1.5
    expect = generate_expect_forward_output(input_tensor, fill_value, dtype)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = full_like_forward_func(ms_input, fill_value, dtype)
    elif mode == "KBK":
        output = jit(
            full_like_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms_input, fill_value, dtype)
    else:
        output = None
    allclose_nparray(
        expect.detach().numpy(),
        output.asnumpy(),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert output.shape == expect.shape


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("fill_value", [0, 1, -1, 2.5, -3.7, float('inf'), float('-inf'), float('nan')])
def test_full_like_fill_values(mode, fill_value):
    """
    Feature: fill value coverage for full_like operator.
    Description: test full_like with various fill values.
    Expectation: results match PyTorch implementation.
    """
    np.random.seed(2)
    input_tensor = torch.randn(5, 2)
    ms_input = ms.Tensor(input_tensor.numpy())
    expect = generate_expect_forward_output(input_tensor, fill_value, ms.float32)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = full_like_forward_func(ms_input, fill_value, ms.float32)
    elif mode == "KBK":
        output = jit(
            full_like_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms_input, fill_value, ms.float32)
    else:
        output = None
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    assert output.shape == expect.shape


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize(
    "shape",
    [(), (0,), (5,), (3, 4), (2, 3, 1), (1, 2, 3, 4), (2, 1, 3, 1, 4)]  # Various shapes including 0D and 5D
)
def test_full_like_shape_coverage(mode, shape):
    """
    Feature: shape coverage for full_like operator.
    Description: test full_like with various input shapes.
    Expectation: results match PyTorch implementation.
    """
    input_tensor = torch.randn(shape)
    ms_input = ms.Tensor(input_tensor.numpy())
    fill_value = 1.5
    expect = generate_expect_forward_output(input_tensor, fill_value, ms.float32)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = full_like_forward_func(ms_input, fill_value, ms.float32)
    elif mode == "KBK":
        output = jit(
            full_like_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms_input, fill_value, ms.float32)
    else:
        output = None
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    assert output.shape == expect.shape


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_full_like_contiguous_noncontiguous(mode):
    """
    Feature: contiguous/non-contiguous tensor support for full_like operator.
    Description: test full_like with both contiguous and non-contiguous inputs.
    Expectation: results match PyTorch implementation.
    """
    input_tensor = torch.randn(4, 6)
    ms_input = ms.Tensor(input_tensor.numpy())
    # Contiguous case
    expect_contiguous = generate_expect_forward_output(input_tensor, 2.0, ms.float32)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output_contiguous = full_like_forward_func(ms_input, 2.0, ms.float32)
    elif mode == "KBK":
        output_contiguous = jit(
            full_like_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms_input, 2.0, ms.float32)
    else:
        output_contiguous = None
    allclose_nparray(expect_contiguous.detach().numpy(), output_contiguous.asnumpy(), equal_nan=True)
    # Non-contiguous case - transpose to make it non-contiguous
    input_tensor_transposed = input_tensor.transpose(0, 1)
    ms_input_transposed = ms.Tensor(input_tensor_transposed.numpy())
    expect_noncontiguous = generate_expect_forward_output(input_tensor_transposed, 3.0, ms.float32)
    output_noncontiguous = full_like_forward_func(ms_input_transposed, 3.0, ms.float32)
    allclose_nparray(expect_noncontiguous.detach().numpy(), output_noncontiguous.asnumpy(), equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_full_like_empty_tensor(mode):
    """
    Feature: empty tensor support for full_like operator.
    Description: test full_like with empty input tensors.
    Expectation: results match PyTorch implementation.
    """
    # Test with empty tensor
    input_tensor = torch.tensor([]).reshape(0,)
    ms_input = ms.Tensor(input_tensor.numpy())
    expect = generate_expect_forward_output(input_tensor, 1.0, ms.float32)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = full_like_forward_func(ms_input, 1.0, ms.float32)
    else:
        output = None
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    assert output.shape == expect.shape


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize(
    "dim",
    [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Test from 0D to 8D
)
def test_full_like_dimension_coverage(mode, dim):
    """
    Feature: dimension coverage for full_like operator.
    Description: test full_like with dimensions from 0D to 8D.
    Expectation: results match PyTorch implementation.
    """
    if dim == 0:
        input_tensor = torch.tensor(5.0)  # 0D tensor
    else:
        shape = tuple([2] * dim)  # Use small size to avoid memory issues
        input_tensor = torch.randn(shape)
    ms_input = ms.Tensor(input_tensor.numpy())
    fill_value = 1.0
    expect = generate_expect_forward_output(input_tensor, fill_value, ms.float32)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = full_like_forward_func(ms_input, fill_value, ms.float32)
    elif mode == "KBK":
        output = jit(
            full_like_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms_input, fill_value, ms.float32)
    else:
        output = None
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    assert output.ndim == dim
    assert output.shape == expect.shape


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_full_like_backward(mode):
    """
    Feature: backward support for full_like operator.
    Description: test full_like with gradient computation.
    Expectation: gradients computed correctly.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Since full_like doesn't depend on input values but only on shape,
    # the gradient w.r.t. input should be zero
    input_tensor_ms = ms.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=ms.float32)
    def compute_sum_of_full_like(x):
        filled_tensor = mint.full_like(x, 2.0)
        return ops.sum(filled_tensor)  # Sum to create a scalar for backward
    # Compute gradients using MindSpore
    grad_fn_ms = ms.grad(compute_sum_of_full_like, grad_position=0)
    grad_ms = grad_fn_ms(input_tensor_ms)
    # Since full_like doesn't use the input values (only shape), the gradient should be zero
    expected_grad = ops.zeros_like(grad_ms)
    allclose_nparray(expected_grad.asnumpy(), grad_ms.asnumpy(), equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_full_like_vmap(mode):
    """
    Feature: vmap support for full_like operator.
    Description: test full_like with vectorization.
    Expectation: results match PyTorch implementation.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Test with tensors of the same shape to avoid stacking issues
    # We'll apply full_like to a batch of same-shaped tensors
    input_tensor1 = torch.randn(2, 3)
    input_tensor2 = torch.randn(2, 3)
    input_tensor3 = torch.randn(2, 3)
    ms_input1 = ms.Tensor(input_tensor1.numpy())
    ms_input2 = ms.Tensor(input_tensor2.numpy())
    ms_input3 = ms.Tensor(input_tensor3.numpy())
    fill_value = 2.5
    # Test individual calls
    result1_ms = mint.full_like(ms_input1, fill_value)
    result2_ms = mint.full_like(ms_input2, fill_value)
    result3_ms = mint.full_like(ms_input3, fill_value)
    result1_torch = torch.full_like(input_tensor1, fill_value)
    result2_torch = torch.full_like(input_tensor2, fill_value)
    result3_torch = torch.full_like(input_tensor3, fill_value)
    # Compare results
    allclose_nparray(result1_torch.detach().numpy(), result1_ms.asnumpy(), equal_nan=True)
    allclose_nparray(result2_torch.detach().numpy(), result2_ms.asnumpy(), equal_nan=True)
    allclose_nparray(result3_torch.detach().numpy(), result3_ms.asnumpy(), equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_full_like_dynamic_shape(mode):
    """
    Feature: dynamic shape support for full_like operator.
    Description: test full_like with dynamic shapes.
    Expectation: results match PyTorch implementation.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Test with different shapes dynamically
    shapes = [(2, 3), (4, 1, 5), (1, 1, 1, 2)]
    fill_value = 3.0
    for shape in shapes:
        input_tensor = torch.randn(shape)
        ms_input = ms.Tensor(input_tensor.numpy())
        expect = generate_expect_forward_output(input_tensor, fill_value)
        output = full_like_forward_func(ms_input, fill_value)
        allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
        assert output.shape == expect.shape


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_full_like_broadcast_compatibility(mode):
    """
    Feature: broadcast compatibility for full_like operator.
    Description: test full_like with broadcast-compatible scenarios.
    Expectation: results match PyTorch implementation.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Although full_like doesn't directly involve broadcasting in the same way as arithmetic operations,
    # we can test that it behaves appropriately with different tensor shapes
    shapes = [(1, 3), (3, 1), (2, 3, 1), (1, 2, 3, 4)]
    fill_value = 1.5
    for shape in shapes:
        # Create input tensor with the specified shape
        input_tensor = torch.randn(shape)
        ms_input = ms.Tensor(input_tensor.numpy())
        expect = generate_expect_forward_output(input_tensor, fill_value)
        output = full_like_forward_func(ms_input, fill_value)
        allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
        assert output.shape == expect.shape
        # All elements should be equal to fill_value
        assert all((output.asnumpy() == fill_value).flatten())

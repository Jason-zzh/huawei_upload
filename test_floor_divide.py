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
""" floor_divide op test case """
# pylint: disable=unused-variable
import numpy as np
import mindspore as ms
from mindspore import mint
from mindspore.common.api import _pynative_executor
from tests.utils.tools import allclose_nparray
from tests.utils.mark_utils import arg_mark
from mindspore import vmap
import torch
import pytest


def generate_expect_forward_output(input_tensor, other):
    """Get PyTorch floor_divide forward output."""
    return torch.floor_divide(input_tensor, other)


def floor_divide_forward_func(input_tensor, other):
    return mint.floor_divide(input_tensor, other)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_floor_divide_normal(mode):
    """
    Feature: standard forward functionality for floor_divide.
    Description: test floor_divide op with normal inputs.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_np = np.array([10.0, -20.0, 30.0, -40.0], dtype=np.float32)
    other_np = np.array([3.0, -4.0, 7.0, -8.0], dtype=np.float32)
    input_ms = ms.Tensor(input_np)
    other_ms = ms.Tensor(other_np)
    input_torch = torch.tensor(input_np)
    other_torch = torch.tensor(other_np)
    output_ms = floor_divide_forward_func(input_ms, other_ms)
    expect = generate_expect_forward_output(input_torch, other_torch)
    allclose_nparray(output_ms.asnumpy(), expect.detach().numpy(), 1e-3, 1e-3)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_floor_divide_tensor_scalar(mode):
    """
    Feature: test floor_divide with tensor and scalar.
    Description: test floor_divide op with tensor and scalar inputs.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_np = np.array([10.0, -20.0, 30.0], dtype=np.float32)
    other_scalar = 3.0
    input_ms = ms.Tensor(input_np)
    input_torch = torch.tensor(input_np)
    output_ms = floor_divide_forward_func(input_ms, other_scalar)
    expect = generate_expect_forward_output(input_torch, other_scalar)
    allclose_nparray(output_ms.asnumpy(), expect.detach().numpy(), 1e-3, 1e-3)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_floor_divide_scalar_tensor(mode):
    """
    Feature: test floor_divide with scalar and tensor.
    Description: test floor_divide op with scalar and tensor inputs.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_scalar = 15.0
    other_np = np.array([2.0, -3.0, 4.0], dtype=np.float32)
    other_ms = ms.Tensor(other_np)
    other_torch = torch.tensor(other_np)
    output_ms = floor_divide_forward_func(input_scalar, other_ms)
    expect = generate_expect_forward_output(torch.tensor(input_scalar), other_torch)
    allclose_nparray(output_ms.asnumpy(), expect.detach().numpy(), 1e-3, 1e-3)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_floor_divide_dtype_float64(mode):
    """
    Feature: test floor_divide with float64 dtype.
    Description: test floor_divide op with float64 inputs.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_np = np.array([10.0, -20.0, 30.0], dtype=np.float64)
    other_np = np.array([3.0, -4.0, 7.0], dtype=np.float64)
    input_ms = ms.Tensor(input_np)
    other_ms = ms.Tensor(other_np)
    input_torch = torch.tensor(input_np)
    other_torch = torch.tensor(other_np)
    output_ms = floor_divide_forward_func(input_ms, other_ms)
    expect = generate_expect_forward_output(input_torch, other_torch)
    allclose_nparray(output_ms.asnumpy(), expect.detach().numpy(), 1e-5, 1e-5)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_floor_divide_dtype_int32(mode):
    """
    Feature: test floor_divide with int32 dtype.
    Description: test floor_divide op with int32 inputs.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_np = np.array([10, -20, 30], dtype=np.int32)
    other_np = np.array([3, -4, 7], dtype=np.int32)
    input_ms = ms.Tensor(input_np)
    other_ms = ms.Tensor(other_np)
    input_torch = torch.tensor(input_np)
    other_torch = torch.tensor(other_np)
    output_ms = floor_divide_forward_func(input_ms, other_ms)
    expect = generate_expect_forward_output(input_torch, other_torch)
    allclose_nparray(output_ms.asnumpy(), expect.detach().numpy(), 1e-3, 1e-3)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_floor_divide_with_nan_values(mode):
    """
    Feature: test floor_divide with NaN values only.
    Description: test floor_divide op with NaN values.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Create tensors with only NaN (no infinities to avoid inconsistent behavior)
    input_np = np.array([1.0, np.nan, 5.0, -10.0], dtype=np.float32)
    other_np = np.array([2.0, 3.0, np.nan, 4.0], dtype=np.float32)
    input_ms = ms.Tensor(input_np)
    other_ms = ms.Tensor(other_np)
    input_torch = torch.tensor(input_np)
    other_torch = torch.tensor(other_np)
    output_ms = floor_divide_forward_func(input_ms, other_ms)
    expect = generate_expect_forward_output(input_torch, other_torch)
    allclose_nparray(output_ms.asnumpy(), expect.detach().numpy(), 1e-3, 1e-3, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_floor_divide_different_shapes(mode):
    """
    Feature: test floor_divide with different tensor shapes.
    Description: test floor_divide op with various tensor shapes.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Test with 2D tensors
    input_np = np.random.randn(2, 3).astype(np.float32)
    other_np = np.random.randn(2, 3).astype(np.float32)
    input_ms = ms.Tensor(input_np)
    other_ms = ms.Tensor(other_np)
    input_torch = torch.tensor(input_np)
    other_torch = torch.tensor(other_np)
    output_ms = floor_divide_forward_func(input_ms, other_ms)
    expect = generate_expect_forward_output(input_torch, other_torch)
    allclose_nparray(output_ms.asnumpy(), expect.detach().numpy(), 1e-3, 1e-3)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_floor_divide_broadcasting(mode):
    """
    Feature: test floor_divide with broadcasting.
    Description: test floor_divide op with broadcasting.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Test with broadcasting: (2, 3) and (3,) shapes
    input_np = np.random.randn(2, 3).astype(np.float32)
    other_np = np.random.randn(3).astype(np.float32)
    input_ms = ms.Tensor(input_np)
    other_ms = ms.Tensor(other_np)
    input_torch = torch.tensor(input_np)
    other_torch = torch.tensor(other_np)
    output_ms = floor_divide_forward_func(input_ms, other_ms)
    expect = generate_expect_forward_output(input_torch, other_torch)
    allclose_nparray(output_ms.asnumpy(), expect.detach().numpy(), 1e-3, 1e-3)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_floor_divide_empty_tensors(mode):
    """
    Feature: test floor_divide with empty tensors.
    Description: test floor_divide op with empty tensors.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Test with empty tensors
    input_np = np.empty((0, 3), dtype=np.float32)
    other_np = np.empty((0, 3), dtype=np.float32)
    input_ms = ms.Tensor(input_np)
    other_ms = ms.Tensor(other_np)
    input_torch = torch.tensor(input_np)
    other_torch = torch.tensor(other_np)
    output_ms = floor_divide_forward_func(input_ms, other_ms)
    expect = generate_expect_forward_output(input_torch, other_torch)
    allclose_nparray(output_ms.asnumpy(), expect.detach().numpy(), 1e-3, 1e-3)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_floor_divide_0d_tensors(mode):
    """
    Feature: test floor_divide with 0-dimensional tensors.
    Description: test floor_divide op with 0-dimensional tensors.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Test with 0-dimensional tensors
    input_np = np.array(10.0, dtype=np.float32)
    other_np = np.array(3.0, dtype=np.float32)
    input_ms = ms.Tensor(input_np)
    other_ms = ms.Tensor(other_np)
    input_torch = torch.tensor(input_np)
    other_torch = torch.tensor(other_np)
    output_ms = floor_divide_forward_func(input_ms, other_ms)
    expect = generate_expect_forward_output(input_torch, other_torch)
    allclose_nparray(output_ms.asnumpy(), expect.detach().numpy(), 1e-3, 1e-3)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_floor_divide_vmap(mode):
    """
    Feature: test floor_divide with vmap.
    Description: test floor_divide op with vectorization.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Test vmap functionality
    def floor_divide_func(input_tensor, other):
        return mint.floor_divide(input_tensor, other)
    # Create batched inputs
    batch_size = 2
    input_np = np.random.randn(batch_size, 3).astype(np.float32)
    other_np = np.random.randn(batch_size, 3).astype(np.float32)
    input_ms = ms.Tensor(input_np)
    other_ms = ms.Tensor(other_np)
    # Apply vmap
    vmapped_func = vmap(floor_divide_func, in_axes=(0, 0))
    output_ms = vmapped_func(input_ms, other_ms)
    # Compare with PyTorch
    input_torch = torch.tensor(input_np)
    other_torch = torch.tensor(other_np)
    expected_outputs = []
    for i in range(batch_size):
        expected_out = torch.floor_divide(input_torch[i], other_torch[i])
        expected_outputs.append(expected_out.detach().numpy())
    expected_result = np.stack(expected_outputs)
    allclose_nparray(output_ms.asnumpy(), expected_result, 1e-3, 1e-3)

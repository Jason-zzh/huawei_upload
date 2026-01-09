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
""" addmm op test case """
# pylint: disable=unused-variable
import numpy as np
import mindspore as ms
from mindspore import mint, ops
from mindspore.common.api import _pynative_executor
from tests.utils.tools import allclose_nparray
from tests.utils.mark_utils import arg_mark
import torch
import pytest


def generate_expect_forward_output(input_tensor, mat1, mat2, beta=1, alpha=1):
    """Get PyTorch addmm forward output."""
    input_tensor = torch.tensor(input_tensor)
    mat1 = torch.tensor(mat1)
    mat2 = torch.tensor(mat2)
    return torch.addmm(input_tensor, mat1, mat2, beta=beta, alpha=alpha)


def generate_expect_backward_output(input_tensor, mat1, mat2, grad_output, beta=1, alpha=1):
    """Generate expected backward propagation results using PyTorch."""
    input_tensor = torch.tensor(input_tensor, requires_grad=True)
    mat1 = torch.tensor(mat1, requires_grad=True)
    mat2 = torch.tensor(mat2, requires_grad=True)
    output = torch.addmm(input_tensor, mat1, mat2, beta=beta, alpha=alpha)
    output.backward(torch.tensor(grad_output))  # Convert grad_output to tensor
    return input_tensor.grad, mat1.grad, mat2.grad


def addmm_forward_func(input_tensor, mat1, mat2, beta=1, alpha=1):
    return mint.addmm(input_tensor, mat1, mat2, beta=beta, alpha=alpha)


def addmm_backward_func(input_tensor, mat1, mat2, beta=1, alpha=1):
    def forward_func(input_tensor, mat1, mat2):
        return mint.addmm(input_tensor, mat1, mat2, beta=beta, alpha=alpha)
    return ops.grad(forward_func, (0, 1, 2))(input_tensor, mat1, mat2)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_addmm_normal(mode):
    """
    Feature: standard forward functionality for addmm.
    Description: test addmm op with normal inputs.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
    input_tensor = ms.Tensor(np.random.randn(3, 4).astype(np.float32))
    mat1 = ms.Tensor(np.random.randn(3, 5).astype(np.float32))
    mat2 = ms.Tensor(np.random.randn(5, 4).astype(np.float32))
    ms_output = addmm_forward_func(input_tensor, mat1, mat2)
    expect_output = generate_expect_forward_output(input_tensor.asnumpy(), mat1.asnumpy(), mat2.asnumpy())
    allclose_nparray(ms_output.asnumpy(), expect_output.numpy(), 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_addmm_with_beta_alpha(mode):
    """
    Feature: standard forward functionality for addmm with beta and alpha.
    Description: test addmm op with custom beta and alpha values.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
    input_tensor = ms.Tensor(np.random.randn(2, 3).astype(np.float32))
    mat1 = ms.Tensor(np.random.randn(2, 4).astype(np.float32))
    mat2 = ms.Tensor(np.random.randn(4, 3).astype(np.float32))
    beta = 0.5
    alpha = 2.0
    ms_output = addmm_forward_func(input_tensor, mat1, mat2, beta, alpha)
    expect_output = generate_expect_forward_output(input_tensor.asnumpy(), mat1.asnumpy(), mat2.asnumpy(), beta, alpha)
    allclose_nparray(ms_output.asnumpy(), expect_output.numpy(), 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_addmm_dtypes(mode):
    """
    Feature: dtype coverage for addmm.
    Description: test addmm op with different dtypes.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
    input_tensor = ms.Tensor(np.random.randn(2, 3).astype(np.float32))
    mat1 = ms.Tensor(np.random.randn(2, 4).astype(np.float32))
    mat2 = ms.Tensor(np.random.randn(4, 3).astype(np.float32))
    ms_output = addmm_forward_func(input_tensor, mat1, mat2)
    expect_output = generate_expect_forward_output(input_tensor.asnumpy(), mat1.asnumpy(), mat2.asnumpy())
    allclose_nparray(ms_output.asnumpy(), expect_output.numpy(), 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_addmm_backward(mode):
    """
    Feature: standard backward functionality for addmm.
    Description: test addmm op backward with normal inputs.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
    input_tensor = ms.Tensor(np.random.randn(2, 3).astype(np.float32))
    mat1 = ms.Tensor(np.random.randn(2, 4).astype(np.float32))
    mat2 = ms.Tensor(np.random.randn(4, 3).astype(np.float32))
    grad_input_ms, grad_mat1_ms, grad_mat2_ms = addmm_backward_func(input_tensor, mat1, mat2)
    grad_input_torch, grad_mat1_torch, grad_mat2_torch = generate_expect_backward_output(
        input_tensor.asnumpy(), mat1.asnumpy(), mat2.asnumpy(),
        np.ones_like(input_tensor.asnumpy()))
    allclose_nparray(grad_input_ms.asnumpy(), grad_input_torch.numpy(), 0, 0, equal_nan=True)
    allclose_nparray(grad_mat1_ms.asnumpy(), grad_mat1_torch.numpy(), 0, 0, equal_nan=True)
    allclose_nparray(grad_mat2_ms.asnumpy(), grad_mat2_torch.numpy(), 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_addmm_backward_with_beta_alpha(mode):
    """
    Feature: standard backward functionality for addmm with beta and alpha.
    Description: test addmm op backward with custom beta and alpha values.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
    input_tensor = ms.Tensor(np.random.randn(2, 3).astype(np.float32))
    mat1 = ms.Tensor(np.random.randn(2, 4).astype(np.float32))
    mat2 = ms.Tensor(np.random.randn(4, 3).astype(np.float32))
    beta = 0.5
    alpha = 2.0
    grad_input_ms, grad_mat1_ms, grad_mat2_ms = addmm_backward_func(input_tensor, mat1, mat2, beta, alpha)
    grad_input_torch, grad_mat1_torch, grad_mat2_torch = generate_expect_backward_output(
        input_tensor.asnumpy(), mat1.asnumpy(), mat2.asnumpy(), 
        np.ones_like(input_tensor.asnumpy()), beta, alpha)
    allclose_nparray(grad_input_ms.asnumpy(), grad_input_torch.numpy(), 0, 0, equal_nan=True)
    allclose_nparray(grad_mat1_ms.asnumpy(), grad_mat1_torch.numpy(), 0, 0, equal_nan=True)
    allclose_nparray(grad_mat2_ms.asnumpy(), grad_mat2_torch.numpy(), 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_addmm_zero_matrices():
    """
    Feature: zero matrix handling for addmm.
    Description: test addmm op with zero matrices.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_tensor = ms.Tensor(np.zeros((2, 3), dtype=np.float32))
    mat1 = ms.Tensor(np.zeros((2, 4), dtype=np.float32))
    mat2 = ms.Tensor(np.zeros((4, 3), dtype=np.float32))
    ms_output = addmm_forward_func(input_tensor, mat1, mat2)
    expect_output = generate_expect_forward_output(input_tensor.asnumpy(), mat1.asnumpy(), mat2.asnumpy())
    allclose_nparray(ms_output.asnumpy(), expect_output.numpy(), 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_addmm_large_values():
    """
    Feature: large value handling for addmm.
    Description: test addmm op with large values.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_tensor = ms.Tensor(np.random.uniform(1e6, 1e9, (2, 3)).astype(np.float32))
    mat1 = ms.Tensor(np.random.uniform(1e6, 1e9, (2, 4)).astype(np.float32))
    mat2 = ms.Tensor(np.random.uniform(1e6, 1e9, (4, 3)).astype(np.float32))
    ms_output = addmm_forward_func(input_tensor, mat1, mat2)
    expect_output = generate_expect_forward_output(input_tensor.asnumpy(), mat1.asnumpy(), mat2.asnumpy())
    allclose_nparray(ms_output.asnumpy(), expect_output.numpy(), 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_addmm_small_values():
    """
    Feature: small value handling for addmm.
    Description: test addmm op with very small values.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_tensor = ms.Tensor(np.random.uniform(-1e-9, 1e-9, (2, 3)).astype(np.float32))
    mat1 = ms.Tensor(np.random.uniform(-1e-9, 1e-9, (2, 4)).astype(np.float32))
    mat2 = ms.Tensor(np.random.uniform(-1e-9, 1e-9, (4, 3)).astype(np.float32))
    ms_output = addmm_forward_func(input_tensor, mat1, mat2)
    expect_output = generate_expect_forward_output(input_tensor.asnumpy(), mat1.asnumpy(), mat2.asnumpy())
    allclose_nparray(ms_output.asnumpy(), expect_output.numpy(), 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_addmm_non_contiguous():
    """
    Feature: non-contiguous tensor handling for addmm.
    Description: test addmm op with non-contiguous tensors.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Create larger tensors and slice them to make them non-contiguous
    full_tensor = ms.Tensor(np.random.randn(4, 6).astype(np.float32))
    input_tensor = full_tensor[::2, ::2]  # Non-contiguous slice
    full_mat1 = ms.Tensor(np.random.randn(4, 8).astype(np.float32))
    mat1 = full_mat1[::2, ::2]  # Non-contiguous slice
    full_mat2 = ms.Tensor(np.random.randn(8, 6).astype(np.float32))
    mat2 = full_mat2[::2, ::2]  # Non-contiguous slice
    ms_output = addmm_forward_func(input_tensor, mat1, mat2)
    expect_output = generate_expect_forward_output(input_tensor.asnumpy(), mat1.asnumpy(), mat2.asnumpy())
    allclose_nparray(ms_output.asnumpy(), expect_output.numpy(), 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addmm_dynamic_shape(mode):
    """
    Feature: dynamic shape support for addmm.
    Description: test addmm op with dynamic shapes.
    Expectation: expect correct result.
    """
    del mode
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Test different matrix dimensions
    shapes = [
        ((1, 1), (1, 1), (1, 1)),  # 1x1 matrices
        ((1, 5), (1, 3), (3, 5)),  # Row vector
        ((5, 1), (5, 3), (3, 1)),  # Column vector
        ((10, 10), (10, 10), (10, 10)),  # Square matrices
        ((5, 8), (5, 12), (12, 8)),  # Rectangular matrices
    ]
    for input_shape, mat1_shape, mat2_shape in shapes:
        input_tensor = ms.Tensor(np.random.randn(*input_shape).astype(np.float32))
        mat1 = ms.Tensor(np.random.randn(*mat1_shape).astype(np.float32))
        mat2 = ms.Tensor(np.random.randn(*mat2_shape).astype(np.float32))
        ms_output = addmm_forward_func(input_tensor, mat1, mat2)
        expect_output = generate_expect_forward_output(input_tensor.asnumpy(), mat1.asnumpy(), mat2.asnumpy())
        allclose_nparray(ms_output.asnumpy(), expect_output.numpy(), 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_addmm_broadcasting():
    """
    Feature: broadcasting support for addmm.
    Description: test addmm op with broadcasting scenarios.
    Expectation: expect correct result when input tensor has different dimensions.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Test broadcasting with scalar-like input (1,1)
    input_tensor = ms.Tensor(np.random.randn(1, 1).astype(np.float32))  # Broadcasting from (1,1)
    mat1 = ms.Tensor(np.random.randn(5, 3).astype(np.float32))
    mat2 = ms.Tensor(np.random.randn(3, 4).astype(np.float32))
    ms_output = addmm_forward_func(input_tensor, mat1, mat2)
    expect_output = generate_expect_forward_output(input_tensor.asnumpy(), mat1.asnumpy(), mat2.asnumpy())
    allclose_nparray(ms_output.asnumpy(), expect_output.numpy(), 0, 0, equal_nan=True)
    # Test broadcasting with row vector (1, n)
    input_tensor = ms.Tensor(np.random.randn(1, 4).astype(np.float32))  # Broadcasting row vector
    mat1 = ms.Tensor(np.random.randn(5, 3).astype(np.float32))
    mat2 = ms.Tensor(np.random.randn(3, 4).astype(np.float32))
    ms_output = addmm_forward_func(input_tensor, mat1, mat2)
    expect_output = generate_expect_forward_output(input_tensor.asnumpy(), mat1.asnumpy(), mat2.asnumpy())
    allclose_nparray(ms_output.asnumpy(), expect_output.numpy(), 0, 0, equal_nan=True)
    # Test broadcasting with column vector (m, 1)
    input_tensor = ms.Tensor(np.random.randn(5, 1).astype(np.float32))  # Broadcasting column vector
    mat1 = ms.Tensor(np.random.randn(5, 3).astype(np.float32))
    mat2 = ms.Tensor(np.random.randn(3, 4).astype(np.float32))
    ms_output = addmm_forward_func(input_tensor, mat1, mat2)
    expect_output = generate_expect_forward_output(input_tensor.asnumpy(), mat1.asnumpy(), mat2.asnumpy())
    allclose_nparray(ms_output.asnumpy(), expect_output.numpy(), 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_addmm_empty_tensors():
    """
    Feature: empty tensor handling for addmm.
    Description: test addmm op with empty tensors.
    Expectation: expect correct result or appropriate error handling.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Test with empty input tensor (0, n)
    input_tensor = ms.Tensor(np.random.randn(0, 3).astype(np.float32))
    mat1 = ms.Tensor(np.random.randn(0, 4).astype(np.float32))
    mat2 = ms.Tensor(np.random.randn(4, 3).astype(np.float32))
    ms_output = addmm_forward_func(input_tensor, mat1, mat2)
    expect_output = generate_expect_forward_output(input_tensor.asnumpy(), mat1.asnumpy(), mat2.asnumpy())
    allclose_nparray(ms_output.asnumpy(), expect_output.numpy(), 0, 0, equal_nan=True)
    # Test with empty mat1 tensor
    input_tensor = ms.Tensor(np.random.randn(3, 5).astype(np.float32))
    mat1 = ms.Tensor(np.random.randn(3, 0).astype(np.float32))
    mat2 = ms.Tensor(np.random.randn(0, 5).astype(np.float32))
    ms_output = addmm_forward_func(input_tensor, mat1, mat2)
    expect_output = generate_expect_forward_output(input_tensor.asnumpy(), mat1.asnumpy(), mat2.asnumpy())
    allclose_nparray(ms_output.asnumpy(), expect_output.numpy(), 0, 0, equal_nan=True)
    # Test with empty mat2 tensor (should fail due to dimension mismatch)
    # Note: This would result in mat1 @ mat2 being (3,0) @ (0,5) = (3,5), which should work
    input_tensor = ms.Tensor(np.random.randn(3, 5).astype(np.float32))
    mat1 = ms.Tensor(np.random.randn(3, 0).astype(np.float32))
    mat2 = ms.Tensor(np.random.randn(0, 5).astype(np.float32))
    ms_output = addmm_forward_func(input_tensor, mat1, mat2)
    expect_output = generate_expect_forward_output(input_tensor.asnumpy(), mat1.asnumpy(), mat2.asnumpy())
    allclose_nparray(ms_output.asnumpy(), expect_output.numpy(), 0, 0, equal_nan=True)


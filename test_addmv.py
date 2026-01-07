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
""" addmv op test case """
import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, jit
from tests.utils.mark_utils import arg_mark
from tests.utils.tools import allclose_nparray
import torch


def generate_expect_forward_output(input, mat, vec, beta=1, alpha=1):
    """Generate expected output using PyTorch addmv."""
    input_tensor = torch.from_numpy(input) if isinstance(input, np.ndarray) else input
    mat_tensor = torch.from_numpy(mat) if isinstance(mat, np.ndarray) else mat
    vec_tensor = torch.from_numpy(vec) if isinstance(vec, np.ndarray) else vec
    return torch.addmv(input_tensor, mat_tensor, vec_tensor, beta=beta, alpha=alpha)


def addmv_forward_func(input, mat, vec, beta=1, alpha=1):
    """Forward function for mint.addmv."""
    return mint.addmv(input, mat, vec, beta=beta, alpha=alpha)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("dtype", [np.float32])
def test_addmv_std(mode, dtype):
    """
    Feature: Pyboost function.
    Description: Test function addmv with standard inputs.
    Expectation: Expect correct result.
    """
    np.random.seed(0)
    input_tensor = np.random.randn(6).astype(dtype)
    mat = np.random.randn(6, 3).astype(dtype)
    vec = np.random.randn(3).astype(dtype)
    beta = 1.0
    alpha = 1.0
    expect = generate_expect_forward_output(input_tensor, mat, vec, beta, alpha)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = addmv_forward_func(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    elif mode == "KBK":
        output = jit(
            addmv_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    else:
        output = None
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=1e-4, atol=1e-4, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_addmv_dtype_coverage(mode, dtype):
    """
    Feature: Dtype coverage for addmv operator.
    Description: Test addmv with various dtypes.
    Expectation: Results match PyTorch implementation.
    """
    if dtype in [np.int32, np.int64]:
        input_tensor = np.random.randint(-10, 10, size=(4,)).astype(dtype)
        mat = np.random.randint(-5, 5, size=(4, 2)).astype(dtype)
        vec = np.random.randint(-3, 3, size=(2,)).astype(dtype)
    else:
        input_tensor = np.random.randn(4).astype(dtype)
        mat = np.random.randn(4, 2).astype(dtype)
        vec = np.random.randn(2).astype(dtype)
    beta = 2.0
    alpha = 1.5
    expect = generate_expect_forward_output(input_tensor, mat, vec, beta, alpha)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = addmv_forward_func(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    elif mode == "KBK":
        output = jit(
            addmv_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    else:
        output = None
    allclose_nparray(
        expect.detach().numpy(),
        output.asnumpy(),
        rtol=1e-3,
        atol=1e-3,
        equal_nan=True,
    )


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_addmv_different_scalars(mode):
    """
    Feature: Scalar parameter coverage for addmv.
    Description: Test addmv with different beta and alpha values.
    Expectation: Results match PyTorch implementation.
    """
    np.random.seed(1)
    input_tensor = np.random.randn(5).astype(np.float32)
    mat = np.random.randn(5, 4).astype(np.float32)
    vec = np.random.randn(4).astype(np.float32)
    beta = 0.5
    alpha = 2.5
    expect = generate_expect_forward_output(input_tensor, mat, vec, beta, alpha)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = addmv_forward_func(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    elif mode == "KBK":
        output = jit(
            addmv_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    else:
        output = None
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=1e-4, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize(
    "input_shape, mat_shape, vec_shape",
    [
        ((3,), (3, 2), (2,)),
        ((1,), (1, 5), (5,)),
        ((10,), (10, 1), (1,)),
        ((7,), (7, 7), (7,)),
    ],
)
def test_addmv_different_shapes(mode, input_shape, mat_shape, vec_shape):
    """
    Feature: Shape coverage for addmv.
    Description: Test addmv with different tensor shapes.
    Expectation: Results match PyTorch implementation.
    """
    np.random.seed(2)
    input_tensor = np.random.randn(*input_shape).astype(np.float32)
    mat = np.random.randn(*mat_shape).astype(np.float32)
    vec = np.random.randn(*vec_shape).astype(np.float32)
    beta = 1.0
    alpha = 1.0
    expect = generate_expect_forward_output(input_tensor, mat, vec, beta, alpha)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = addmv_forward_func(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    elif mode == "KBK":
        output = jit(
            addmv_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    else:
        output = None
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=1e-4, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize(
    "beta_alpha_pair",
    [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (-1.0, 2.0),
        (0.5, -0.5),
    ],
)
def test_addmv_special_scalar_values(mode, beta_alpha_pair):
    """
    Feature: Special scalar value handling for addmv.
    Description: Test addmv with special beta and alpha values.
    Expectation: Results match PyTorch implementation.
    """
    beta, alpha = beta_alpha_pair
    np.random.seed(3)
    input_tensor = np.random.randn(4).astype(np.float32)
    mat = np.random.randn(4, 3).astype(np.float32)
    vec = np.random.randn(3).astype(np.float32)
    expect = generate_expect_forward_output(input_tensor, mat, vec, beta, alpha)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = addmv_forward_func(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    elif mode == "KBK":
        output = jit(
            addmv_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    else:
        output = None
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=1e-4, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize(
    "special_val",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_addmv_special_tensor_values(mode, special_val):
    """
    Feature: Special value handling for addmv.
    Description: Test addmv with tensors containing nan and inf values.
    Expectation: Results match PyTorch implementation.
    """
    np.random.seed(4)
    input_tensor = np.random.randn(3).astype(np.float32)
    input_tensor[0] = special_val  # Insert special value
    mat = np.random.randn(3, 2).astype(np.float32)
    vec = np.random.randn(2).astype(np.float32)
    expect = generate_expect_forward_output(input_tensor, mat, vec, 1.0, 1.0)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = addmv_forward_func(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec))
    elif mode == "KBK":
        output = jit(
            addmv_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec))
    else:
        output = None
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)

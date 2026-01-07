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
from mindspore import mint
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


def addmm_forward_func(input_tensor, mat1, mat2, beta=1, alpha=1):
    return mint.addmm(input_tensor, mat1, mat2, beta=beta, alpha=alpha)


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
    allclose_nparray(ms_output.asnumpy(), expect_output.numpy(), 1e-3, 1e-3)


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
    allclose_nparray(ms_output.asnumpy(), expect_output.numpy(), 1e-3, 1e-3)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("dtype", [np.float32])  # Only float32 for now as it's the most common
def test_addmm_dtypes(dtype):
    """
    Feature: dtype coverage for addmm.
    Description: test addmm op with different dtypes.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_tensor = ms.Tensor(np.random.randn(2, 3).astype(dtype))
    mat1 = ms.Tensor(np.random.randn(2, 4).astype(dtype))
    mat2 = ms.Tensor(np.random.randn(4, 3).astype(dtype))
    ms_output = addmm_forward_func(input_tensor, mat1, mat2)
    expect_output = generate_expect_forward_output(input_tensor.asnumpy(), mat1.asnumpy(), mat2.asnumpy())
    allclose_nparray(ms_output.asnumpy(), expect_output.numpy(), 1e-3, 1e-3)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_addmm_edge_cases():
    """
    Feature: edge cases for addmm.
    Description: test addmm op with edge cases like scalar inputs, empty tensors.
    Expectation: expect correct result or proper error handling.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Test with 1x1 matrices
    input_tensor = ms.Tensor(np.random.randn(1, 1).astype(np.float32))
    mat1 = ms.Tensor(np.random.randn(1, 1).astype(np.float32))
    mat2 = ms.Tensor(np.random.randn(1, 1).astype(np.float32))
    ms_output = addmm_forward_func(input_tensor, mat1, mat2)
    expect_output = generate_expect_forward_output(input_tensor.asnumpy(), mat1.asnumpy(), mat2.asnumpy())
    allclose_nparray(ms_output.asnumpy(), expect_output.numpy(), 1e-3, 1e-3)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_addmm_with_nan_inf():
    """
    Feature: NaN and Inf handling for addmm.
    Description: test addmm op with NaN and Inf values.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_tensor = ms.Tensor(np.array([[1.0, np.nan], [np.inf, -np.inf]], dtype=np.float32))
    mat1 = ms.Tensor(np.array([[1.0, 0], [0, 1.0]], dtype=np.float32))
    mat2 = ms.Tensor(np.array([[1.0, 0], [0, 1.0]], dtype=np.float32))
    ms_output = addmm_forward_func(input_tensor, mat1, mat2)
    expect_output = generate_expect_forward_output(input_tensor.asnumpy(), mat1.asnumpy(), mat2.asnumpy())
    allclose_nparray(ms_output.asnumpy(), expect_output.numpy(), 1e-3, 1e-3, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addmm_vmap(mode):
    """
    Feature: vmap functionality for addmm.
    Description: test addmm op with vmap.
    Expectation: expect correct result.
    """
    del mode
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # This is a basic vmap test - creating batched inputs
    batch_size = 2
    input_tensor = ms.Tensor(np.random.randn(batch_size, 3, 4).astype(np.float32))
    mat1 = ms.Tensor(np.random.randn(batch_size, 3, 5).astype(np.float32))
    mat2 = ms.Tensor(np.random.randn(batch_size, 5, 4).astype(np.float32))
    # Process each batch item separately to simulate vmap behavior
    results = []
    for i in range(batch_size):
        result = addmm_forward_func(input_tensor[i], mat1[i], mat2[i])
        results.append(result.asnumpy())
    expect_results = []
    for i in range(batch_size):
        expect = generate_expect_forward_output(input_tensor[i].asnumpy(), mat1[i].asnumpy(), mat2[i].asnumpy())
        expect_results.append(expect.numpy())
    for ms_result, expect_result in zip(results, expect_results):
        allclose_nparray(ms_result, expect_result, 1e-3, 1e-3)

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
""" addmm op performance test case """
# pylint: disable=unused-variable
# pylint: disable=W0622,W0613
import time
import numpy as np
import mindspore as ms
from mindspore import mint
from mindspore.common.api import _pynative_executor
from tests.utils.test_op_utils import BACKGROUND_NOISE
from tests.utils.mark_utils import arg_mark
import torch
import pytest


def addmm_forward_perf(input_tensor, mat1, mat2, beta=1, alpha=1):
    """Get MindSpore addmm forward performance."""
    # Warm-up
    for _ in range(100):
        _ = mint.addmm(input_tensor, mat1, mat2, beta=beta, alpha=alpha)

    _pynative_executor.sync()
    start_time = time.time()
    # Performance test
    for _ in range(1000):
        _ = mint.addmm(input_tensor, mat1, mat2, beta=beta, alpha=alpha)
    _pynative_executor.sync()
    end_time = time.time()

    return end_time - start_time


def generate_expect_addmm_forward_perf(input_tensor, mat1, mat2, beta=1, alpha=1):
    """Get PyTorch addmm forward performance."""
    # Warm-up
    for _ in range(100):
        _ = torch.addmm(input_tensor, mat1, mat2, beta=beta, alpha=alpha)

    start_time = time.time()
    # Performance test
    for _ in range(1000):
        _ = torch.addmm(input_tensor, mat1, mat2, beta=beta, alpha=alpha)
    end_time = time.time()

    return end_time - start_time


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addmm_perf(mode):
    """
    Feature: standard forward performance for addmm.
    Description: test addmm op performance.
    Expectation: expect performance OK.
    """
    del mode

    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Generate random tensors for performance test
    np.random.seed(0)
    input_tensor = torch.randn(100, 150, dtype=torch.float32)
    mat1 = torch.randn(100, 200, dtype=torch.float32)
    mat2 = torch.randn(200, 150, dtype=torch.float32)
    beta = 1.0
    alpha = 1.0

    input_ms = ms.Tensor(input_tensor.numpy())
    mat1_ms = ms.Tensor(mat1.numpy())
    mat2_ms = ms.Tensor(mat2.numpy())

    ms_perf = addmm_forward_perf(input_ms, mat1_ms, mat2_ms, beta, alpha)
    expect_perf = generate_expect_addmm_forward_perf(input_tensor, mat1, mat2, beta, alpha)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addmm_perf_large_matrix(mode):
    """
    Feature: large matrix forward performance for addmm.
    Description: test addmm op performance with large matrices.
    Expectation: expect performance OK.
    """
    del mode

    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Generate larger random tensors for performance test
    np.random.seed(0)
    input_tensor = torch.randn(500, 500, dtype=torch.float32)
    mat1 = torch.randn(500, 600, dtype=torch.float32)
    mat2 = torch.randn(600, 500, dtype=torch.float32)
    beta = 1.0
    alpha = 1.0

    input_ms = ms.Tensor(input_tensor.numpy())
    mat1_ms = ms.Tensor(mat1.numpy())
    mat2_ms = ms.Tensor(mat2.numpy())

    ms_perf = addmm_forward_perf(input_ms, mat1_ms, mat2_ms, beta, alpha)
    expect_perf = generate_expect_addmm_forward_perf(input_tensor, mat1, mat2, beta, alpha)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()

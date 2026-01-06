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
""" addbmm op performance test case """
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


def addbmm_forward_perf(input_tensor, batch1, batch2, beta=1, alpha=1):
    """Get MindSpore addbmm forward performance."""
    # Warm-up
    for _ in range(100):
        _ = mint.addbmm(input_tensor, batch1, batch2, beta=beta, alpha=alpha)

    _pynative_executor.sync()
    start_time = time.time()
    # Performance test
    for _ in range(1000):
        _ = mint.addbmm(input_tensor, batch1, batch2, beta=beta, alpha=alpha)
    _pynative_executor.sync()
    end_time = time.time()

    return end_time - start_time


def generate_expect_addbmm_forward_perf(input_tensor, batch1, batch2, beta=1, alpha=1):
    """Get PyTorch addbmm forward performance."""
    # Warm-up
    for _ in range(100):
        _ = torch.addbmm(input_tensor, batch1, batch2, beta=beta, alpha=alpha)

    start_time = time.time()
    # Performance test
    for _ in range(1000):
        _ = torch.addbmm(input_tensor, batch1, batch2, beta=beta, alpha=alpha)
    end_time = time.time()

    return end_time - start_time


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addbmm_perf(mode):
    """
    Feature: standard forward performance for addbmm.
    Description: test addbmm op performance.
    Expectation: expect performance OK.
    """
    del mode

    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Generate test inputs
    np.random.seed(0)
    input_shape = (10, 10)
    batch1_shape = (20, 10, 15)
    batch2_shape = (20, 15, 10)

    input_np = np.random.randn(*input_shape).astype(np.float32)
    batch1_np = np.random.randn(*batch1_shape).astype(np.float32)
    batch2_np = np.random.randn(*batch2_shape).astype(np.float32)

    input_ms = ms.Tensor(input_np)
    batch1_ms = ms.Tensor(batch1_np)
    batch2_ms = ms.Tensor(batch2_np)

    input_torch = torch.tensor(input_np)
    batch1_torch = torch.tensor(batch1_np)
    batch2_torch = torch.tensor(batch2_np)

    ms_perf = addbmm_forward_perf(input_ms, batch1_ms, batch2_ms)
    expect_perf = generate_expect_addbmm_forward_perf(input_torch, batch1_torch, batch2_torch)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addbmm_perf_with_beta_alpha(mode):
    """
    Feature: performance for addbmm with beta and alpha.
    Description: test addbmm op performance with custom parameters.
    Expectation: expect performance OK.
    """
    del mode

    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Generate test inputs
    np.random.seed(0)
    input_shape = (5, 5)
    batch1_shape = (10, 5, 8)
    batch2_shape = (10, 8, 5)

    input_np = np.random.randn(*input_shape).astype(np.float32)
    batch1_np = np.random.randn(*batch1_shape).astype(np.float32)
    batch2_np = np.random.randn(*batch2_shape).astype(np.float32)

    input_ms = ms.Tensor(input_np)
    batch1_ms = ms.Tensor(batch1_np)
    batch2_ms = ms.Tensor(batch2_np)

    input_torch = torch.tensor(input_np)
    batch1_torch = torch.tensor(batch1_np)
    batch2_torch = torch.tensor(batch2_np)

    beta, alpha = 0.5, 2.0

    ms_perf = addbmm_forward_perf(input_ms, batch1_ms, batch2_ms, beta=beta, alpha=alpha)
    expect_perf = generate_expect_addbmm_forward_perf(input_torch, batch1_torch, batch2_torch, beta=beta, alpha=alpha)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()

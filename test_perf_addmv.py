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
""" addmv op performance test case """
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


def addmv_forward_perf(input_tensor, mat, vec, beta=1, alpha=1):
    """Get MindSpore addmv forward performance."""
    # Warm-up
    for _ in range(100):
        _ = mint.addmv(input_tensor, mat, vec, beta=beta, alpha=alpha)
    _pynative_executor.sync()
    start_time = time.time()
    # Performance test
    for _ in range(100):
        _ = mint.addmv(input_tensor, mat, vec, beta=beta, alpha=alpha)
    _pynative_executor.sync()
    end_time = time.time()
    return end_time - start_time


def generate_expect_addmv_forward_perf(input_tensor, mat, vec, beta=1, alpha=1):
    """Get PyTorch addmv forward performance."""
    # Warm-up
    for _ in range(100):
        _ = torch.addmv(input_tensor, mat, vec, beta=beta, alpha=alpha)
    start_time = time.time()
    # Performance test
    for _ in range(100):
        _ = torch.addmv(input_tensor, mat, vec, beta=beta, alpha=alpha)
    end_time = time.time()
    return end_time - start_time


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addmv_perf(mode):
    """
    Feature: Standard forward performance for addmv.
    Description: Test addmv op performance.
    Expectation: Expect performance OK.
    """
    del mode
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Generate random tensors for performance test
    np.random.seed(0)
    input_tensor = ms.Tensor(np.random.randn(1000).astype(np.float32))
    mat = ms.Tensor(np.random.randn(1000, 500).astype(np.float32))
    vec = ms.Tensor(np.random.randn(500).astype(np.float32))
    beta = 1.0
    alpha = 1.0
    # Convert to torch tensors for performance comparison
    torch_input = torch.from_numpy(input_tensor.asnumpy())
    torch_mat = torch.from_numpy(mat.asnumpy())
    torch_vec = torch.from_numpy(vec.asnumpy())
    ms_perf = addmv_forward_perf(input_tensor, mat, vec, beta, alpha)
    expect_perf = generate_expect_addmv_forward_perf(torch_input, torch_mat, torch_vec, beta, alpha)
    # Check that MindSpore performance is within 1.1x of Torch baseline
    assert ms_perf - BACKGROUND_NOISE <= expect_perf * 1.1, \
        f"MindSpore performance ({ms_perf}) exceeds Torch baseline ({expect_perf}) by more than 1.1x"

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
""" floor_divide op performance test case """
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


def floor_divide_forward_perf(input_tensor, other):
    """Get MindSpore floor_divide forward performance."""
    # Warm-up
    for _ in range(100):
        _ = mint.floor_divide(input_tensor, other)
    _pynative_executor.sync()
    start_time = time.time()
    # Performance test
    for _ in range(1000):
        _ = mint.floor_divide(input_tensor, other)
    _pynative_executor.sync()
    end_time = time.time()
    return end_time - start_time


def generate_expect_floor_divide_forward_perf(input_tensor, other):
    """Get PyTorch floor_divide forward performance."""
    # Warm-up
    for _ in range(100):
        _ = torch.floor_divide(input_tensor, other)
    start_time = time.time()
    # Performance test
    for _ in range(1000):
        _ = torch.floor_divide(input_tensor, other)
    end_time = time.time()
    return end_time - start_time


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_floor_divide_perf(mode):
    """
    Feature: standard forward performance for floor_divide.
    Description: test floor_divide op performance.
    Expectation: expect performance OK.
    """
    del mode
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Generate test inputs
    np.random.seed(0)
    input_shape = (1000, 1000)
    input_np = np.random.uniform(-100, 100, size=input_shape).astype(np.float32)
    other_np = np.random.uniform(-10, 10, size=input_shape).astype(np.float32)
    # Avoid zeros to prevent division by zero
    other_np = np.where(other_np == 0, 1.0, other_np)
    input_ms = ms.Tensor(input_np)
    other_ms = ms.Tensor(other_np)
    input_torch = torch.tensor(input_np)
    other_torch = torch.tensor(other_np)
    ms_perf = floor_divide_forward_perf(input_ms, other_ms)
    expect_perf = generate_expect_floor_divide_forward_perf(input_torch, other_torch)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_floor_divide_perf_tensor_scalar(mode):
    """
    Feature: performance for floor_divide with tensor and scalar.
    Description: test floor_divide op performance with tensor and scalar.
    Expectation: expect performance OK.
    """
    del mode
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Generate test inputs
    np.random.seed(0)
    input_shape = (1000, 1000)
    input_np = np.random.uniform(-100, 100, size=input_shape).astype(np.float32)
    other_scalar = 2.5
    input_ms = ms.Tensor(input_np)
    input_torch = torch.tensor(input_np)
    ms_perf = floor_divide_forward_perf(input_ms, other_scalar)
    expect_perf = generate_expect_floor_divide_forward_perf(input_torch, other_scalar)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_floor_divide_perf_small_tensors(mode):
    """
    Feature: performance for floor_divide with small tensors.
    Description: test floor_divide op performance with small tensors.
    Expectation: expect performance OK.
    """
    del mode
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Generate test inputs
    np.random.seed(0)
    input_shape = (100, 100)
    input_np = np.random.uniform(-50, 50, size=input_shape).astype(np.float32)
    other_np = np.random.uniform(-5, 5, size=input_shape).astype(np.float32)
    # Avoid zeros to prevent division by zero
    other_np = np.where(other_np == 0, 1.0, other_np)
    input_ms = ms.Tensor(input_np)
    other_ms = ms.Tensor(other_np)
    input_torch = torch.tensor(input_np)
    other_torch = torch.tensor(other_np)
    ms_perf = floor_divide_forward_perf(input_ms, other_ms)
    expect_perf = generate_expect_floor_divide_forward_perf(input_torch, other_torch)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_floor_divide_perf_int32(mode):
    """
    Feature: performance for floor_divide with int32 tensors.
    Description: test floor_divide op performance with int32 tensors.
    Expectation: expect performance OK.
    """
    del mode
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Generate test inputs
    np.random.seed(0)
    input_shape = (1000, 1000)
    input_np = np.random.randint(-50, 50, size=input_shape).astype(np.int32)
    other_np = np.random.randint(-10, 10, size=input_shape).astype(np.int32)
    # Avoid zeros to prevent division by zero
    other_np = np.where(other_np == 0, 1, other_np)
    input_ms = ms.Tensor(input_np)
    other_ms = ms.Tensor(other_np)
    input_torch = torch.tensor(input_np)
    other_torch = torch.tensor(other_np)
    ms_perf = floor_divide_forward_perf(input_ms, other_ms)
    expect_perf = generate_expect_floor_divide_forward_perf(input_torch, other_torch)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()

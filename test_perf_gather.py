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
""" gather op performance test case """
# pylint: disable=unused-variable
# pylint: disable=W0622,W0613
import time
import mindspore as ms
from mindspore import mint
from mindspore.common.api import _pynative_executor
from tests.utils.test_op_utils import BACKGROUND_NOISE
from tests.utils.mark_utils import arg_mark
import torch
import numpy as np
import pytest


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_random_index(shape, max_idx):
    """Generate random index data within range [0, max_idx)"""
    return np.random.randint(0, max_idx, shape).astype(np.int64)  # Changed to int64


def gather_forward_perf(input, dim, index):
    """get ms op forward performance"""
    op = mint.gather
    print("================shape: ", input.shape)
    for _ in range(1000):
        output = op(input, dim, index)
    _pynative_executor.sync()
    start = time.time()
    for _ in range(1000):
        output = op(input, dim, index)
    _pynative_executor.sync()
    end = time.time()
    print(f"MindSpore {op} e2e time: ", (end - start))
    return end - start


def generate_expect_forward_perf(input, dim, index):
    """get torch op forward performance"""
    op = torch.gather
    print("================shape: ", input.shape)
    for _ in range(1000):
        output = op(input, dim, index)
    start = time.time()
    for _ in range(1000):
        output = op(input, dim, index)
    end = time.time()
    print(f"Torch {op} e2e time: ", end - start)
    return end - start


@arg_mark(plat_marks=['cpu_linux'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_gather_perf_basic(mode):
    """
    Feature: gather forward performance.
    Description: test gather op performance between MindSpore and PyTorch.
    Expectation: MindSpore performance is within 110% of PyTorch performance.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    
    shape = (1000, 500)
    dim = 0
    index_shape = (100, 500)
    
    input_np = generate_random_input(shape, np.float32)
    index_np = generate_random_index(index_shape, shape[dim])
    
    ms_input = ms.Tensor(input_np)
    ms_index = ms.Tensor(index_np)
    torch_input = torch.tensor(input_np, dtype=torch.float32)
    torch_index = torch.tensor(index_np, dtype=torch.int64)  # Changed to int64
    
    ms_perf = gather_forward_perf(ms_input, dim, ms_index)
    expect_perf = generate_expect_forward_perf(torch_input, dim, torch_index)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_gather_perf_3d(mode):
    """
    Feature: gather forward performance on 3D tensors.
    Description: test gather op performance between MindSpore and PyTorch with 3D tensors.
    Expectation: MindSpore performance is within 110% of PyTorch performance.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    
    shape = (100, 200, 300)
    dim = 1
    index_shape = (100, 50, 300)
    
    input_np = generate_random_input(shape, np.float32)
    index_np = generate_random_index(index_shape, shape[dim])
    
    ms_input = ms.Tensor(input_np)
    ms_index = ms.Tensor(index_np)
    torch_input = torch.tensor(input_np, dtype=torch.float32)
    torch_index = torch.tensor(index_np, dtype=torch.int64)  # Changed to int64
    
    ms_perf = gather_forward_perf(ms_input, dim, ms_index)
    expect_perf = generate_expect_forward_perf(torch_input, dim, torch_index)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_gather_perf_large(mode):
    """
    Feature: gather forward performance on large tensors.
    Description: test gather op performance between MindSpore and PyTorch with large tensors.
    Expectation: MindSpore performance is within 110% of PyTorch performance.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    
    shape = (5000, 1000)
    dim = 0
    index_shape = (500, 1000)
    
    input_np = generate_random_input(shape, np.float32)
    index_np = generate_random_index(index_shape, shape[dim])
    
    ms_input = ms.Tensor(input_np)
    ms_index = ms.Tensor(index_np)
    torch_input = torch.tensor(input_np, dtype=torch.float32)
    torch_index = torch.tensor(index_np, dtype=torch.int64)  # Changed to int64
    
    ms_perf = gather_forward_perf(ms_input, dim, ms_index)
    expect_perf = generate_expect_forward_perf(torch_input, dim, torch_index)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()

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
""" fmod op performance test case """
# pylint: disable=unused-variable
# pylint: disable=W0622,W0613
import time
import random
import mindspore as ms
from mindspore import mint
from mindspore.common.api import _pynative_executor
from tests.utils.test_op_utils import BACKGROUND_NOISE
from tests.utils.mark_utils import arg_mark
import torch
import numpy as np
import pytest


def generate_random_input(shape):
    return np.random.uniform(-10, 10, shape)


def generate_scalar_input():
    return random.uniform(-10, 10)


def fmod_forward_perf(input, other):
    """get ms op forward performance"""
    op = mint.fmod

    for _ in range(1000):
        output = op(input, other)

    _pynative_executor.sync()
    start = time.time()
    for _ in range(1000):
        output = op(input, other)
    _pynative_executor.sync()
    end = time.time()

    print(f"MindSpore {op} e2e time: ", (end-start))
    return  end-start


def generate_expect_forward_perf(input, other):
    """get torch op forward performance"""
    op = torch.fmod

    for _ in range(1000):
        output = op(input, other)

    start = time.time()
    for _ in range(1000):
        output = op(input, other)
    end = time.time()

    print(f"Torch {op} e2e time: ", end-start)
    return end-start


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_fmod_perf(mode):
    """
    Feature: standard forward performance.
    Description: test fmod op performance.
    Expectation: expect performance OK.
    """
    shape = (10, 10, 10, 10)
    input = generate_random_input(shape)
    other = generate_random_input(shape)
    ms_perf = fmod_forward_perf(ms.Tensor(input), ms.Tensor(other))
    expect_perf = generate_expect_forward_perf(torch.Tensor(input), torch.Tensor(other))
    # Allow 1.1x performance factor as specified in the requirements
    assert ms_perf <= expect_perf * 1.1 + BACKGROUND_NOISE


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_fmod_scalar_perf(mode):
    """
    Feature: scalar performance.
    Description: test fmod op performance with scalar.
    Expectation: expect performance OK.
    """
    shape = (10, 10, 10, 10)
    input = generate_random_input(shape)
    other = generate_scalar_input()
    ms_perf = fmod_forward_perf(ms.Tensor(input), other)
    expect_perf = generate_expect_forward_perf(torch.Tensor(input), other)
    # Allow 1.1x performance factor as specified in the requirements
    assert ms_perf <= expect_perf * 1.1 + BACKGROUND_NOISE


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_fmod_broadcast_perf(mode):
    """
    Feature: broadcast performance.
    Description: test fmod op performance with broadcasting.
    Expectation: expect performance OK.
    """
    shape1 = (10, 10, 10, 10)
    shape2 = (1, 10, 1, 1)
    input = generate_random_input(shape1)
    other = generate_random_input(shape2)
    ms_perf = fmod_forward_perf(ms.Tensor(input), ms.Tensor(other))
    expect_perf = generate_expect_forward_perf(torch.Tensor(input), torch.Tensor(other))
    # Allow 1.1x performance factor as specified in the requirements
    assert ms_perf <= expect_perf * 1.1 + BACKGROUND_NOISE

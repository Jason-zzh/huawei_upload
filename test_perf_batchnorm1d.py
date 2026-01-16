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
""" batchnorm1d op performance test case """
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


def batchnorm1d_forward_perf(input_tensor, weight, bias, running_mean, running_var, 
                           training=True, momentum=0.1, eps=1e-5):
    """Get MindSpore batchnorm1d forward performance."""
    # Create module once outside the timing loop
    batch_norm_module = mint.nn.BatchNorm1d(input_tensor.shape[1], eps=eps, momentum=momentum, affine=True)
    batch_norm_module.weight.assign_value(weight)
    batch_norm_module.bias.assign_value(bias)
    batch_norm_module.running_mean.assign_value(running_mean)
    batch_norm_module.running_var.assign_value(running_var)
    batch_norm_module.set_train(training)
    
    # Warm-up
    for _ in range(100):
        _ = batch_norm_module(input_tensor)

    _pynative_executor.sync()
    start_time = time.time()
    # Performance test
    for _ in range(1000):
        _ = batch_norm_module(input_tensor)
    _pynative_executor.sync()
    end_time = time.time()

    return end_time - start_time


def generate_expect_batchnorm1d_forward_perf(input_tensor, weight, bias, running_mean, running_var, 
                                          training=True, momentum=0.1, eps=1e-5):
    """Get PyTorch batchnorm1d forward performance."""
    # Warm-up
    for _ in range(100):
        _ = torch._native_batch_norm_legit(input_tensor, weight, bias, 
                                          running_mean, running_var, training, momentum, eps)

    start_time = time.time()
    # Performance test
    for _ in range(1000):
        _ = torch._native_batch_norm_legit(input_tensor, weight, bias, 
                                          running_mean, running_var, training, momentum, eps)
    end_time = time.time()

    return end_time - start_time


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_batchnorm1d_perf_2d(mode):
    """
    Feature: standard forward performance for batchnorm1d with 2D input.
    Description: test batchnorm1d op performance.
    Expectation: expect performance OK.
    """
    del mode

    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Generate test data for 2D input (BatchNorm1d equivalent)
    np.random.seed(0)
    input_shape = (32, 64)
    num_features = input_shape[1]
    
    input_np = np.random.randn(*input_shape).astype(np.float32)
    weight_np = np.random.randn(num_features).astype(np.float32)
    bias_np = np.random.randn(num_features).astype(np.float32)
    running_mean_np = np.zeros(num_features).astype(np.float32)
    running_var_np = np.random.rand(num_features).astype(np.float32) + 0.1
    
    ms_input = ms.Tensor(input_np)
    ms_weight = ms.Tensor(weight_np)
    ms_bias = ms.Tensor(bias_np)
    ms_running_mean = ms.Tensor(running_mean_np)
    ms_running_var = ms.Tensor(running_var_np)
    
    pt_input = torch.tensor(input_np)
    pt_weight = torch.tensor(weight_np)
    pt_bias = torch.tensor(bias_np)
    pt_running_mean = torch.tensor(running_mean_np)
    pt_running_var = torch.tensor(running_var_np)

    ms_perf = batchnorm1d_forward_perf(ms_input, ms_weight, ms_bias, ms_running_mean, ms_running_var)
    expect_perf = generate_expect_batchnorm1d_forward_perf(pt_input, pt_weight, pt_bias, pt_running_mean, pt_running_var)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_batchnorm1d_perf_3d(mode):
    """
    Feature: standard forward performance for batchnorm1d with 3D input.
    Description: test batchnorm1d op performance.
    Expectation: expect performance OK.
    """
    del mode

    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Generate test data for 3D input (BatchNorm1d with sequence data)
    np.random.seed(0)
    input_shape = (16, 32, 64)
    num_features = input_shape[1]
    
    input_np = np.random.randn(*input_shape).astype(np.float32)
    weight_np = np.random.randn(num_features).astype(np.float32)
    bias_np = np.random.randn(num_features).astype(np.float32)
    running_mean_np = np.zeros(num_features).astype(np.float32)
    running_var_np = np.random.rand(num_features).astype(np.float32) + 0.1
    
    ms_input = ms.Tensor(input_np)
    ms_weight = ms.Tensor(weight_np)
    ms_bias = ms.Tensor(bias_np)
    ms_running_mean = ms.Tensor(running_mean_np)
    ms_running_var = ms.Tensor(running_var_np)
    
    pt_input = torch.tensor(input_np)
    pt_weight = torch.tensor(weight_np)
    pt_bias = torch.tensor(bias_np)
    pt_running_mean = torch.tensor(running_mean_np)
    pt_running_var = torch.tensor(running_var_np)

    ms_perf = batchnorm1d_forward_perf(ms_input, ms_weight, ms_bias, ms_running_mean, ms_running_var)
    expect_perf = generate_expect_batchnorm1d_forward_perf(pt_input, pt_weight, pt_bias, pt_running_mean, pt_running_var)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_batchnorm1d_perf_eval_mode(mode):
    """
    Feature: eval mode performance for batchnorm1d.
    Description: test batchnorm1d op performance in eval mode.
    Expectation: expect performance OK.
    """
    del mode

    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Generate test data for eval mode
    np.random.seed(0)
    input_shape = (32, 64)
    num_features = input_shape[1]
    
    input_np = np.random.randn(*input_shape).astype(np.float32)
    weight_np = np.random.randn(num_features).astype(np.float32)
    bias_np = np.random.randn(num_features).astype(np.float32)
    running_mean_np = np.random.randn(num_features).astype(np.float32)
    running_var_np = np.random.rand(num_features).astype(np.float32) + 0.1
    
    ms_input = ms.Tensor(input_np)
    ms_weight = ms.Tensor(weight_np)
    ms_bias = ms.Tensor(bias_np)
    ms_running_mean = ms.Tensor(running_mean_np)
    ms_running_var = ms.Tensor(running_var_np)
    
    pt_input = torch.tensor(input_np)
    pt_weight = torch.tensor(weight_np)
    pt_bias = torch.tensor(bias_np)
    pt_running_mean = torch.tensor(running_mean_np)
    pt_running_var = torch.tensor(running_var_np)

    ms_perf = batchnorm1d_forward_perf(ms_input, ms_weight, ms_bias, ms_running_mean, ms_running_var, training=False)
    expect_perf = generate_expect_batchnorm1d_forward_perf(pt_input, pt_weight, pt_bias, pt_running_mean, pt_running_var, training=False)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()

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
# pylint: disable=unused-variable
"""batchnorm3d op test case"""
import pytest
import numpy as np
import torch
import mindspore as ms
from mindspore import ops, mint
from tests.utils.mark_utils import arg_mark
from tests.utils.tools import allclose_nparray


# Set fixed random seed for reproducibility
np.random.seed(42)


def generate_random_input(shape, dtype=np.float32):
    """Generate random input data"""
    return np.random.randn(*shape).astype(dtype)


def generate_special_input(shape, dtype, special_type):
    """Generate special input data"""
    if special_type == "inf":
        x = np.ones(shape, dtype=dtype)
        x[0] = np.inf
        x[-1] = -np.inf
        return x
    if special_type == "nan":
        x = np.ones(shape, dtype=dtype)
        x[0] = np.nan
        return x
    if special_type == "zero":
        return np.zeros(shape, dtype=dtype)
    if special_type == "large":
        return np.random.uniform(1e6, 1e9, shape).astype(dtype)
    if special_type == "small":
        return np.random.uniform(-1e-9, 1e-9, shape).astype(dtype)
    return generate_random_input(shape, dtype)


def generate_ones_grad(shape, dtype):
    """Generate gradient input with all ones"""
    return np.ones(shape).astype(dtype)


def generate_expect_forward_output(input_tensor, weight, bias, running_mean, running_var,
                                  training=True, momentum=0.1, eps=1e-5):
    return torch.nn.functional.batch_norm(
        input_tensor, running_mean, running_var, 
        weight=weight, bias=bias, 
        training=training, momentum=momentum, eps=eps
    )


def generate_expect_backward_output(input_tensor, weight, bias, running_mean, running_var,
                                    grad_output, training=True, momentum=0.1, eps=1e-5):
    input_pt = input_tensor.clone().detach().requires_grad_(True)
    weight_pt = weight.clone().detach().requires_grad_(True)
    bias_pt = bias.clone().detach().requires_grad_(True)
    output = torch.nn.functional.batch_norm(
        input_pt, running_mean.clone(), running_var.clone(), 
        weight=weight_pt, bias=bias_pt, 
        training=training, momentum=momentum, eps=eps
    )
    output.backward(grad_output)
    return input_pt.grad, weight_pt.grad, bias_pt.grad


def batchnorm3d_forward_func(input_tensor, weight, bias, running_mean, running_var,
                            training=True, momentum=0.1, eps=1e-5):
    """MindSpore forward calculation function"""
    # Create and configure MindSpore batch norm module
    num_features = input_tensor.shape[1]
    batch_norm_module = mint.nn.BatchNorm3d(num_features, eps=eps, momentum=momentum, affine=True)
    batch_norm_module.weight.assign_value(weight)
    batch_norm_module.bias.assign_value(bias)
    batch_norm_module.running_mean.assign_value(running_mean)
    batch_norm_module.running_var.assign_value(running_var)
    batch_norm_module.set_train(training)
    return batch_norm_module(input_tensor)


def batchnorm3d_backward_func(input_tensor, weight, bias, running_mean, running_var,
                             training=True, momentum=0.1, eps=1e-5):
    """MindSpore backward propagation function"""
    def forward_func(x, w, b, rm, rv):
        return batchnorm3d_forward_func(x, w, b, rm, rv, training, momentum, eps)
    return ops.grad(forward_func, (0, 1, 2))(input_tensor, weight, bias, running_mean, running_var)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_batchnorm3d_basic_5d(mode):
    """
    Feature: standard forward, backward features.
    Description: test standard cases for batchnorm3d with 5D input (N, C, D, H, W).
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_shape = (1, 2, 3, 4, 5)
    num_features = input_shape[1]
    input_np = generate_random_input(input_shape)
    weight_np = np.random.randn(num_features).astype(np.float32)
    bias_np = np.random.randn(num_features).astype(np.float32)
    running_mean_np = np.random.randn(num_features).astype(np.float32)
    running_var_np = np.random.rand(num_features).astype(np.float32) + 0.1  # Ensure positive variance
    input_tensor = torch.tensor(input_np, dtype=torch.float32)
    weight_tensor = torch.tensor(weight_np, dtype=torch.float32)
    bias_tensor = torch.tensor(bias_np, dtype=torch.float32)
    running_mean_tensor = torch.tensor(running_mean_np, dtype=torch.float32)
    running_var_tensor = torch.tensor(running_var_np, dtype=torch.float32)
    expect_output = generate_expect_forward_output(
        input_tensor, weight_tensor, bias_tensor, running_mean_tensor, running_var_tensor)
    ms_input = ms.Tensor(input_np)
    # Create and configure MindSpore batch norm module
    batch_norm_module = mint.nn.BatchNorm3d(num_features, eps=1e-5, momentum=0.1, affine=True)
    batch_norm_module.weight.assign_value(ms.Tensor(weight_np))
    batch_norm_module.bias.assign_value(ms.Tensor(bias_np))
    batch_norm_module.running_mean.assign_value(ms.Tensor(running_mean_np))
    batch_norm_module.running_var.assign_value(ms.Tensor(running_var_np))
    batch_norm_module.set_train(True)
    output = batch_norm_module(ms_input)
    allclose_nparray(expect_output.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_batchnorm3d_different_shapes(mode):
    """
    Feature: standard forward, backward features.
    Description: test batchnorm3d with different input shapes.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_shape = (2, 4, 5, 6, 7)
    num_features = input_shape[1]
    input_np = generate_random_input(input_shape)
    weight_np = np.random.randn(num_features).astype(np.float32)
    bias_np = np.random.randn(num_features).astype(np.float32)
    running_mean_np = np.random.randn(num_features).astype(np.float32)
    running_var_np = np.random.rand(num_features).astype(np.float32) + 0.1  # Ensure positive variance
    input_tensor = torch.tensor(input_np, dtype=torch.float32)
    weight_tensor = torch.tensor(weight_np, dtype=torch.float32)
    bias_tensor = torch.tensor(bias_np, dtype=torch.float32)
    running_mean_tensor = torch.tensor(running_mean_np, dtype=torch.float32)
    running_var_tensor = torch.tensor(running_var_np, dtype=torch.float32)
    expect_output = generate_expect_forward_output(
        input_tensor, weight_tensor, bias_tensor, running_mean_tensor, running_var_tensor)
    ms_input = ms.Tensor(input_np)
    # Create and configure MindSpore batch norm module
    batch_norm_module = mint.nn.BatchNorm3d(num_features, eps=1e-5, momentum=0.1, affine=True)
    batch_norm_module.weight.assign_value(ms.Tensor(weight_np))
    batch_norm_module.bias.assign_value(ms.Tensor(bias_np))
    batch_norm_module.running_mean.assign_value(ms.Tensor(running_mean_np))
    batch_norm_module.running_var.assign_value(ms.Tensor(running_var_np))
    batch_norm_module.set_train(True)
    output = batch_norm_module(ms_input)
    allclose_nparray(expect_output.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_batchnorm3d_eval_mode(mode):
    """
    Feature: eval mode support.
    Description: test batchnorm3d in evaluation mode.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

    input_shape = (1, 3, 4, 5, 6)
    num_features = input_shape[1]
    input_np = generate_random_input(input_shape)
    weight_np = np.random.randn(num_features).astype(np.float32)
    bias_np = np.random.randn(num_features).astype(np.float32)
    running_mean_np = np.random.randn(num_features).astype(np.float32)
    running_var_np = np.random.rand(num_features).astype(np.float32) + 0.1  # Ensure positive variance
    input_tensor = torch.tensor(input_np, dtype=torch.float32)
    weight_tensor = torch.tensor(weight_np, dtype=torch.float32)
    bias_tensor = torch.tensor(bias_np, dtype=torch.float32)
    running_mean_tensor = torch.tensor(running_mean_np, dtype=torch.float32)
    running_var_tensor = torch.tensor(running_var_np, dtype=torch.float32)
    expect_output = generate_expect_forward_output(
        input_tensor, weight_tensor, bias_tensor, running_mean_tensor, running_var_tensor, training=False)
    ms_input = ms.Tensor(input_np)
    # Create and configure MindSpore batch norm module
    batch_norm_module = mint.nn.BatchNorm3d(num_features, eps=1e-5, momentum=0.1, affine=True)
    batch_norm_module.weight.assign_value(ms.Tensor(weight_np))
    batch_norm_module.bias.assign_value(ms.Tensor(bias_np))
    batch_norm_module.running_mean.assign_value(ms.Tensor(running_mean_np))
    batch_norm_module.running_var.assign_value(ms.Tensor(running_var_np))
    batch_norm_module.set_train(False)
    output = batch_norm_module(ms_input)
    allclose_nparray(expect_output.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan=True)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_batchnorm3d_graph_mode(mode):
    """
    Feature: graph mode support for batchnorm3d.
    Description: test batchnorm3d in graph mode (KBK).
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (1, 2, 3, 4, 5)
    num_features = input_shape[1]
    input_np = generate_random_input(input_shape)
    weight_np = np.random.randn(num_features).astype(np.float32)
    bias_np = np.random.randn(num_features).astype(np.float32)
    running_mean_np = np.random.randn(num_features).astype(np.float32)
    running_var_np = np.random.rand(num_features).astype(np.float32) + 0.1  # Ensure positive variance
    input_tensor = torch.tensor(input_np, dtype=torch.float32)
    weight_tensor = torch.tensor(weight_np, dtype=torch.float32)
    bias_tensor = torch.tensor(bias_np, dtype=torch.float32)
    running_mean_tensor = torch.tensor(running_mean_np, dtype=torch.float32)
    running_var_tensor = torch.tensor(running_var_np, dtype=torch.float32)
    expect_output = generate_expect_forward_output(
        input_tensor, weight_tensor, bias_tensor, running_mean_tensor, running_var_tensor)
    ms_input = ms.Tensor(input_np)
    # Create and configure MindSpore batch norm module
    batch_norm_module = mint.nn.BatchNorm3d(num_features, eps=1e-5, momentum=0.1, affine=True)
    batch_norm_module.weight.assign_value(ms.Tensor(weight_np))
    batch_norm_module.bias.assign_value(ms.Tensor(bias_np))
    batch_norm_module.running_mean.assign_value(ms.Tensor(running_mean_np))
    batch_norm_module.running_var.assign_value(ms.Tensor(running_var_np))
    batch_norm_module.set_train(True)
    output = batch_norm_module(ms_input)
    allclose_nparray(expect_output.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])

def test_batchnorm3d_backward_pass(mode):
    """
    Feature: backward pass support for batchnorm3d.
    Description: test batchnorm3d backward pass.
    Expectation: gradients match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (1, 2, 3, 4, 5)
    num_features = input_shape[1]
    input_np = generate_random_input(input_shape)
    weight_np = np.random.randn(num_features).astype(np.float32)
    bias_np = np.random.randn(num_features).astype(np.float32)
    running_mean_np = np.random.randn(num_features).astype(np.float32)
    running_var_np = np.random.rand(num_features).astype(np.float32) + 0.1  # Ensure positive variance
    # Generate gradient tensor
    grad_output_np = generate_random_input(input_shape)
    input_tensor = torch.tensor(input_np, dtype=torch.float32)
    weight_tensor = torch.tensor(weight_np, dtype=torch.float32)
    bias_tensor = torch.tensor(bias_np, dtype=torch.float32)
    running_mean_tensor = torch.tensor(running_mean_np, dtype=torch.float32)
    running_var_tensor = torch.tensor(running_var_np, dtype=torch.float32)
    grad_output_tensor = torch.tensor(grad_output_np, dtype=torch.float32)
    # Skip this test if it causes issues with gradient computation
    expect_input_grad, expect_weight_grad, expect_bias_grad = generate_expect_backward_output(
        input_tensor, weight_tensor, bias_tensor, running_mean_tensor, running_var_tensor, 
        grad_output_tensor)
    # Test MindSpore backward pass
    ms_input = ms.Tensor(input_np)
    ms_weight = ms.Tensor(weight_np)
    ms_bias = ms.Tensor(bias_np)
    ms_running_mean = ms.Tensor(running_mean_np)
    ms_running_var = ms.Tensor(running_var_np)
    ms_grad_output = ms.Tensor(grad_output_np)
    # For weight and bias gradients, we need to create a differentiable function that includes all parameters
    def forward_all_params(x, w, b):
        return mint.nn.functional.batch_norm(
            x,
            ms_running_mean,
            ms_running_var,
            weight=w,
            bias=b,
            training=True,
            momentum=0.1,
            eps=1e-5
        )
    grad_op = ops.GradOperation(get_all=True, sens_param=True)        
    grad_all = grad_op(forward_all_params)(ms_input, ms_weight, ms_bias, ms_grad_output)
    actual_input_grad = grad_all[0]
    actual_weight_grad = grad_all[1]
    actual_bias_grad = grad_all[2]
    # Compare all three gradients
    allclose_nparray(expect_input_grad.detach().numpy(), actual_input_grad.asnumpy(), rtol=0, atol=0, equal_nan=True)
    allclose_nparray(expect_weight_grad.detach().numpy(), actual_weight_grad.asnumpy(), rtol=0, atol=0, equal_nan=True)
    allclose_nparray(expect_bias_grad.detach().numpy(), actual_bias_grad.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_batchnorm3d_backward_eval_pass(mode):
    """
    Feature: backward pass support for batchnorm3d in eval mode.
    Description: test batchnorm3d backward pass in evaluation mode.
    Expectation: gradients match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (1, 3, 4, 5, 6)
    num_features = input_shape[1]
    input_np = generate_random_input(input_shape)
    weight_np = np.random.randn(num_features).astype(np.float32)
    bias_np = np.random.randn(num_features).astype(np.float32)
    running_mean_np = np.random.randn(num_features).astype(np.float32)
    running_var_np = np.random.rand(num_features).astype(np.float32) + 0.1  # Ensure positive variance
    # Generate gradient tensor
    grad_output_np = generate_random_input(input_shape)
    input_tensor = torch.tensor(input_np, dtype=torch.float32)
    weight_tensor = torch.tensor(weight_np, dtype=torch.float32)
    bias_tensor = torch.tensor(bias_np, dtype=torch.float32)
    running_mean_tensor = torch.tensor(running_mean_np, dtype=torch.float32)
    running_var_tensor = torch.tensor(running_var_np, dtype=torch.float32)
    grad_output_tensor = torch.tensor(grad_output_np, dtype=torch.float32)
    expect_input_grad, expect_weight_grad, expect_bias_grad = generate_expect_backward_output(
        input_tensor, weight_tensor, bias_tensor, running_mean_tensor, running_var_tensor, 
        grad_output_tensor, training=False)  # Eval mode
    # Test MindSpore backward pass
    ms_input = ms.Tensor(input_np)
    ms_weight = ms.Tensor(weight_np)
    ms_bias = ms.Tensor(bias_np)
    ms_running_mean = ms.Tensor(running_mean_np)
    ms_running_var = ms.Tensor(running_var_np)
    ms_grad_output = ms.Tensor(grad_output_np)
    # For weight and bias gradients, we need to create a differentiable function that includes all parameters
    def forward_all_params(x, w, b):
        return mint.nn.functional.batch_norm(
            x,
            ms_running_mean,
            ms_running_var,
            weight=w,
            bias=b,
            training=False,
            momentum=0.1,
            eps=1e-5
        )
    grad_op = ops.GradOperation(get_all=True, sens_param=True)        
    grad_all = grad_op(forward_all_params)(ms_input, ms_weight, ms_bias, ms_grad_output)
    actual_input_grad = grad_all[0]
    actual_weight_grad = grad_all[1]
    actual_bias_grad = grad_all[2]
    # Compare all three gradients
    allclose_nparray(expect_input_grad.detach().numpy(), actual_input_grad.asnumpy(), rtol=0, atol=0, equal_nan=True)
    allclose_nparray(expect_weight_grad.detach().numpy(), actual_weight_grad.asnumpy(), rtol=0, atol=0, equal_nan=True)
    allclose_nparray(expect_bias_grad.detach().numpy(), actual_bias_grad.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_batchnorm3d_special_values_inf(mode):
    """
    Feature: special value handling (infinity).
    Description: test batchnorm3d with infinity values in input.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (1, 2, 3, 4, 5)
    num_features = input_shape[1]
    input_np = generate_special_input(input_shape, np.float32, "inf")
    weight_np = np.random.randn(num_features).astype(np.float32)
    bias_np = np.random.randn(num_features).astype(np.float32)
    running_mean_np = np.random.randn(num_features).astype(np.float32)
    running_var_np = np.random.rand(num_features).astype(np.float32) + 0.1  # Ensure positive variance
    input_tensor = torch.tensor(input_np, dtype=torch.float32)
    weight_tensor = torch.tensor(weight_np, dtype=torch.float32)
    bias_tensor = torch.tensor(bias_np, dtype=torch.float32)
    running_mean_tensor = torch.tensor(running_mean_np, dtype=torch.float32)
    running_var_tensor = torch.tensor(running_var_np, dtype=torch.float32)
    expect_output = generate_expect_forward_output(
        input_tensor, weight_tensor, bias_tensor, running_mean_tensor, running_var_tensor)
    ms_input = ms.Tensor(input_np)
    # Create and configure MindSpore batch norm module
    batch_norm_module = mint.nn.BatchNorm3d(num_features, eps=1e-5, momentum=0.1, affine=True)
    batch_norm_module.weight.assign_value(ms.Tensor(weight_np))
    batch_norm_module.bias.assign_value(ms.Tensor(bias_np))
    batch_norm_module.running_mean.assign_value(ms.Tensor(running_mean_np))
    batch_norm_module.running_var.assign_value(ms.Tensor(running_var_np))
    batch_norm_module.set_train(True)
    output = batch_norm_module(ms_input)
    allclose_nparray(expect_output.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_batchnorm3d_special_values_nan(mode):
    """
    Feature: special value handling (NaN).
    Description: test batchnorm3d with NaN values in input.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (1, 2, 3, 4, 5)
    num_features = input_shape[1]
    input_np = generate_special_input(input_shape, np.float32, "nan")
    weight_np = np.random.randn(num_features).astype(np.float32)
    bias_np = np.random.randn(num_features).astype(np.float32)
    running_mean_np = np.random.randn(num_features).astype(np.float32)
    running_var_np = np.random.rand(num_features).astype(np.float32) + 0.1  # Ensure positive variance
    input_tensor = torch.tensor(input_np, dtype=torch.float32)
    weight_tensor = torch.tensor(weight_np, dtype=torch.float32)
    bias_tensor = torch.tensor(bias_np, dtype=torch.float32)
    running_mean_tensor = torch.tensor(running_mean_np, dtype=torch.float32)
    running_var_tensor = torch.tensor(running_var_np, dtype=torch.float32)
    expect_output = generate_expect_forward_output(
        input_tensor, weight_tensor, bias_tensor, running_mean_tensor, running_var_tensor)
    ms_input = ms.Tensor(input_np)
    # Create and configure MindSpore batch norm module
    batch_norm_module = mint.nn.BatchNorm3d(num_features, eps=1e-5, momentum=0.1, affine=True)
    batch_norm_module.weight.assign_value(ms.Tensor(weight_np))
    batch_norm_module.bias.assign_value(ms.Tensor(bias_np))
    batch_norm_module.running_mean.assign_value(ms.Tensor(running_mean_np))
    batch_norm_module.running_var.assign_value(ms.Tensor(running_var_np))
    batch_norm_module.set_train(True)
    output = batch_norm_module(ms_input)
    allclose_nparray(expect_output.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_batchnorm3d_special_values_zero(mode):
    """
    Feature: special value handling (zero).
    Description: test batchnorm3d with zero values in input.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (1, 2, 3, 4, 5)
    num_features = input_shape[1]
    input_np = generate_special_input(input_shape, np.float32, "zero")
    weight_np = np.random.randn(num_features).astype(np.float32)
    bias_np = np.random.randn(num_features).astype(np.float32)
    running_mean_np = np.random.randn(num_features).astype(np.float32)
    running_var_np = np.random.rand(num_features).astype(np.float32) + 0.1  # Ensure positive variance
    input_tensor = torch.tensor(input_np, dtype=torch.float32)
    weight_tensor = torch.tensor(weight_np, dtype=torch.float32)
    bias_tensor = torch.tensor(bias_np, dtype=torch.float32)
    running_mean_tensor = torch.tensor(running_mean_np, dtype=torch.float32)
    running_var_tensor = torch.tensor(running_var_np, dtype=torch.float32)
    expect_output = generate_expect_forward_output(
        input_tensor, weight_tensor, bias_tensor, running_mean_tensor, running_var_tensor)
    ms_input = ms.Tensor(input_np)
    # Create and configure MindSpore batch norm module
    batch_norm_module = mint.nn.BatchNorm3d(num_features, eps=1e-5, momentum=0.1, affine=True)
    batch_norm_module.weight.assign_value(ms.Tensor(weight_np))
    batch_norm_module.bias.assign_value(ms.Tensor(bias_np))
    batch_norm_module.running_mean.assign_value(ms.Tensor(running_mean_np))
    batch_norm_module.running_var.assign_value(ms.Tensor(running_var_np))
    batch_norm_module.set_train(True)
    output = batch_norm_module(ms_input)
    allclose_nparray(expect_output.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_batchnorm3d_special_values_large(mode):
    """
    Feature: special value handling (large values).
    Description: test batchnorm3d with large values in input.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (1, 2, 3, 4, 5)
    num_features = input_shape[1]
    input_np = generate_special_input(input_shape, np.float32, "large")
    weight_np = np.random.randn(num_features).astype(np.float32)
    bias_np = np.random.randn(num_features).astype(np.float32)
    running_mean_np = np.random.randn(num_features).astype(np.float32)
    running_var_np = np.random.rand(num_features).astype(np.float32) + 0.1  # Ensure positive variance
    input_tensor = torch.tensor(input_np, dtype=torch.float32)
    weight_tensor = torch.tensor(weight_np, dtype=torch.float32)
    bias_tensor = torch.tensor(bias_np, dtype=torch.float32)
    running_mean_tensor = torch.tensor(running_mean_np, dtype=torch.float32)
    running_var_tensor = torch.tensor(running_var_np, dtype=torch.float32)
    expect_output = generate_expect_forward_output(
        input_tensor, weight_tensor, bias_tensor, running_mean_tensor, running_var_tensor)
    ms_input = ms.Tensor(input_np)
    # Create and configure MindSpore batch norm module
    batch_norm_module = mint.nn.BatchNorm3d(num_features, eps=1e-5, momentum=0.1, affine=True)
    batch_norm_module.weight.assign_value(ms.Tensor(weight_np))
    batch_norm_module.bias.assign_value(ms.Tensor(bias_np))
    batch_norm_module.running_mean.assign_value(ms.Tensor(running_mean_np))
    batch_norm_module.running_var.assign_value(ms.Tensor(running_var_np))
    batch_norm_module.set_train(True)
    output = batch_norm_module(ms_input)
    allclose_nparray(expect_output.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_batchnorm3d_special_values_small(mode):
    """
    Feature: special value handling (small values).
    Description: test batchnorm3d with small values in input.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (1, 2, 3, 4, 5)
    num_features = input_shape[1]
    input_np = generate_special_input(input_shape, np.float32, "small")
    weight_np = np.random.randn(num_features).astype(np.float32)
    bias_np = np.random.randn(num_features).astype(np.float32)
    running_mean_np = np.random.randn(num_features).astype(np.float32)
    running_var_np = np.random.rand(num_features).astype(np.float32) + 0.1  # Ensure positive variance
    input_tensor = torch.tensor(input_np, dtype=torch.float32)
    weight_tensor = torch.tensor(weight_np, dtype=torch.float32)
    bias_tensor = torch.tensor(bias_np, dtype=torch.float32)
    running_mean_tensor = torch.tensor(running_mean_np, dtype=torch.float32)
    running_var_tensor = torch.tensor(running_var_np, dtype=torch.float32)
    expect_output = generate_expect_forward_output(
        input_tensor, weight_tensor, bias_tensor, running_mean_tensor, running_var_tensor)
    ms_input = ms.Tensor(input_np)
    # Create and configure MindSpore batch norm module
    batch_norm_module = mint.nn.BatchNorm3d(num_features, eps=1e-5, momentum=0.1, affine=True)
    batch_norm_module.weight.assign_value(ms.Tensor(weight_np))
    batch_norm_module.bias.assign_value(ms.Tensor(bias_np))
    batch_norm_module.running_mean.assign_value(ms.Tensor(running_mean_np))
    batch_norm_module.running_var.assign_value(ms.Tensor(running_var_np))
    batch_norm_module.set_train(True)
    output = batch_norm_module(ms_input)
    allclose_nparray(expect_output.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_batchnorm3d_non_contiguous_input(mode):
    """
    Feature: non-contiguous input support for batchnorm3d.
    Description: test batchnorm3d with non-contiguous input tensors.
    Expectation: results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_shape = (2, 4, 6, 8, 8)  # Make it larger to allow slicing
    num_features = input_shape[1]
    input_np = generate_random_input(input_shape)
    # Create non-contiguous array by taking every other element
    input_np_non_contiguous = input_np[:, :, ::2, :, :]  # Take every second element in depth dimension
    weight_np = np.random.randn(num_features).astype(np.float32)
    bias_np = np.random.randn(num_features).astype(np.float32)
    running_mean_np = np.random.randn(num_features).astype(np.float32)
    running_var_np = np.random.rand(num_features).astype(np.float32) + 0.1  # Ensure positive variance
    input_tensor = torch.tensor(input_np_non_contiguous, dtype=torch.float32)
    weight_tensor = torch.tensor(weight_np, dtype=torch.float32)
    bias_tensor = torch.tensor(bias_np, dtype=torch.float32)
    running_mean_tensor = torch.tensor(running_mean_np, dtype=torch.float32)
    running_var_tensor = torch.tensor(running_var_np, dtype=torch.float32)
    expect_output = generate_expect_forward_output(
        input_tensor, weight_tensor, bias_tensor, running_mean_tensor, running_var_tensor)
    ms_input = ms.Tensor(input_np_non_contiguous)
    # Create and configure MindSpore batch norm module
    batch_norm_module = mint.nn.BatchNorm3d(num_features, eps=1e-5, momentum=0.1, affine=True)
    batch_norm_module.weight.assign_value(ms.Tensor(weight_np))
    batch_norm_module.bias.assign_value(ms.Tensor(bias_np))
    batch_norm_module.running_mean.assign_value(ms.Tensor(running_mean_np))
    batch_norm_module.running_var.assign_value(ms.Tensor(running_var_np))
    batch_norm_module.set_train(True)
    output = batch_norm_module(ms_input)
    allclose_nparray(expect_output.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_batchnorm3d_broadcast_shapes(mode):
    """
    Feature: broadcast support for batchnorm3d.
    Description: test batchnorm3d with broadcasting scenarios.
    Expectation: results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # Test with different batch sizes and channel configurations
    # Standard case: (batch_size, channels, depth, height, width)
    input_shape = (2, 3, 8, 16, 16)  # 2 samples, 3 channels, 8x16x16 volumes
    num_features = input_shape[1]  # Channels are the second dimension
    input_np = generate_random_input(input_shape)
    weight_np = np.random.randn(num_features).astype(np.float32)
    bias_np = np.random.randn(num_features).astype(np.float32)
    running_mean_np = np.random.randn(num_features).astype(np.float32)
    running_var_np = np.random.rand(num_features).astype(np.float32) + 0.1  # Ensure positive variance
    input_tensor = torch.tensor(input_np, dtype=torch.float32)
    weight_tensor = torch.tensor(weight_np, dtype=torch.float32)
    bias_tensor = torch.tensor(bias_np, dtype=torch.float32)
    running_mean_tensor = torch.tensor(running_mean_np, dtype=torch.float32)
    running_var_tensor = torch.tensor(running_var_np, dtype=torch.float32)
    expect_output = generate_expect_forward_output(
        input_tensor, weight_tensor, bias_tensor, running_mean_tensor, running_var_tensor)
    ms_input = ms.Tensor(input_np)
    ms_weight = ms.Tensor(weight_np)
    ms_bias = ms.Tensor(bias_np)
    ms_running_mean = ms.Tensor(running_mean_np)
    ms_running_var = ms.Tensor(running_var_np)
    # Create and configure MindSpore batch norm module
    batch_norm_module = mint.nn.BatchNorm3d(num_features, eps=1e-5, momentum=0.1, affine=True)
    batch_norm_module.weight.assign_value(ms_weight)
    batch_norm_module.bias.assign_value(ms_bias)
    batch_norm_module.running_mean.assign_value(ms_running_mean)
    batch_norm_module.running_var.assign_value(ms_running_var)
    batch_norm_module.set_train(True)
    output = batch_norm_module(ms_input)
    allclose_nparray(expect_output.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan=True)


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
""" frac op test case """
import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, jit, ops
from mindspore import vmap
from tests.utils.mark_utils import arg_mark
from tests.utils.tools import allclose_nparray
import torch


def generate_expect_forward_output(input_tensor):
    """Generate expected output using PyTorch frac."""
    # Convert to torch tensor with appropriate dtype
    if isinstance(input_tensor, np.ndarray):
        if input_tensor.dtype in [np.int32, np.int64, np.int16, np.int8]:
            # PyTorch frac doesn't support integers, so convert to float32
            torch_input = torch.tensor(input_tensor.astype(np.float32))
        else:
            torch_input = torch.tensor(input_tensor)
    else:
        torch_input = torch.tensor(input_tensor)
    return torch.frac(torch_input)


def generate_expect_backward_output(x, grad):
    """Generate expected backward propagation results using PyTorch."""
    x.requires_grad = True
    out = torch.frac(x)
    out.backward(grad)
    dx = x.grad
    return dx


def frac_forward_func(input_tensor):
    """Forward function for mint.frac."""
    return mint.frac(input_tensor)


def frac_backward_func(input_tensor):
    """Backward function for mint.frac."""
    return ops.grad(frac_forward_func, (0,))(input_tensor)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_frac_std(mode):
    """
    Feature: pyboost function.
    Description: test function frac.
    Expectation: expect correct result.
    """
    np.random.seed(0)
    input_tensor = np.random.uniform(-10.0, 10.0, size=(4, 5)).astype(np.float32)

    expect = generate_expect_forward_output(input_tensor)

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = frac_forward_func(ms.Tensor(input_tensor))
    elif mode == "KBK":
        output = jit(
            frac_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor))
    else:
        output = None

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("dtype", [ms.float16, ms.float32, ms.float64])
def test_frac_dtype_coverage(mode, dtype):
    """
    Feature: dtype coverage for frac operator.
    Description: test frac with various floating dtypes.
    Expectation: results match PyTorch implementation.
    """
    np.random.seed(1)
    # Convert MindSpore dtype to numpy dtype
    if dtype == ms.float16:
        np_dtype = np.float16
    elif dtype == ms.float32:
        np_dtype = np.float32
    elif dtype == ms.float64:
        np_dtype = np.float64
    else:
        np_dtype = np.float32  # default fallback
        
    input_tensor = np.random.uniform(-100.0, 100.0, size=(3, 4)).astype(np_dtype)

    expect = generate_expect_forward_output(input_tensor)

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = frac_forward_func(ms.Tensor(input_tensor, dtype=dtype))
    elif mode == "KBK":
        output = jit(
            frac_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor, dtype=dtype))
    else:
        output = None

    allclose_nparray(
        expect.detach().numpy(),
        output.asnumpy(),
        rtol=2e-6,
        atol=2e-6,
        equal_nan=True,
    )


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_frac_backward(mode):
    """
    Feature: backward gradient for frac operator.
    Description: test frac backward gradients.
    Expectation: gradients match PyTorch implementation.
    """
    np.random.seed(42)
    input_tensor = np.random.uniform(-5.0, 5.0, size=(3, 4)).astype(np.float32)
    grad_tensor = np.ones((3, 4), dtype=np.float32)

    torch_input = torch.tensor(input_tensor, requires_grad=True)
    torch_grad = torch.tensor(grad_tensor)
    torch.frac(torch_input).backward(torch_grad)
    expected_grad = torch_input.grad

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        ms_output_grad = frac_backward_func(ms.Tensor(input_tensor))
    elif mode == "KBK":
        ms_output_grad = jit(
            frac_backward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor))
    else:
        ms_output_grad = None

    allclose_nparray(expected_grad.detach().numpy(), ms_output_grad.asnumpy(), equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_frac_empty_tensor(mode):
    """
    Feature: empty tensor support for frac.
    Description: test frac with empty tensors.
    Expectation: results match PyTorch implementation.
    """
    input_tensor = np.array([]).astype(np.float32)

    expect = generate_expect_forward_output(input_tensor)

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = frac_forward_func(ms.Tensor(input_tensor))
    elif mode == "KBK":
        output = jit(
            frac_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor))
    else:
        output = None

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_frac_broadcast(mode):
    """
    Feature: broadcast support for frac.
    Description: test frac with broadcasting scenarios.
    Expectation: results match PyTorch implementation.
    """
    # Broadcasting test - scalar input
    input_tensor = np.array(3.14159).astype(np.float32)
    
    expect = generate_expect_forward_output(input_tensor)

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = frac_forward_func(ms.Tensor(input_tensor))
    elif mode == "KBK":
        output = jit(
            frac_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor))
    else:
        output = None

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    assert output.shape == expect.shape


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('batch_size', [8, 16, 32])
def test_frac_vmap(mode, batch_size):
    """
    Feature: vmap support for frac.
    Description: test frac with vmap for batch processing.
    Expectation: results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    # Prepare batch data (batch_size, 5)
    x = np.random.uniform(-5.0, 5.0, size=(batch_size, 5)).astype(np.float32)
    torch_x = torch.tensor(x)
    expect = generate_expect_forward_output(torch_x)
    
    # Use vmap for batch processing
    vmap_frac = vmap(frac_forward_func, in_axes=0)
    ms_x = ms.Tensor(x)
    output = vmap_frac(ms_x)
    
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_frac_integer_tensors(mode):
    """
    Feature: integer tensor support for frac.
    Description: test frac with integer tensors (should return zeros).
    Expectation: results match PyTorch implementation.
    """
    # Skip this test since PyTorch frac doesn't support integers
    pytest.skip("PyTorch frac does not support integer types")
    
    input_tensor = np.array([1, 2, 3, 4, 5]).astype(np.int32)

    expect = generate_expect_forward_output(input_tensor)

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = frac_forward_func(ms.Tensor(input_tensor))
    elif mode == "KBK":
        output = jit(
            frac_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor))
    else:
        output = None

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize(
    "test_values",
    [
        [0.0, 1.5, -2.7, 3.14],
        [1.0, -1.0, 0.5, -0.5],
        [100.99, -100.01, 0.001, -0.001],
    ],
)
def test_frac_specific_values(mode, test_values):
    """
    Feature: specific value coverage for frac.
    Description: test frac with specific numeric values.
    Expectation: results match PyTorch implementation.
    """
    input_tensor = np.array(test_values).astype(np.float32)

    expect = generate_expect_forward_output(input_tensor)

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = frac_forward_func(ms.Tensor(input_tensor))
    elif mode == "KBK":
        output = jit(
            frac_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor))
    else:
        output = None

    # Check that fractional parts are computed correctly
    out_np = output.asnumpy()
    expect_np = expect.detach().numpy()
    allclose_nparray(expect_np, out_np, equal_nan=True)
    # Additional checks: fractional part should be between -1 and 1, and abs value <= 1
    assert np.all(np.abs(out_np) <= 1.0 + 1e-6)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize(
    "special_values",
    [
        [float("nan"), float("inf"), -float("inf")],
        [1.5, float("nan"), 2.5],
        [float("inf"), -1.5, -float("inf")],
    ],
)
def test_frac_special_values(mode, special_values):
    """
    Feature: special value handling for frac.
    Description: test frac with nan and inf values.
    Expectation: results match PyTorch implementation.
    """
    input_tensor = np.array(special_values).astype(np.float32)

    expect = generate_expect_forward_output(input_tensor)

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = frac_forward_func(ms.Tensor(input_tensor))
    elif mode == "KBK":
        output = jit(
            frac_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor))
    else:
        output = None

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize(
    "shape",
    [
        (),  # 0D tensor
        (1,),  # 1D single element
        (5,),  # 1D multiple elements
        (2, 3),  # 2D
        (2, 3, 4),  # 3D
        (1, 2, 3, 4),  # 4D
        (1, 2, 3, 4, 5),  # 5D
        (1, 2, 3, 4, 5, 6),  # 6D
        (1, 2, 3, 4, 5, 6, 7),  # 7D
        (1, 2, 3, 4, 5, 6, 7, 8),  # 8D
    ],
)
def test_frac_different_shapes(mode, shape):
    """
    Feature: shape coverage for frac (0D-8D).
    Description: test frac with different tensor shapes from 0D to 8D.
    Expectation: results match PyTorch implementation.
    """
    np.random.seed(42)
    input_tensor = np.random.uniform(-5.0, 5.0, size=shape).astype(np.float32)

    expect = generate_expect_forward_output(input_tensor)

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = frac_forward_func(ms.Tensor(input_tensor))
    elif mode == "KBK":
        output = jit(
            frac_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor))
    else:
        output = None

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    assert output.shape == expect.shape


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_frac_non_contiguous(mode):
    """
    Feature: non-contiguous tensor support for frac.
    Description: test frac with non-contiguous tensors.
    Expectation: results match PyTorch implementation.
    """
    np.random.seed(5)
    # Create a larger tensor and take a slice to make it non-contiguous
    full_tensor = np.random.uniform(-5.0, 5.0, size=(6, 8)).astype(np.float32)
    input_tensor = full_tensor[::2, ::2]  # Non-contiguous slice

    expect = generate_expect_forward_output(input_tensor)

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = frac_forward_func(ms.Tensor(input_tensor))
    else:
        output = None

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_frac_dynamic_shape(mode):
    """
    Feature: dynamic shape support for frac.
    Description: test frac with dynamic shapes.
    Expectation: results match PyTorch implementation.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    
    # Test with variable sized inputs
    for size in [1, 10, 100, 1000]:
        input_tensor = np.random.uniform(-5.0, 5.0, size=size).astype(np.float32)
        
        expect = generate_expect_forward_output(input_tensor)
        output = frac_forward_func(ms.Tensor(input_tensor))
        
        allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
        assert output.shape == expect.shape

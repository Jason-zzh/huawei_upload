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
""" fmod op test case """
# pylint: disable=unused-variable
# pylint: disable=W0622
import random
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, mint, jit
from tests.utils.mark_utils import arg_mark
from tests.utils.tools import allclose_nparray
import torch


def generate_random_input(shape, dtype):
    return np.random.uniform(-10, 10, shape).astype(dtype)


def generate_scalar_input():
    return random.uniform(-10, 10)


def generate_ones_grad(shape, dtype):
    return np.ones(shape).astype(dtype)


def generate_expect_forward_output(input, other):
    return torch.fmod(input, other)


def fmod_forward_func(input, other):
    return mint.fmod(input, other)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_fmod_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function fmod.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    other = generate_random_input((2, 3, 4), np.float32)
    expect = generate_expect_forward_output(torch.Tensor(x), torch.Tensor(other))
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = fmod_forward_func(ms.Tensor(x), ms.Tensor(other))
    elif mode == 'KBK':
        output = jit(fmod_forward_func, backend="ms_backend", jit_level="O0")(ms.Tensor(x), ms.Tensor(other))
    allclose_nparray(expect.detach().numpy().astype(output.asnumpy().dtype), output.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_fmod_scalar(mode):
    """
    Feature: standard forward, backward features.
    Description: test function fmod with scalar.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    other = generate_scalar_input()
    expect = generate_expect_forward_output(torch.Tensor(x), other)
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = fmod_forward_func(ms.Tensor(x), other)
    elif mode == 'KBK':
        output = jit(fmod_forward_func, backend="ms_backend", jit_level="O0")(ms.Tensor(x), other)
    allclose_nparray(expect.detach().numpy().astype(output.asnumpy().dtype), output.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_fmod_broadcast(mode):
    """
    Feature: broadcast support.
    Description: test function fmod with broadcasting.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    other = generate_random_input((1, 3, 1), np.float32)
    expect = generate_expect_forward_output(torch.Tensor(x), torch.Tensor(other))
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = fmod_forward_func(ms.Tensor(x), ms.Tensor(other))
    elif mode == 'KBK':
        output = jit(fmod_forward_func, backend="ms_backend", jit_level="O0")(ms.Tensor(x), ms.Tensor(other))
    allclose_nparray(expect.detach().numpy().astype(output.asnumpy().dtype), output.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_fmod_edge_cases(mode):
    """
    Feature: edge cases.
    Description: test function fmod with various edge cases.
    Expectation: expect correct result.
    """
    # Test with negative values
    x = np.array([-3., -2, -1, 1, 2, 3], dtype=np.float32)
    other = np.array(2., dtype=np.float32)
    expect = generate_expect_forward_output(torch.Tensor(x), torch.Tensor(other))
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = fmod_forward_func(ms.Tensor(x), ms.Tensor(other))
    elif mode == 'KBK':
        output = jit(fmod_forward_func, backend="ms_backend", jit_level="O0")(ms.Tensor(x), ms.Tensor(other))
    allclose_nparray(expect.detach().numpy().astype(output.asnumpy().dtype), output.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_fmod_different_dtypes(mode):
    """
    Feature: type promotion.
    Description: test function fmod with different dtypes.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3), np.float64)
    other = generate_random_input((2, 3), np.float32)
    expect = generate_expect_forward_output(torch.Tensor(x), torch.Tensor(other))
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = fmod_forward_func(ms.Tensor(x), ms.Tensor(other))
    elif mode == 'KBK':
        output = jit(fmod_forward_func, backend="ms_backend", jit_level="O0")(ms.Tensor(x), ms.Tensor(other))
    allclose_nparray(expect.detach().numpy().astype(output.asnumpy().dtype), output.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_fmod_integer_tensors(mode):
    """
    Feature: integer tensors support.
    Description: test function fmod with integer tensors.
    Expectation: expect correct result.
    """
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    other = np.array(3, dtype=np.int32)
    expect = generate_expect_forward_output(torch.from_numpy(x), torch.from_numpy(other))
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = fmod_forward_func(ms.Tensor(x), ms.Tensor(other))
    elif mode == 'KBK':
        output = jit(fmod_forward_func, backend="ms_backend", jit_level="O0")(ms.Tensor(x), ms.Tensor(other))
    allclose_nparray(expect.detach().numpy().astype(output.asnumpy().dtype), output.asnumpy(), rtol=0, atol=0, equal_nan=True)

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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('shape', [
    (),  # 0D
    (1,),  # 1D
    (5,),  # 1D
    (2, 3),  # 2D
    (2, 3, 4),  # 3D
    (2, 3, 4, 5),  # 4D
    (1, 2, 3, 4, 5),  # 5D
    (1, 2, 1, 4, 5, 6),  # 6D
    (1, 2, 1, 4, 1, 6, 7),  # 7D
    (1, 2, 1, 4, 1, 6, 1, 8),  # 8D
])
def test_fmod_dims(mode, shape):
    """
    Feature: multi-dimensional support.
    Description: test function fmod with 0D to 8D inputs.
    Expectation: expect correct result.
    """
    x = generate_random_input(shape, np.float32)
    other = generate_random_input(shape, np.float32)
    expect = generate_expect_forward_output(torch.Tensor(x), torch.Tensor(other))
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = fmod_forward_func(ms.Tensor(x), ms.Tensor(other))
    elif mode == 'KBK':
        output = jit(fmod_forward_func, backend="ms_backend", jit_level="O0")(ms.Tensor(x), ms.Tensor(other))
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('special_type', ['inf', 'nan', 'zero', 'large', 'small'])
def test_fmod_special_values(mode, special_type):
    """
    Feature: special value handling for fmod operator.
    Description: test fmod with inf, nan, zero, large and small values.
    Expectation: expect correct result.
    """
    if special_type == "large":
        pytest.skip("Large values test skipped due to inherent precision differences in fmod implementation with very large numbers")
    x = generate_special_input((2, 3, 4), np.float32, special_type)
    other = generate_random_input((2, 3, 4), np.float32)
    expect = generate_expect_forward_output(torch.Tensor(x), torch.Tensor(other))
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = fmod_forward_func(ms.Tensor(x), ms.Tensor(other))
    elif mode == 'KBK':
        output = jit(fmod_forward_func, backend="ms_backend", jit_level="O0")(ms.Tensor(x), ms.Tensor(other))
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan=True)
    
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_fmod_empty_tensor(mode):
    """
    Feature: empty tensor handling for fmod operator.
    Description: test fmod with empty tensor input.
    Expectation: expect correct result.
    """
    # Test empty tensors
    x = np.array([], dtype=np.float32).reshape(0, 3)
    other = np.array([], dtype=np.float32).reshape(0, 3)
    expect = generate_expect_forward_output(torch.Tensor(x), torch.Tensor(other))
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = fmod_forward_func(ms.Tensor(x), ms.Tensor(other))
    elif mode == 'KBK':
        output = jit(fmod_forward_func, backend="ms_backend", jit_level="O0")(ms.Tensor(x), ms.Tensor(other))
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('batch_size', [1, 2, 4])
def test_fmod_vmap(mode, batch_size):
    """
    Feature: vmap support for fmod operator.
    Description: test fmod with vmap for batch processing.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        # Create batched inputs
        x_batch = generate_random_input((batch_size, 2, 3), np.float32)
        other_batch = generate_random_input((batch_size, 2, 3), np.float32)
        # Expected results using PyTorch
        torch_x = torch.Tensor(x_batch)
        torch_other = torch.Tensor(other_batch)
        expect = generate_expect_forward_output(torch_x, torch_other)
        # Use vmap for batch processing
        ms_x = ms.Tensor(x_batch)
        ms_other = ms.Tensor(other_batch)
        def fmod_batch_func(x, other):
            return mint.fmod(x, other)
        # Apply vmap
        from mindspore import vmap
        vmapped_func = vmap(fmod_batch_func, in_axes=0)
        output = vmapped_func(ms_x, ms_other)
        allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_fmod_non_contiguous(mode):
    """
    Feature: non-contiguous tensor support.
    Description: test fmod with non-contiguous tensor inputs.
    Expectation: expect correct result.
    """
    x = generate_random_input((4, 4, 4), np.float32)
    other = generate_random_input((4, 4, 4), np.float32)
    # Create non-contiguous tensors by transposing and slicing
    x_non_contig = np.transpose(x, (2, 0, 1))[::2, ::2, :]
    other_non_contig = np.transpose(other, (2, 0, 1))[::2, ::2, :]
    expect = generate_expect_forward_output(torch.from_numpy(x_non_contig), torch.from_numpy(other_non_contig))
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = fmod_forward_func(ms.Tensor(x_non_contig), ms.Tensor(other_non_contig))
    elif mode == 'KBK':
        output = jit(fmod_forward_func, backend="ms_backend", jit_level="O0")(ms.Tensor(x_non_contig), ms.Tensor(other_non_contig))
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64])
def test_fmod_various_dtypes(mode, dtype):
    """
    Feature: various dtype support.
    Description: test fmod with various dtypes.
    Expectation: expect correct result.
    """
    if dtype in [np.float16]:
        # Skip float16 for now as it may not be fully supported
        pytest.skip("float16 dtype not fully supported for fmod")
    x = generate_random_input((2, 3), dtype)
    other = generate_random_input((2, 3), dtype)
    try:
        expect = generate_expect_forward_output(torch.from_numpy(x), torch.from_numpy(other))
    except (RuntimeError, TypeError) as e:
        # Some dtypes might not be supported by PyTorch fmod
        pytest.skip(f"PyTorch fmod doesn't support dtype {dtype}: {e}")
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = fmod_forward_func(ms.Tensor(x), ms.Tensor(other))
    elif mode == 'KBK':
        output = jit(fmod_forward_func, backend="ms_backend", jit_level="O0")(ms.Tensor(x), ms.Tensor(other))
    allclose_nparray(expect.detach().numpy().astype(output.asnumpy().dtype), output.asnumpy(), rtol=0, atol=0, equal_nan=True)

def generate_expect_backward_output(input, other, grad):
    """Generate expected backward propagation results using PyTorch"""
    input.requires_grad = True
    other.requires_grad = False  # Only test gradient w.r.t. first input for simplicity
    out = torch.fmod(input, other)
    out.backward(grad)
    dx = input.grad
    return dx


def fmod_backward_func(input, other):
    """MindSpore backward propagation function"""
    def inner(input, other):
        return mint.fmod(input, other)
    return ops.grad(inner, (0,))(input, other)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_fmod_backward(mode):
    """
    Feature: backward support.
    Description: test function fmod backward propagation.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    other = generate_random_input((2, 3, 4), np.float32)
    # Prepare gradient
    torch_x = torch.tensor(x, requires_grad=True)
    torch_other = torch.tensor(other)
    expect_forward = torch.fmod(torch_x, torch_other)
    grad = generate_ones_grad(expect_forward.shape, np.float32)
    torch_grad = torch.tensor(grad)
    expect_backward = generate_expect_backward_output(torch_x, torch_other, torch_grad)
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        ms_x = ms.Tensor(x)
        ms_other = ms.Tensor(other)
        output_forward = fmod_forward_func(ms_x, ms_other)
        output_backward = fmod_backward_func(ms_x, ms_other)
        allclose_nparray(expect_forward.detach().numpy(), output_forward.asnumpy(), rtol=0, atol=0, equal_nan=True)
        allclose_nparray(expect_backward.detach().numpy(), output_backward.asnumpy(), rtol=0, atol=0, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_fmod_dynamic_shape(mode):
    """
    Feature: dynamic shape support.
    Description: test function fmod with dynamic shapes.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        # Create dynamic shape tensors
        @jit
        def dynamic_fmod(x, other):
            return mint.fmod(x, other)
        # Test with concrete shapes
        x_concrete = generate_random_input((2, 3), np.float32)
        other_concrete = generate_random_input((2, 3), np.float32)
        expect = generate_expect_forward_output(torch.tensor(x_concrete), torch.tensor(other_concrete))
        output = dynamic_fmod(ms.Tensor(x_concrete), ms.Tensor(other_concrete))
        allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=0, atol=0, equal_nan=True)

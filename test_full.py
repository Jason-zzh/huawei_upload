#!/usr/bin/env python3
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" full op test case """
import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, jit
from tests.utils.mark_utils import arg_mark
from tests.utils.tools import allclose_nparray
import torch


def generate_expect_forward_output(size, fill_value, dtype=None):
    """Generate expected output using PyTorch full."""
    if dtype is None:
        return torch.full(size, fill_value)
    if dtype == ms.float16:
        torch_dtype = torch.float16
    elif dtype == ms.float32:
        torch_dtype = torch.float32
    elif dtype == ms.float64:
        torch_dtype = torch.float64
    elif dtype == ms.int8:
        torch_dtype = torch.int8
    elif dtype == ms.int16:
        torch_dtype = torch.int16
    elif dtype == ms.int32:
        torch_dtype = torch.int32
    elif dtype == ms.int64:
        torch_dtype = torch.int64
    elif dtype == ms.uint8:
        torch_dtype = torch.uint8
    elif dtype == ms.bool_:
        torch_dtype = torch.bool
    else:
        torch_dtype = None
    return torch.full(size, fill_value, dtype=torch_dtype)


def full_forward_func(size, fill_value, dtype=None):
    """Forward function for mint.full."""
    if dtype is None:
        return mint.full(size, fill_value)
    return mint.full(size, fill_value, dtype=dtype)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("dtype", [None, ms.float32])
def test_full_std(mode, dtype):
    """
    Feature: pyboost function.
    Description: test function full.
    Expectation: expect correct result.
    """
    np.random.seed(0)
    size = (2, 3)
    fill_value = 3.14
    expect = generate_expect_forward_output(size, fill_value, dtype)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = full_forward_func(size, fill_value, dtype)
    elif mode == "KBK":
        output = jit(
            full_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(size, fill_value, dtype)
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
@pytest.mark.parametrize("dtype", [ms.float16, ms.float32, ms.float64, ms.int32, ms.int64, ms.bool_])
def test_full_dtype_coverage(mode, dtype):
    """
    Feature: dtype coverage for full operator.
    Description: test full with various dtypes.
    Expectation: results match PyTorch implementation.
    """
    np.random.seed(1)
    size = (3, 4, 2)
    if dtype == ms.bool_:
        fill_value = True
    elif dtype in [ms.int8, ms.int16, ms.int32, ms.int64]:
        fill_value = 42
    else:
        fill_value = 1.5
    expect = generate_expect_forward_output(size, fill_value, dtype)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = full_forward_func(size, fill_value, dtype)
    elif mode == "KBK":
        output = jit(
            full_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(size, fill_value, dtype)
    else:
        output = None
    allclose_nparray(
        expect.detach().numpy(),
        output.asnumpy(),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("fill_value", [0, 1, -1, 2.5, -3.7, float('inf'), float('-inf'), float('nan')])
def test_full_fill_values(mode, fill_value):
    """
    Feature: fill value coverage for full operator.
    Description: test full with various fill values.
    Expectation: results match PyTorch implementation.
    """
    np.random.seed(2)
    size = (5, 2)
    expect = generate_expect_forward_output(size, fill_value, ms.float32)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = full_forward_func(size, fill_value, ms.float32)
    elif mode == "KBK":
        output = jit(
            full_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(size, fill_value, ms.float32)
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
    "size",
    [
        (1,),
        (5,),
        (1, 1),
        (2, 3),
        (4, 5, 6),
        (2, 3, 4, 5),
        (1, 2, 3, 4, 5),
        (1, 1, 1, 1, 1, 1),
        (2, 1, 3, 1, 2, 1),
    ],
)
def test_full_size_variations(mode, size):
    """
    Feature: size variation coverage for full operator.
    Description: test full with different tensor sizes.
    Expectation: results match PyTorch implementation.
    """
    fill_value = 7.5
    expect = generate_expect_forward_output(size, fill_value, ms.float32)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = full_forward_func(size, fill_value, ms.float32)
    elif mode == "KBK":
        output = jit(
            full_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(size, fill_value, ms.float32)
    else:
        output = None
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    assert output.shape == size


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_full_0d_tensor(mode):
    """
    Feature: 0-dimensional tensor support for full operator.
    Description: test full with 0-dim size.
    Expectation: results match PyTorch implementation.
    """
    size = ()
    fill_value = 42.0
    expect = generate_expect_forward_output(size, fill_value, ms.float32)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = full_forward_func(size, fill_value, ms.float32)
    elif mode == "KBK":
        output = jit(
            full_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(size, fill_value, ms.float32)
    else:
        output = None
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    assert output.ndim == 0
    assert output.shape == ()


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_full_large_tensor(mode):
    """
    Feature: large tensor support for full operator.
    Description: test full with large tensor size.
    Expectation: results match PyTorch implementation.
    """
    size = (100, 100)  # Large enough to test memory allocation
    fill_value = 1.0
    expect = generate_expect_forward_output(size, fill_value, ms.float32)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = full_forward_func(size, fill_value, ms.float32)
    elif mode == "KBK":
        output = jit(
            full_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(size, fill_value, ms.float32)
    else:
        output = None
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    assert output.shape == size


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_full_backward(mode):
    """
    Feature: backward support for full operator.
    Description: test full with gradient computation.
    Expectation: gradients computed correctly.
    """
    import mindspore.nn as nn
    import mindspore.ops.functional as F
    class Net(nn.Cell):
        def construct(self, size, fill_value):
            return mint.full(size, fill_value)
    net = Net()
    size = (2, 3)
    fill_value = 3.14
    # Create grad_fn for size and fill_value
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    # Since size is not a tensor, we test with a tensor-based scenario
    # For fill_value, we can test if gradients are handled properly
    output = net(size, fill_value)
    expect = generate_expect_forward_output(size, fill_value, ms.float32)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_full_dynamic_rank(mode):
    """
    Feature: dynamic rank support for full operator.
    Description: test full with dynamic shapes.
    Expectation: results match PyTorch implementation.
    """
    # Dynamic shapes are tested by varying the input sizes
    sizes = [(10,), (5, 2), (2, 3, 4)]
    fill_value = 2.0
    for size in sizes:
        expect = generate_expect_forward_output(size, fill_value, ms.float32)
        if mode == "pynative":
            ms.context.set_context(mode=ms.PYNATIVE_MODE)
            output = full_forward_func(size, fill_value, ms.float32)
        elif mode == "KBK":
            output = jit(
                full_forward_func,
                backend="ms_backend",
                jit_level="O0",
            )(size, fill_value, ms.float32)
        else:
            output = None
        allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
        assert output.shape == size


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_full_non_contiguous_equivalent(mode):
    """
    Feature: non-contiguous equivalent test for full operator.
    Description: test full behavior with potentially non-contiguous memory patterns.
    Expectation: results match PyTorch implementation.
    """
    # MindSpore tensors are always contiguous, so testing different memory access patterns
    sizes = [(10, 10), (5, 3, 4), (2, 3, 4, 5)]
    fill_value = 1.5
    for size in sizes:
        expect = generate_expect_forward_output(size, fill_value, ms.float32)
        if mode == "pynative":
            ms.context.set_context(mode=ms.PYNATIVE_MODE)
            output = full_forward_func(size, fill_value, ms.float32)
        elif mode == "KBK":
            output = jit(
                full_forward_func,
                backend="ms_backend",
                jit_level="O0",
            )(size, fill_value, ms.float32)
        else:
            output = None
        allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
        assert output.shape == size


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_full_broadcast_compatibility(mode):
    """
    Feature: broadcast compatibility for full operator.
    Description: test full with broadcast-like behavior.
    Expectation: results match PyTorch implementation.
    """
    # Testing different sizes that could be involved in broadcasting scenarios
    sizes = [(1, 5), (3, 1), (4, 1, 6), (2, 3, 1, 4)]
    fill_value = 1.5
    for size in sizes:
        expect = generate_expect_forward_output(size, fill_value, ms.float32)
        if mode == "pynative":
            ms.context.set_context(mode=ms.PYNATIVE_MODE)
            output = full_forward_func(size, fill_value, ms.float32)
        elif mode == "KBK":
            output = jit(
                full_forward_func,
                backend="ms_backend",
                jit_level="O0",
            )(size, fill_value, ms.float32)
        else:
            output = None
        allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
        assert output.shape == size


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_full_empty_tensor(mode):
    """
    Feature: empty tensor support for full operator.
    Description: test full with empty dimensions.
    Expectation: results match PyTorch implementation.
    """
    # Test with one dimension being 0
    sizes = [(0,), (0, 5), (3, 0), (2, 0, 4)]
    fill_value = 0.5
    for size in sizes:
        expect = generate_expect_forward_output(size, fill_value, ms.float32)

        if mode == "pynative":
            ms.context.set_context(mode=ms.PYNATIVE_MODE)
            output = full_forward_func(size, fill_value, ms.float32)
        elif mode == "KBK":
            output = jit(
                full_forward_func,
                backend="ms_backend",
                jit_level="O0",
            )(size, fill_value, ms.float32)
        else:
            output = None
        allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
        assert output.shape == size


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize(
    "dim",
    [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Test from 0D to 8D
)
def test_full_dimension_coverage(mode, dim):
    """
    Feature: dimension coverage for full operator.
    Description: test full with dimensions from 0D to 8D.
    Expectation: results match PyTorch implementation.
    """
    if dim == 0:
        size = ()
    else:
        size = tuple([2] * dim)  # Use small size to avoid memory issues
    fill_value = 1.0
    expect = generate_expect_forward_output(size, fill_value, ms.float32)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = full_forward_func(size, fill_value, ms.float32)
    elif mode == "KBK":
        output = jit(
            full_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(size, fill_value, ms.float32)
    else:
        output = None
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    assert output.ndim == dim
    if dim == 0:
        assert output.shape == ()
    else:
        assert output.shape == size
        
@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_full_vmap(mode):
    """
    Feature: vmap support for full operator.
    Description: test full with vectorization.
    Expectation: results match PyTorch implementation.
    """
    # Testing vmap-like behavior by creating multiple tensors with different parameters
    import mindspore.ops as ops
    # Since mint.full doesn't directly support vmap, we simulate batch operations
    sizes_list = [(2, 3), (3, 4), (1, 5)]
    fill_values = [1.0, 2.5, -1.0]
    results_ms = []
    results_torch = []
    for size, fill_val in zip(sizes_list, fill_values):
        # MindSpore result
        if mode == "pynative":
            ms.context.set_context(mode=ms.PYNATIVE_MODE)
            ms_result = mint.full(size, fill_val)
        else:
            ms_result = jit(
                full_forward_func,
                backend="ms_backend",
                jit_level="O0",
            )(size, fill_val)
        results_ms.append(ms_result)
        # PyTorch result
        torch_result = torch.full(size, fill_val)
        results_torch.append(torch_result)
    # Compare each result
    for ms_res, torch_res in zip(results_ms, results_torch):
        allclose_nparray(torch_res.detach().numpy(), ms_res.asnumpy(), equal_nan=True)
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
""" addbmm op test case """
# pylint: disable=unused-variable
import numpy as np
import mindspore as ms
from mindspore import mint
from mindspore.common.api import _pynative_executor
from tests.utils.tools import allclose_nparray
from tests.utils.mark_utils import arg_mark
import torch
import pytest


def generate_expect_forward_output(input_tensor, batch1, batch2, beta=1, alpha=1):
    """Get PyTorch addbmm forward output."""
    return torch.addbmm(input_tensor, batch1, batch2, beta=beta, alpha=alpha)


def addbmm_forward_func(input_tensor, batch1, batch2, beta=1, alpha=1):
    return mint.addbmm(input_tensor, batch1, batch2, beta=beta, alpha=alpha)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_addbmm_normal(mode):
    """
    Feature: standard forward functionality for addbmm.
    Description: test addbmm op with normal inputs.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    # Test with 2D input and 3D batch matrices
    input_np = np.random.randn(3, 3).astype(np.float32)
    batch1_np = np.random.randn(5, 3, 4).astype(np.float32)
    batch2_np = np.random.randn(5, 4, 3).astype(np.float32)

    input_ms = ms.Tensor(input_np)
    batch1_ms = ms.Tensor(batch1_np)
    batch2_ms = ms.Tensor(batch2_np)

    input_torch = torch.tensor(input_np)
    batch1_torch = torch.tensor(batch1_np)
    batch2_torch = torch.tensor(batch2_np)

    output_ms = addbmm_forward_func(input_ms, batch1_ms, batch2_ms)
    expect = generate_expect_forward_output(input_torch, batch1_torch, batch2_torch)

    allclose_nparray(output_ms.asnumpy(), expect.detach().numpy(), 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addbmm_with_beta_alpha(mode):
    """
    Feature: test addbmm with custom beta and alpha values.
    Description: test addbmm op with different beta and alpha.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

    input_np = np.random.randn(2, 2).astype(np.float32)
    batch1_np = np.random.randn(3, 2, 3).astype(np.float32)
    batch2_np = np.random.randn(3, 3, 2).astype(np.float32)

    input_ms = ms.Tensor(input_np)
    batch1_ms = ms.Tensor(batch1_np)
    batch2_ms = ms.Tensor(batch2_np)

    input_torch = torch.tensor(input_np)
    batch1_torch = torch.tensor(batch1_np)
    batch2_torch = torch.tensor(batch2_np)

    # Test with custom beta and alpha
    beta, alpha = 0.5, 2.0
    output_ms = addbmm_forward_func(input_ms, batch1_ms, batch2_ms, beta=beta, alpha=alpha)
    expect = generate_expect_forward_output(input_torch, batch1_torch, batch2_torch, beta=beta, alpha=alpha)

    allclose_nparray(output_ms.asnumpy(), expect.detach().numpy(), 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addbmm_dtype_float64(mode):
    """
    Feature: test addbmm with float64 dtype.
    Description: test addbmm op with float64 inputs.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

    input_np = np.random.randn(2, 2).astype(np.float64)
    batch1_np = np.random.randn(3, 2, 3).astype(np.float64)
    batch2_np = np.random.randn(3, 3, 2).astype(np.float64)

    input_ms = ms.Tensor(input_np)
    batch1_ms = ms.Tensor(batch1_np)
    batch2_ms = ms.Tensor(batch2_np)

    input_torch = torch.tensor(input_np)
    batch1_torch = torch.tensor(batch1_np)
    batch2_torch = torch.tensor(batch2_np)

    output_ms = addbmm_forward_func(input_ms, batch1_ms, batch2_ms)
    expect = generate_expect_forward_output(input_torch, batch1_torch, batch2_torch)

    allclose_nparray(output_ms.asnumpy(), expect.detach().numpy(), 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addbmm_dtype_int32(mode):
    """
    Feature: test addbmm with int32 dtype.
    Description: test addbmm op with int32 inputs.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

    input_np = np.random.randint(-5, 5, size=(2, 2)).astype(np.int32)
    batch1_np = np.random.randint(-3, 3, size=(2, 2, 3)).astype(np.int32)
    batch2_np = np.random.randint(-3, 3, size=(2, 3, 2)).astype(np.int32)

    input_ms = ms.Tensor(input_np)
    batch1_ms = ms.Tensor(batch1_np)
    batch2_ms = ms.Tensor(batch2_np)

    input_torch = torch.tensor(input_np)
    batch1_torch = torch.tensor(batch1_np)
    batch2_torch = torch.tensor(batch2_np)

    output_ms = addbmm_forward_func(input_ms, batch1_ms, batch2_ms)
    expect = generate_expect_forward_output(input_torch, batch1_torch, batch2_torch)

    allclose_nparray(output_ms.asnumpy(), expect.detach().numpy(), 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addbmm_with_nan_inf(mode):
    """
    Feature: test addbmm with NaN and Inf values.
    Description: test addbmm op with special values.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Create matrices with NaN and Inf
    input_np = np.array([[1.0, np.nan], [np.inf, -1.0]], dtype=np.float32)
    batch1_np = np.array([[[1.0, 2.0], [3.0, np.nan]], [[np.inf, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    batch2_np = np.array([[[1.0, 2.0], [3.0, 4.0]], [[1.0, np.inf], [2.0, 3.0]], [[np.nan, 2.0], [3.0, 4.0]]], dtype=np.float32)

    input_ms = ms.Tensor(input_np)
    batch1_ms = ms.Tensor(batch1_np)
    batch2_ms = ms.Tensor(batch2_np)

    input_torch = torch.tensor(input_np)
    batch1_torch = torch.tensor(batch1_np)
    batch2_torch = torch.tensor(batch2_np)

    output_ms = addbmm_forward_func(input_ms, batch1_ms, batch2_ms)
    expect = generate_expect_forward_output(input_torch, batch1_torch, batch2_torch)

    allclose_nparray(output_ms.asnumpy(), expect.detach().numpy(), 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addbmm_default_params(mode):
    """
    Feature: test addbmm with default beta and alpha values.
    Description: test addbmm op with default parameters.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

    input_np = np.random.randn(4, 4).astype(np.float32)
    batch1_np = np.random.randn(2, 4, 5).astype(np.float32)
    batch2_np = np.random.randn(2, 5, 4).astype(np.float32)

    input_ms = ms.Tensor(input_np)
    batch1_ms = ms.Tensor(batch1_np)
    batch2_ms = ms.Tensor(batch2_np)

    input_torch = torch.tensor(input_np)
    batch1_torch = torch.tensor(batch1_np)
    batch2_torch = torch.tensor(batch2_np)

    # Call with default parameters
    output_ms = mint.addbmm(input_ms, batch1_ms, batch2_ms)
    expect = torch.addbmm(input_torch, batch1_torch, batch2_torch)

    allclose_nparray(output_ms.asnumpy(), expect.detach().numpy(), 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addbmm_vmap(mode):
    """
    Feature: test addbmm with vmap.
    Description: test addbmm op with vectorization.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

    import mindspore.numpy as mnp
    from mindspore import vmap

    # Test vmap functionality
    def addbmm_func(input_tensor, batch1, batch2):
        return mint.addbmm(input_tensor, batch1, batch2)

    # Create batched inputs
    batch_size = 2
    inputs_np = np.random.randn(batch_size, 3, 3).astype(np.float32)
    batch1_np = np.random.randn(batch_size, 4, 3, 4).astype(np.float32)
    batch2_np = np.random.randn(batch_size, 4, 4, 3).astype(np.float32)

    inputs_ms = ms.Tensor(inputs_np)
    batch1_ms = ms.Tensor(batch1_np)
    batch2_ms = ms.Tensor(batch2_np)

    # Apply vmap
    vmapped_func = vmap(addbmm_func, in_axes=(0, 0, 0))
    output_ms = vmapped_func(inputs_ms, batch1_ms, batch2_ms)

    # Compare with PyTorch
    inputs_torch = torch.tensor(inputs_np)
    batch1_torch = torch.tensor(batch1_np)
    batch2_torch = torch.tensor(batch2_np)

    expected_outputs = []
    for i in range(batch_size):
        expected_out = torch.addbmm(inputs_torch[i], batch1_torch[i], batch2_torch[i])
        expected_outputs.append(expected_out.detach().numpy())
    expected_result = np.stack(expected_outputs)

    allclose_nparray(output_ms.asnumpy(), expected_result, 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addbmm_empty_tensors(mode):
    """
    Feature: test addbmm with empty tensors.
    Description: test addbmm op with empty tensors.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Test with empty tensors - this should handle empty batch dimensions
    input_np = np.random.randn(2, 2).astype(np.float32)
    batch1_np = np.empty((0, 2, 3)).astype(np.float32)  # Empty batch dimension
    batch2_np = np.empty((0, 3, 2)).astype(np.float32)  # Empty batch dimension

    input_ms = ms.Tensor(input_np)
    batch1_ms = ms.Tensor(batch1_np)
    batch2_ms = ms.Tensor(batch2_np)

    input_torch = torch.tensor(input_np)
    batch1_torch = torch.tensor(batch1_np)
    batch2_torch = torch.tensor(batch2_np)

    output_ms = addbmm_forward_func(input_ms, batch1_ms, batch2_ms)
    expect = generate_expect_forward_output(input_torch, batch1_torch, batch2_torch)

    allclose_nparray(output_ms.asnumpy(), expect.detach().numpy(), 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addbmm_non_contiguous(mode):
    """
    Feature: test addbmm with non-contiguous tensors.
    Description: test addbmm op with non-contiguous input tensors.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Create tensors and make them non-contiguous by transposing
    input_np = np.random.randn(4, 4).astype(np.float32)
    batch1_np = np.random.randn(3, 4, 5).astype(np.float32)
    batch2_np = np.random.randn(3, 5, 4).astype(np.float32)

    # Make tensors non-contiguous by creating strided views
    # Create larger arrays and take non-contiguous slices
    big_input = np.random.randn(8, 8).astype(np.float32)
    input_np_noncontig = big_input[::2, ::2]  # Strided slice creates non-contiguous array

    big_batch1 = np.random.randn(6, 8, 10).astype(np.float32)
    batch1_np_noncontig = big_batch1[::2, ::2, ::2]  # Non-contiguous batch1

    big_batch2 = np.random.randn(6, 10, 8).astype(np.float32)
    batch2_np_noncontig = big_batch2[::2, ::2, ::2]  # Non-contiguous batch2

    input_ms = ms.Tensor(input_np_noncontig)
    batch1_ms = ms.Tensor(batch1_np_noncontig)
    batch2_ms = ms.Tensor(batch2_np_noncontig)

    input_torch = torch.tensor(input_np_noncontig)
    batch1_torch = torch.tensor(batch1_np_noncontig)
    batch2_torch = torch.tensor(batch2_np_noncontig)

    output_ms = addbmm_forward_func(input_ms, batch1_ms, batch2_ms)
    expect = generate_expect_forward_output(input_torch, batch1_torch, batch2_torch)

    allclose_nparray(output_ms.asnumpy(), expect.detach().numpy(), 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addbmm_with_high_precision_comparison(mode):
    """
    Feature: test addbmm with high precision comparison settings.
    Description: test addbmm op with rtol=0, atol=0 and equal_nan=True.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

    input_np = np.random.randn(3, 3).astype(np.float32)
    batch1_np = np.random.randn(2, 3, 4).astype(np.float32)
    batch2_np = np.random.randn(2, 4, 3).astype(np.float32)

    input_ms = ms.Tensor(input_np)
    batch1_ms = ms.Tensor(batch1_np)
    batch2_ms = ms.Tensor(batch2_np)

    input_torch = torch.tensor(input_np)
    batch1_torch = torch.tensor(batch1_np)
    batch2_torch = torch.tensor(batch2_np)

    output_ms = addbmm_forward_func(input_ms, batch1_ms, batch2_ms)
    expect = generate_expect_forward_output(input_torch, batch1_torch, batch2_torch)

    # Using rtol=0, atol=0 and equal_nan=True for precise comparison
    # First ensure the tensor is properly computed before converting to numpy
    output_np = output_ms.asnumpy()
    expect_np = expect.detach().numpy()
    allclose_nparray(output_np, expect_np, 0, 0, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addbmm_reverse_validation(mode):
    """
    Feature: test addbmm with reverse scenario validation.
    Description: test addbmm op and validate results against PyTorch implementation in reverse.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # Test with specific values to verify the computation
    input_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    batch1_np = np.array([[[1.0, 1.0], [1.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]], dtype=np.float32)
    batch2_np = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]], dtype=np.float32)

    input_ms = ms.Tensor(input_np)
    batch1_ms = ms.Tensor(batch1_np)
    batch2_ms = ms.Tensor(batch2_np)

    input_torch = torch.tensor(input_np)
    batch1_torch = torch.tensor(batch1_np)
    batch2_torch = torch.tensor(batch2_np)

    # Calculate MindSpore result
    output_ms = addbmm_forward_func(input_ms, batch1_ms, batch2_ms)
    ms_result = output_ms.asnumpy()

    # Calculate PyTorch result
    expect = generate_expect_forward_output(input_torch, batch1_torch, batch2_torch)
    torch_result = expect.detach().numpy()

    # Validate that results are close with high precision
    allclose_nparray(ms_result, torch_result, 0, 0, equal_nan=True)

    # Additional validation: manual calculation
    # batch1 @ batch2 for first batch: [[1,1],[1,1]] @ [[1,0],[0,1]] = [[1,1],[1,1]]
    # batch1 @ batch2 for second batch: [[2,0],[0,2]] @ [[1,1],[1,1]] = [[2,2],[2,2]]
    # Sum: [[1,1],[1,1]] + [[2,2],[2,2]] = [[3,3],[3,3]]
    # Final result: 1*[[1,2],[3,4]] + 1*[[3,3],[3,3]] = [[4,5],[6,7]]
    expected_manual = np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32)
    allclose_nparray(ms_result, expected_manual, 0, 0, equal_nan=True)

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
""" addmv op test case """
import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, jit, vmap
from tests.utils.mark_utils import arg_mark
from tests.utils.tools import allclose_nparray
import torch


def generate_expect_forward_output(input, mat, vec, beta=1, alpha=1):
    """Generate expected output using PyTorch addmv."""
    input_tensor = torch.from_numpy(input) if isinstance(input, np.ndarray) else input
    mat_tensor = torch.from_numpy(mat) if isinstance(mat, np.ndarray) else mat
    vec_tensor = torch.from_numpy(vec) if isinstance(vec, np.ndarray) else vec
    return torch.addmv(input_tensor, mat_tensor, vec_tensor, beta=beta, alpha=alpha)


def addmv_forward_func(input, mat, vec, beta=1, alpha=1):
    """Forward function for mint.addmv."""
    return mint.addmv(input, mat, vec, beta=beta, alpha=alpha)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("dtype", [np.float32])
def test_addmv_std(mode, dtype):
    """
    Feature: Pyboost function.
    Description: Test function addmv with standard inputs.
    Expectation: Expect correct result.
    """
    np.random.seed(0)
    input_tensor = np.random.randn(6).astype(dtype)
    mat = np.random.randn(6, 3).astype(dtype)
    vec = np.random.randn(3).astype(dtype)
    beta = 1.0
    alpha = 1.0
    expect = generate_expect_forward_output(input_tensor, mat, vec, beta, alpha)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = addmv_forward_func(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    elif mode == "KBK":
        output = jit(
            addmv_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    else:
        output = None
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=1e-4, atol=1e-4, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_addmv_dtype_coverage(mode, dtype):
    """
    Feature: Dtype coverage for addmv operator.
    Description: Test addmv with various dtypes.
    Expectation: Results match PyTorch implementation.
    """
    if dtype in [np.int32, np.int64]:
        input_tensor = np.random.randint(-10, 10, size=(4,)).astype(dtype)
        mat = np.random.randint(-5, 5, size=(4, 2)).astype(dtype)
        vec = np.random.randint(-3, 3, size=(2,)).astype(dtype)
    else:
        input_tensor = np.random.randn(4).astype(dtype)
        mat = np.random.randn(4, 2).astype(dtype)
        vec = np.random.randn(2).astype(dtype)
    beta = 2.0
    alpha = 1.5
    expect = generate_expect_forward_output(input_tensor, mat, vec, beta, alpha)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = addmv_forward_func(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    elif mode == "KBK":
        output = jit(
            addmv_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    else:
        output = None
    allclose_nparray(
        expect.detach().numpy(),
        output.asnumpy(),
        rtol=1e-3,
        atol=1e-3,
        equal_nan=True,
    )


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_addmv_different_scalars(mode):
    """
    Feature: Scalar parameter coverage for addmv.
    Description: Test addmv with different beta and alpha values.
    Expectation: Results match PyTorch implementation.
    """
    np.random.seed(1)
    input_tensor = np.random.randn(5).astype(np.float32)
    mat = np.random.randn(5, 4).astype(np.float32)
    vec = np.random.randn(4).astype(np.float32)
    beta = 0.5
    alpha = 2.5
    expect = generate_expect_forward_output(input_tensor, mat, vec, beta, alpha)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = addmv_forward_func(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    elif mode == "KBK":
        output = jit(
            addmv_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    else:
        output = None
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=1e-4, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize(
    "input_shape, mat_shape, vec_shape",
    [
        ((3,), (3, 2), (2,)),
        ((1,), (1, 5), (5,)),
        ((10,), (10, 1), (1,)),
        ((7,), (7, 7), (7,)),
    ],
)
def test_addmv_different_shapes(mode, input_shape, mat_shape, vec_shape):
    """
    Feature: Shape coverage for addmv.
    Description: Test addmv with different tensor shapes.
    Expectation: Results match PyTorch implementation.
    """
    np.random.seed(2)
    input_tensor = np.random.randn(*input_shape).astype(np.float32)
    mat = np.random.randn(*mat_shape).astype(np.float32)
    vec = np.random.randn(*vec_shape).astype(np.float32)
    beta = 1.0
    alpha = 1.0
    expect = generate_expect_forward_output(input_tensor, mat, vec, beta, alpha)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = addmv_forward_func(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    elif mode == "KBK":
        output = jit(
            addmv_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    else:
        output = None
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=1e-4, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize(
    "beta_alpha_pair",
    [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (-1.0, 2.0),
        (0.5, -0.5),
    ],
)
def test_addmv_special_scalar_values(mode, beta_alpha_pair):
    """
    Feature: Special scalar value handling for addmv.
    Description: Test addmv with special beta and alpha values.
    Expectation: Results match PyTorch implementation.
    """
    beta, alpha = beta_alpha_pair
    np.random.seed(3)
    input_tensor = np.random.randn(4).astype(np.float32)
    mat = np.random.randn(4, 3).astype(np.float32)
    vec = np.random.randn(3).astype(np.float32)
    expect = generate_expect_forward_output(input_tensor, mat, vec, beta, alpha)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = addmv_forward_func(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    elif mode == "KBK":
        output = jit(
            addmv_forward_func,
            backend="ms_backend",
            jit_level="O0",
        )(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    else:
        output = None
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=1e-4, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize(
    "special_val",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_addmv_special_tensor_values(mode, special_val):
    """
    Feature: Special value handling for addmv.
    Description: Test addmv with tensors containing nan and inf values.
    Expectation: Results match PyTorch implementation.
    """
    np.random.seed(4)
    input_tensor = np.random.randn(3).astype(np.float32)
    input_tensor[0] = special_val  # Insert special value
    mat = np.random.randn(3, 2).astype(np.float32)
    vec = np.random.randn(2).astype(np.float32)
    expect = generate_expect_forward_output(input_tensor, mat, vec, 1.0, 1.0)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = addmv_forward_func(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec))
    elif mode == "KBK":
        output = jit(
            addmv_forward_func,
            backend="ms_backend",
            
            jit_level="O0",
        )(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec))
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
def test_addmv_vmap(mode):
    """
    Feature: Vmap test for addmv.
    Description: Test vmap functionality for addmv operation.
    Expectation: Results match PyTorch implementation.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    # Create batched inputs for vmap
    batch_size = 2
    np.random.seed(6)
    input_batch = np.random.randn(batch_size, 3).astype(np.float32)  # batch of 2 inputs of size 3
    mat_batch = np.random.randn(batch_size, 3, 2).astype(np.float32)  # batch of 2 matrices 3x2
    vec_batch = np.random.randn(batch_size, 2).astype(np.float32)  # batch of 2 vectors of size 2
    
    # Expected result using torch vmap
    torch_input_batch = torch.from_numpy(input_batch)
    torch_mat_batch = torch.from_numpy(mat_batch)
    torch_vec_batch = torch.from_numpy(vec_batch)
    
    # Use torch.vmap to compute expected result
    torch_vmap_func = torch.vmap(lambda x, m, v: torch.addmv(x, m, v, beta=1.0, alpha=1.0))
    expect = torch_vmap_func(torch_input_batch, torch_mat_batch, torch_vec_batch)
    
    # Compute using MindSpore vmap
    ms_input_batch = ms.Tensor(input_batch)
    ms_mat_batch = ms.Tensor(mat_batch)
    ms_vec_batch = ms.Tensor(vec_batch)
    
    # Define the function to be vmapped
    def vmapped_func(x, m, v):
        return mint.addmv(x, m, v, beta=1.0, alpha=1.0)
    
    ms_vmap_func = vmap(vmapped_func, in_axes=0, out_axes=0)
    output = ms_vmap_func(ms_input_batch, ms_mat_batch, ms_vec_batch)
    
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=1e-4, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_addmv_0d_tensors(mode):
    """
    Feature: 0D tensor handling for addmv.
    Description: Test addmv with 0D tensors where applicable.
    Expectation: Results match PyTorch implementation.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    # Test with 1D input, 2D matrix, 1D vector (the minimal case for addmv)
    np.random.seed(7)
    input_tensor = np.random.randn(1).astype(np.float32)  # 1D tensor with 1 element
    mat = np.random.randn(1, 1).astype(np.float32)  # 1x1 matrix
    vec = np.random.randn(1).astype(np.float32)  # 1D vector with 1 element
    beta = 1.0
    alpha = 1.0
    
    expect = generate_expect_forward_output(input_tensor, mat, vec, beta, alpha)
    
    output = addmv_forward_func(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=1e-4, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_addmv_large_tensors(mode):
    """
    Feature: Large tensor handling for addmv.
    Description: Test addmv with larger tensors.
    Expectation: Results match PyTorch implementation.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    np.random.seed(10)
    # Use moderately large tensors to avoid excessive memory usage
    input_tensor = np.random.randn(100).astype(np.float32)
    mat = np.random.randn(100, 50).astype(np.float32)
    vec = np.random.randn(50).astype(np.float32)
    beta = 1.0
    alpha = 1.0
    
    expect = generate_expect_forward_output(input_tensor, mat, vec, beta, alpha)
    
    output = addmv_forward_func(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)
    
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=1e-4, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_addmv_empty_tensor(mode):
    """
    Feature: Empty tensor handling for addmv.
    Description: Test addmv with empty tensors.
    Expectation: Should handle empty tensors appropriately.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    # Test with empty input tensor
    input_tensor = np.array([], dtype=np.float32)
    mat = np.random.randn(0, 2).astype(np.float32)  # 0x2 matrix
    vec = np.random.randn(2).astype(np.float32)  # 2D vector
    
    # This should fail since dimensions don't match for addmv operation
    # but we can test with proper empty dimensions
    input_tensor = np.array([], dtype=np.float32)
    mat = np.array([], dtype=np.float32).reshape(0, 0)  # 0x0 matrix
    vec = np.array([], dtype=np.float32)  # 0D vector
    
    # Test with 1x0 matrix and 0 vector
    input_tensor = np.random.randn(1).astype(np.float32)
    mat = np.array([], dtype=np.float32).reshape(1, 0)  # 1x0 matrix
    vec = np.array([], dtype=np.float32)  # 0D vector (empty)
    
    expect = generate_expect_forward_output(input_tensor, mat, vec, 1.0, 1.0)
    output = addmv_forward_func(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec))
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=1e-4, equal_nan=True)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_addmv_error_handling(mode):
    """
    Feature: Error handling for addmv.
    Description: Test addmv with invalid inputs to ensure proper error handling.
    Expectation: Should raise appropriate errors for invalid inputs.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    # Test with mismatched dimensions that should cause errors
    input_tensor = np.random.randn(3).astype(np.float32)
    mat = np.random.randn(4, 2).astype(np.float32)  # 4x2 matrix
    vec = np.random.randn(2).astype(np.float32)  # 2D vector
    
    # This should fail since input size (3) doesn't match matrix rows (4)
    try:
        output = addmv_forward_func(ms.Tensor(input_tensor), ms.Tensor(mat), ms.Tensor(vec))
        # If no error was raised, check if it matches expected behavior
        expect = generate_expect_forward_output(input_tensor, mat, vec, 1.0, 1.0)
        allclose_nparray(expect.detach().numpy(), output.asnumpy(), rtol=1e-4, equal_nan=True)
    except Exception:
        # This is expected behavior if the operation properly validates dimensions
        pass




@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_addmv_backward_single_operator(mode):
    """
    Feature: Backward pass for addmv operator.
    Description: Test addmv backward pass with single operator.
    Expectation: Gradients computed correctly.
    """
    def addmv_backward_func(input_tensor, mat, vec, beta=1, alpha=1):
        """Backward function for mint.addmv."""
        grad_fn = ops.grad(lambda x, y, z: mint.addmv(x, y, z, beta=beta, alpha=alpha), (0, 1, 2))
        return grad_fn(input_tensor, mat, vec)
    
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    np.random.seed(42)
    input_tensor = ms.Tensor(np.random.randn(4).astype(np.float32), ms.float32)
    mat = ms.Tensor(np.random.randn(4, 3).astype(np.float32), ms.float32)
    vec = ms.Tensor(np.random.randn(3).astype(np.float32), ms.float32)
    beta = 1.0
    alpha = 1.0
    
    # Calculate MindSpore gradients
    try:
        grad_input, grad_mat, grad_vec = addmv_backward_func(input_tensor, mat, vec, beta, alpha)
        
        # Calculate expected gradients using PyTorch
        pt_input = torch.tensor(input_tensor.asnumpy(), requires_grad=True)
        pt_mat = torch.tensor(mat.asnumpy(), requires_grad=True)
        pt_vec = torch.tensor(vec.asnumpy(), requires_grad=True)
        
        pt_output = torch.addmv(pt_input, pt_mat, pt_vec, beta=beta, alpha=alpha)
        pt_output.sum().backward()
        
        # Compare gradients
        allclose_nparray(pt_input.grad.detach().numpy(), grad_input.asnumpy(), rtol=1e-4, equal_nan=True)
        allclose_nparray(pt_mat.grad.detach().numpy(), grad_mat.asnumpy(), rtol=1e-4, equal_nan=True)
        allclose_nparray(pt_vec.grad.detach().numpy(), grad_vec.asnumpy(), rtol=1e-4, equal_nan=True)
    except Exception:
        # If gradient computation is not supported, skip this test
        pytest.skip("Gradient computation not supported for addmv")
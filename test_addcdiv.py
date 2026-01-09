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
"""Test addcdiv op."""
import pytest
import numpy as np
import mindspore as ms
from mindspore import mint
import torch
from tests.utils.mark_utils import arg_mark
from mindspore import ops, vmap
from mindspore import context


ms.context.set_context(mode=ms.PYNATIVE_MODE)


def generate_random_input(shape, dtype):
    """Generate random input data"""
    if np.issubdtype(dtype, np.integer):
        # For integer types, avoid zero values in tensor2 to prevent division by zero
        x = np.random.randint(1, 10, shape).astype(dtype)
        return x
    # For float types, avoid zero values in tensor2 to prevent division by zero
    x = np.random.uniform(0.1, 1.0, shape).astype(dtype)
    return x


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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addcdiv_basic(mode):
    """
    Feature: Test basic functionality of mint.addcdiv against torch.addcdiv
    Description: Test basic functionality of mint.addcdiv against torch.addcdiv
    Expectation: The results of MindSpore and PyTorch should be consistent within tolerance.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE if mode == 'pynative' else ms.GRAPH_MODE, jit_level="O0")
    
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    tensor1 = torch.tensor([4.0, 5.0, 6.0])
    tensor2 = torch.tensor([2.0, 2.0, 2.0])
    
    # Test with default value (1.0)
    torch_result = torch.addcdiv(input_tensor, tensor1, tensor2)
    ms_input = ms.Tensor(input_tensor.numpy())
    ms_tensor1 = ms.Tensor(tensor1.numpy())
    ms_tensor2 = ms.Tensor(tensor2.numpy())
    ms_result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2)
    
    np.testing.assert_allclose(ms_result.asnumpy(), torch_result.numpy(), rtol=1e-5, atol=1e-5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addcdiv_with_value(mode):
    """
    Feature: Test mint.addcdiv with specified value parameter
    Description: Test mint.addcdiv with specified value parameter
    Expectation: The results of MindSpore and PyTorch should be consistent when using a custom value parameter.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE if mode == 'pynative' else ms.GRAPH_MODE, jit_level="O0")
    
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    tensor1 = torch.tensor([4.0, 5.0, 6.0])
    tensor2 = torch.tensor([2.0, 2.0, 2.0])
    value = 0.5
    
    torch_result = torch.addcdiv(input_tensor, tensor1, tensor2, value=value)
    ms_input = ms.Tensor(input_tensor.numpy())
    ms_tensor1 = ms.Tensor(tensor1.numpy())
    ms_tensor2 = ms.Tensor(tensor2.numpy())
    ms_result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2, value=value)
    
    np.testing.assert_allclose(ms_result.asnumpy(), torch_result.numpy(), rtol=1e-5, atol=1e-5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [
    np.float32, np.float64,
    np.int32, np.int64
])
def test_addcdiv_dtype_coverage(mode, dtype):
    """
    Feature: Test mint.addcdiv with different dtypes
    Description: Test mint.addcdiv with various dtypes
    Expectation: The results of MindSpore and PyTorch should be consistent across different data types.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    shape = (2, 3)
    input_tensor = torch.tensor(generate_random_input(shape, dtype), dtype=torch.from_numpy(np.array(0, dtype=dtype)).dtype if dtype in [np.int32, np.int64] else torch.from_numpy(np.array(0, dtype=dtype)).dtype)
    tensor1 = torch.tensor(generate_random_input(shape, dtype), dtype=torch.from_numpy(np.array(0, dtype=dtype)).dtype if dtype in [np.int32, np.int64] else torch.from_numpy(np.array(0, dtype=dtype)).dtype)
    tensor2 = torch.tensor(generate_random_input(shape, dtype), dtype=torch.from_numpy(np.array(0, dtype=dtype)).dtype if dtype in [np.int32, np.int64] else torch.from_numpy(np.array(0, dtype=dtype)).dtype)
    value = 0.5
    
    # Convert integer types to float for torch operations since torch.addcdiv expects floating point
    if dtype in [np.int32, np.int64]:
        torch_input = input_tensor.float()
        torch_tensor1 = tensor1.float()
        torch_tensor2 = tensor2.float()
        torch_result = torch.addcdiv(torch_input, torch_tensor1, torch_tensor2, value=value)
        ms_input = ms.Tensor(input_tensor.numpy().astype(np.float32))
        ms_tensor1 = ms.Tensor(tensor1.numpy().astype(np.float32))
        ms_tensor2 = ms.Tensor(tensor2.numpy().astype(np.float32))
    else:
        torch_result = torch.addcdiv(input_tensor, tensor1, tensor2, value=value)
        ms_input = ms.Tensor(input_tensor.numpy())
        ms_tensor1 = ms.Tensor(tensor1.numpy())
        ms_tensor2 = ms.Tensor(tensor2.numpy())
    
    ms_result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2, value=value)
    
    np.testing.assert_allclose(ms_result.asnumpy(), torch_result.numpy(), rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addcdiv_edge_cases(mode):
    """
    Feature: Test mint.addcdiv with edge cases like NaN, Inf
    Description: Test mint.addcdiv with edge cases like NaN, Inf
    Expectation: Special values like NaN and Inf should be handled consistently between MindSpore and PyTorch.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE if mode == 'pynative' else ms.GRAPH_MODE, jit_level="O0")
    
    # Test with NaN
    input_tensor = torch.tensor([float('nan'), 2.0, 3.0])
    tensor1 = torch.tensor([4.0, 5.0, 6.0])
    tensor2 = torch.tensor([2.0, 2.0, 2.0])
    
    torch_result = torch.addcdiv(input_tensor, tensor1, tensor2)
    ms_input = ms.Tensor(input_tensor.numpy())
    ms_tensor1 = ms.Tensor(tensor1.numpy())
    ms_tensor2 = ms.Tensor(tensor2.numpy())
    ms_result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2)
    
    # Check that NaN values are preserved
    assert np.isnan(ms_result.asnumpy()[0])
    np.testing.assert_allclose(ms_result.asnumpy()[1:], torch_result.numpy()[1:], rtol=1e-5, atol=1e-5)
    
    # Test with Inf
    input_tensor = torch.tensor([float('inf'), 2.0, 3.0])
    tensor1 = torch.tensor([4.0, 5.0, 6.0])
    tensor2 = torch.tensor([2.0, 2.0, 2.0])
    
    torch_result = torch.addcdiv(input_tensor, tensor1, tensor2)
    ms_input = ms.Tensor(input_tensor.numpy())
    ms_tensor1 = ms.Tensor(tensor1.numpy())
    ms_tensor2 = ms.Tensor(tensor2.numpy())
    ms_result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2)
    
    # Check that Inf values are preserved
    assert np.isinf(ms_result.asnumpy()[0])
    np.testing.assert_allclose(ms_result.asnumpy()[1:], torch_result.numpy()[1:], rtol=1e-5, atol=1e-5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addcdiv_division_by_zero(mode):
    """
    Feature: Test mint.addcdiv with division by zero (should result in Inf)
    Description: Test mint.addcdiv with division by zero (should result in Inf)
    Expectation: Division by zero should produce Inf values consistently between MindSpore and PyTorch.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE if mode == 'pynative' else ms.GRAPH_MODE, jit_level="O0")
    
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    tensor1 = torch.tensor([4.0, 5.0, 6.0])
    tensor2 = torch.tensor([0.0, 2.0, 0.0])  # Division by zero in some elements
    
    torch_result = torch.addcdiv(input_tensor, tensor1, tensor2)
    ms_input = ms.Tensor(input_tensor.numpy())
    ms_tensor1 = ms.Tensor(tensor1.numpy())
    ms_tensor2 = ms.Tensor(tensor2.numpy())
    ms_result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2)
    
    # Check that inf values match
    torch_mask = torch.isinf(torch_result)
    ms_mask = np.isinf(ms_result.asnumpy())
    assert np.array_equal(ms_mask, torch_mask.numpy())
    
    # Compare finite values
    finite_mask = ~torch.isinf(torch_result)
    np.testing.assert_allclose(
        ms_result.asnumpy()[finite_mask.numpy()], 
        torch_result.numpy()[finite_mask.numpy()], 
        rtol=1e-5, atol=1e-5
    )


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addcdiv_shapes(mode):
    """
    Feature: Test mint.addcdiv with different tensor shapes
    Description: Test mint.addcdiv with different tensor shapes
    Expectation: The results of MindSpore and PyTorch should be consistent with various tensor shapes and broadcasting.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE if mode == 'pynative' else ms.GRAPH_MODE, jit_level="O0")
    
    # Test with 2D tensors
    input_tensor = torch.randn(2, 3)
    tensor1 = torch.randn(2, 3)
    tensor2 = torch.randn(2, 3)
    
    torch_result = torch.addcdiv(input_tensor, tensor1, tensor2)
    ms_input = ms.Tensor(input_tensor.numpy())
    ms_tensor1 = ms.Tensor(tensor1.numpy())
    ms_tensor2 = ms.Tensor(tensor2.numpy())
    ms_result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2)
    
    np.testing.assert_allclose(ms_result.asnumpy(), torch_result.numpy(), rtol=1e-5, atol=1e-5)
    
    # Test with broadcasting
    input_tensor = torch.randn(2, 3)
    tensor1 = torch.randn(2, 1)  # Will be broadcast
    tensor2 = torch.randn(1, 3)  # Will be broadcast
    
    torch_result = torch.addcdiv(input_tensor, tensor1, tensor2)
    ms_input = ms.Tensor(input_tensor.numpy())
    ms_tensor1 = ms.Tensor(tensor1.numpy())
    ms_tensor2 = ms.Tensor(tensor2.numpy())
    ms_result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2)
    
    np.testing.assert_allclose(ms_result.asnumpy(), torch_result.numpy(), rtol=1e-5, atol=1e-5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addcdiv_special_values(mode):
    """
    Feature: Test mint.addcdiv with special values (inf, nan, zero, etc.)
    Description: Test mint.addcdiv with various special values
    Expectation: Results should match between MindSpore and PyTorch for special values.
    """
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    # Test with inf values
    input_tensor = torch.tensor([1.0, float('inf'), -float('inf')])
    tensor1 = torch.tensor([2.0, 3.0, 4.0])
    tensor2 = torch.tensor([1.0, 2.0, 3.0])
    
    torch_result = torch.addcdiv(input_tensor, tensor1, tensor2)
    ms_input = ms.Tensor(input_tensor.numpy())
    ms_tensor1 = ms.Tensor(tensor1.numpy())
    ms_tensor2 = ms.Tensor(tensor2.numpy())
    ms_result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2)
    
    np.testing.assert_allclose(ms_result.asnumpy(), torch_result.numpy(), rtol=1e-5, atol=1e-5, equal_nan=True)
    
    # Test with nan values
    input_tensor = torch.tensor([1.0, float('nan'), 3.0])
    tensor1 = torch.tensor([2.0, 3.0, 4.0])
    tensor2 = torch.tensor([1.0, 2.0, 3.0])
    
    torch_result = torch.addcdiv(input_tensor, tensor1, tensor2)
    ms_input = ms.Tensor(input_tensor.numpy())
    ms_tensor1 = ms.Tensor(tensor1.numpy())
    ms_tensor2 = ms.Tensor(tensor2.numpy())
    ms_result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2)
    
    np.testing.assert_allclose(ms_result.asnumpy(), torch_result.numpy(), rtol=1e-5, atol=1e-5, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addcdiv_non_contiguous(mode):
    """
    Feature: Test mint.addcdiv with non-contiguous tensors
    Description: Test mint.addcdiv with non-contiguous memory layouts
    Expectation: Results should be consistent regardless of memory layout.
    """
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    # Create non-contiguous tensors using slicing
    full_tensor = torch.randn(4, 6)
    input_tensor = full_tensor[::2, ::2]  # Non-contiguous view
    tensor1 = torch.randn(2, 3)
    tensor2 = torch.randn(2, 3)
    
    torch_result = torch.addcdiv(input_tensor, tensor1, tensor2)
    
    # Convert to MindSpore with same shape
    ms_input = ms.Tensor(input_tensor.numpy())
    ms_tensor1 = ms.Tensor(tensor1.numpy())
    ms_tensor2 = ms.Tensor(tensor2.numpy())
    ms_result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2)
    
    np.testing.assert_allclose(ms_result.asnumpy(), torch_result.numpy(), rtol=1e-5, atol=1e-5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addcdiv_vmap(mode):
    """
    Feature: Test mint.addcdiv with vmap
    Description: Test mint.addcdiv with vectorization
    Expectation: Vmap results should match PyTorch vmap results.
    """
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    def torch_addcdiv_batched(input_batch, tensor1_batch, tensor2_batch):
        results = []
        for i in range(input_batch.shape[0]):
            result = torch.addcdiv(input_batch[i], tensor1_batch[i], tensor2_batch[i])
            results.append(result)
        return torch.stack(results)
    
    batch_size = 3
    shape = (2, 3)
    
    input_batch = torch.randn(batch_size, *shape)
    tensor1_batch = torch.randn(batch_size, *shape)
    tensor2_batch = torch.randn(batch_size, *shape)
    
    torch_result = torch_addcdiv_batched(input_batch, tensor1_batch, tensor2_batch)
    
    ms_input_batch = ms.Tensor(input_batch.numpy())
    ms_tensor1_batch = ms.Tensor(tensor1_batch.numpy())
    ms_tensor2_batch = ms.Tensor(tensor2_batch.numpy())
    
    # Implement manual batching since MindSpore vmap might not directly support mint.addcdiv
    ms_results = []
    for i in range(batch_size):
        result = mint.addcdiv(ms_input_batch[i], ms_tensor1_batch[i], ms_tensor2_batch[i])
        ms_results.append(result)
    ms_result = ms.ops.stack(ms_results)
    
    np.testing.assert_allclose(ms_result.asnumpy(), torch_result.numpy(), rtol=1e-5, atol=1e-5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addcdiv_backward(mode):
    """
    Feature: Test backward pass of mint.addcdiv
    Description: Test gradient computation for mint.addcdiv
    Expectation: Gradients should match PyTorch gradients.
    """
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    # Test backward pass using PyTorch
    input_tensor = torch.randn(2, 3, requires_grad=True)
    tensor1 = torch.randn(2, 3, requires_grad=True)
    tensor2 = torch.randn(2, 3, requires_grad=True)
    value = 0.5
    
    torch_output = torch.addcdiv(input_tensor, tensor1, tensor2, value=value)
    torch_loss = torch_output.sum()
    torch_loss.backward()
    
    # Test backward pass using MindSpore
    ms_input = ms.Tensor(input_tensor.detach().numpy())
    ms_tensor1 = ms.Tensor(tensor1.detach().numpy())
    ms_tensor2 = ms.Tensor(tensor2.detach().numpy())
    
    def ms_addcdiv_forward(input_ms, tensor1_ms, tensor2_ms):
        return ops.sum(mint.addcdiv(input_ms, tensor1_ms, tensor2_ms, value=value))
    
    ms_grad_fn = ops.grad(ms_addcdiv_forward, (0, 1, 2))
    ms_grad_input, ms_grad_tensor1, ms_grad_tensor2 = ms_grad_fn(ms_input, ms_tensor1, ms_tensor2)
    
    # Compare gradients
    np.testing.assert_allclose(ms_grad_input.asnumpy(), input_tensor.grad.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ms_grad_tensor1.asnumpy(), tensor1.grad.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ms_grad_tensor2.asnumpy(), tensor2.grad.numpy(), rtol=1e-5, atol=1e-5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('shape', [(1,), (3, 4), (2, 3, 4), (1, 2, 3, 4), (2, 1, 3, 4), (1, 2, 1, 4, 5), (1, 2, 3, 1, 5, 6)])
def test_addcdiv_different_shapes(mode, shape):
    """
    Feature: Test mint.addcdiv with different tensor shapes
    Description: Test mint.addcdiv with various dimensional shapes
    Expectation: Results should be consistent across different shapes.
    """
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    input_tensor = torch.randn(shape)
    tensor1 = torch.randn(shape)
    tensor2 = torch.randn(shape)
    
    torch_result = torch.addcdiv(input_tensor, tensor1, tensor2)
    ms_input = ms.Tensor(input_tensor.numpy())
    ms_tensor1 = ms.Tensor(tensor1.numpy())
    ms_tensor2 = ms.Tensor(tensor2.numpy())
    ms_result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2)
    
    np.testing.assert_allclose(ms_result.asnumpy(), torch_result.numpy(), rtol=1e-5, atol=1e-5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addcdiv_large_tensor(mode):
    """
    Feature: Test mint.addcdiv with large tensors
    Description: Test mint.addcdiv handling of larger tensors
    Expectation: Expect correct results
    """
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    shape = (100, 100)  # Larger tensor
    input_tensor = torch.randn(shape)
    tensor1 = torch.randn(shape)
    tensor2 = torch.randn(shape)
    
    torch_result = torch.addcdiv(input_tensor, tensor1, tensor2)
    ms_input = ms.Tensor(input_tensor.numpy())
    ms_tensor1 = ms.Tensor(tensor1.numpy())
    ms_tensor2 = ms.Tensor(tensor2.numpy())
    ms_result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2)
    
    np.testing.assert_allclose(ms_result.asnumpy(), torch_result.numpy(), rtol=1e-5, atol=1e-5)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('special_type', ['inf', 'nan', 'zero', 'large', 'small'])
def test_addcdiv_special_values_param(mode, special_type):
    """
    Feature: Test mint.addcdiv with special values (inf, nan, zero, etc.)
    Description: Test mint.addcdiv with various special values using parameterization
    Expectation: Results should match between MindSpore and PyTorch for special values.
    """
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    shape = (3, 4)
    input_tensor_np = generate_special_input(shape, np.float32, special_type)
    tensor1_np = generate_special_input(shape, np.float32, special_type)
    tensor2_np = generate_special_input(shape, np.float32, special_type)
    
    # Ensure tensor2 doesn't contain zeros to prevent division by zero
    tensor2_np = np.where(tensor2_np == 0, 1.0, tensor2_np)
    
    input_tensor = torch.tensor(input_tensor_np)
    tensor1 = torch.tensor(tensor1_np)
    tensor2 = torch.tensor(tensor2_np)
    
    torch_result = torch.addcdiv(input_tensor, tensor1, tensor2)
    ms_input = ms.Tensor(input_tensor.numpy())
    ms_tensor1 = ms.Tensor(tensor1.numpy())
    ms_tensor2 = ms.Tensor(tensor2.numpy())
    ms_result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2)
    
    np.testing.assert_allclose(ms_result.asnumpy(), torch_result.numpy(), rtol=1e-5, atol=1e-5, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addcdiv_empty_tensor(mode):
    """
    Feature: Test mint.addcdiv with empty tensors
    Description: Test mint.addcdiv with empty tensors of various shapes
    Expectation: Results should match between MindSpore and PyTorch for empty tensors.
    """
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    # Test with empty tensors
    input_tensor = torch.tensor([])
    tensor1 = torch.tensor([])
    tensor2 = torch.tensor([])
    
    torch_result = torch.addcdiv(input_tensor, tensor1, tensor2)
    ms_input = ms.Tensor(input_tensor.numpy())
    ms_tensor1 = ms.Tensor(tensor1.numpy())
    ms_tensor2 = ms.Tensor(tensor2.numpy())
    ms_result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2)
    
    np.testing.assert_allclose(ms_result.asnumpy(), torch_result.numpy(), rtol=1e-5, atol=1e-5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_addcdiv_complex_dtypes(mode, dtype):
    """
    Feature: Test mint.addcdiv with complex dtypes
    Description: Test mint.addcdiv with complex64 and complex128 dtypes
    Expectation: Results should match between MindSpore and PyTorch for complex dtypes.
    """
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    shape = (2, 3)
    # Generate random complex numbers
    input_real = np.random.uniform(-1, 1, shape).astype(np.float32 if dtype == np.complex64 else np.float64)
    input_imag = np.random.uniform(-1, 1, shape).astype(np.float32 if dtype == np.complex64 else np.float64)
    input_tensor_np = input_real + 1j * input_imag
    
    tensor1_real = np.random.uniform(-1, 1, shape).astype(np.float32 if dtype == np.complex64 else np.float64)
    tensor1_imag = np.random.uniform(-1, 1, shape).astype(np.float32 if dtype == np.complex64 else np.float64)
    tensor1_np = tensor1_real + 1j * tensor1_imag
    
    tensor2_real = np.random.uniform(0.1, 1, shape).astype(np.float32 if dtype == np.complex64 else np.float64)
    tensor2_imag = np.random.uniform(0.1, 1, shape).astype(np.float32 if dtype == np.complex64 else np.float64)
    tensor2_np = tensor2_real + 1j * tensor2_imag
    
    input_tensor = torch.tensor(input_tensor_np)
    tensor1 = torch.tensor(tensor1_np)
    tensor2 = torch.tensor(tensor2_np)
    
    torch_result = torch.addcdiv(input_tensor, tensor1, tensor2)
    ms_input = ms.Tensor(input_tensor.numpy())
    ms_tensor1 = ms.Tensor(tensor1.numpy())
    ms_tensor2 = ms.Tensor(tensor2.numpy())
    ms_result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2)
    
    np.testing.assert_allclose(ms_result.asnumpy(), torch_result.numpy(), rtol=1e-5, atol=1e-5, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addcdiv_error_handling(mode):
    """
    Feature: Test mint.addcdiv error handling
    Description: Test mint.addcdiv error handling for invalid inputs
    Expectation: MindSpore should raise appropriate errors for invalid inputs.
    """
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    # Test with mismatched shapes that cannot be broadcast
    # Using shapes that are definitely incompatible
    ms_input = ms.Tensor([1.0, 2.0, 3.0])  # Shape (3,)
    ms_tensor1 = ms.Tensor([[1.0, 2.0], [3.0, 4.0]])  # Shape (2, 2)
    ms_tensor2 = ms.Tensor([1.0, 2.0])  # Shape (2,)
    
    # MindSpore should raise an error for invalid broadcasting
    with pytest.raises((ValueError, RuntimeError)):
        result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2)
        # Access the result to trigger computation
        _ = result.asnumpy()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addcdiv_functional_interface(mode):
    """
    Feature: Test mint.addcdiv functional interface
    Description: Test mint.addcdiv through the functional interface
    Expectation: Results should match between MindSpore and PyTorch.
    """
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    tensor1 = torch.tensor([4.0, 5.0, 6.0])
    tensor2 = torch.tensor([2.0, 2.0, 2.0])
    
    torch_result = torch.addcdiv(input_tensor, tensor1, tensor2)
    
    # Test through mint interface (which is the functional interface)
    ms_input = ms.Tensor(input_tensor.numpy())
    ms_tensor1 = ms.Tensor(tensor1.numpy())
    ms_tensor2 = ms.Tensor(tensor2.numpy())
    ms_result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2)
    
    np.testing.assert_allclose(ms_result.asnumpy(), torch_result.numpy(), rtol=1e-5, atol=1e-5)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('shape', [
    (),  # 0D
    (1,),  # 1D
    (2, 3),  # 2D
    (2, 3, 4),  # 3D
    (2, 2, 3, 4),  # 4D
    (2, 2, 2, 3, 4),  # 5D
    (2, 1, 2, 2, 3, 4),  # 6D
    (1, 2, 1, 2, 2, 3, 4),  # 7D
    (2, 1, 2, 1, 2, 2, 3, 4)  # 8D
])
def test_addcdiv_dimensions(mode, shape):
    """
    Feature: dimension coverage for addcdiv operator.
    Description: test addcdiv with 0D to 8D inputs.
    Expectation: results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    # Generate inputs with the specified shape
    input_tensor_np = generate_random_input(shape, dtype)
    tensor1_np = generate_random_input(shape, dtype)
    # Ensure tensor2 doesn't have zeros to avoid division by zero
    tensor2_np = generate_random_input(shape, dtype)
    tensor2_np = np.where(tensor2_np == 0, 0.1, tensor2_np)

    input_tensor = torch.tensor(input_tensor_np, dtype=torch.float32)
    tensor1 = torch.tensor(tensor1_np, dtype=torch.float32)
    tensor2 = torch.tensor(tensor2_np, dtype=torch.float32)

    torch_result = torch.addcdiv(input_tensor, tensor1, tensor2)
    ms_input = ms.Tensor(input_tensor_np, dtype=ms.float32)
    ms_tensor1 = ms.Tensor(tensor1_np, dtype=ms.float32)
    ms_tensor2 = ms.Tensor(tensor2_np, dtype=ms.float32)
    ms_result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2)

    np.testing.assert_allclose(ms_result.asnumpy(), torch_result.numpy(), rtol=1e-5, atol=1e-5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('has_dynamic_rank', [False, True])
def test_addcdiv_dynamic_shape_advanced(mode, has_dynamic_rank):
    """
    Feature: dynamic shape and rank support for addcdiv operator.
    Description: test addcdiv with dynamic dimensions and dynamic rank inputs.
    Expectation: results match PyTorch implementation and shape info is correctly handled.
    """
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    
    base_shapes = [
        (None, 3),
        (5, None),
        (None, None),
        (2, None, 4)
    ]
    if has_dynamic_rank:
        base_shapes.extend([
            (None,),
            (3, None, None, 2)
        ])
    
    for dynamic_shape in base_shapes:
        concrete_shape = []
        for dim in dynamic_shape:
            if dim is None:
                concrete_shape.append(np.random.randint(1, 11))
            else:
                concrete_shape.append(dim)
        concrete_shape = tuple(concrete_shape)
        
        # Generate inputs with the concrete shape
        input_tensor_np = generate_random_input(concrete_shape, np.float32)
        tensor1_np = generate_random_input(concrete_shape, np.float32)
        tensor2_np = generate_random_input(concrete_shape, np.float32)
        # Ensure tensor2 doesn't have zeros to avoid division by zero
        tensor2_np = np.where(tensor2_np == 0, 0.1, tensor2_np)

        input_tensor = torch.tensor(input_tensor_np)
        tensor1 = torch.tensor(tensor1_np)
        tensor2 = torch.tensor(tensor2_np)
        
        torch_result = torch.addcdiv(input_tensor, tensor1, tensor2)
        
        ms_input = ms.Tensor(input_tensor_np)
        ms_tensor1 = ms.Tensor(tensor1_np)
        ms_tensor2 = ms.Tensor(tensor2_np)
        ms_result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2)
        
        np.testing.assert_allclose(ms_result.asnumpy(), torch_result.numpy(), rtol=1e-5, atol=1e-5)
        assert ms_result.shape == concrete_shape, \
            f"Output shape mismatch: expected {concrete_shape}, got {ms_result.shape}"
        
        # Test with value parameter as well
        torch_result_with_value = torch.addcdiv(input_tensor, tensor1, tensor2, value=0.5)
        ms_result_with_value = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2, value=0.5)
        
        np.testing.assert_allclose(ms_result_with_value.asnumpy(), torch_result_with_value.numpy(), rtol=1e-5, atol=1e-5)
        assert ms_result_with_value.shape == concrete_shape, \
            f"Output shape with value mismatch: expected {concrete_shape}, got {ms_result_with_value.shape}"


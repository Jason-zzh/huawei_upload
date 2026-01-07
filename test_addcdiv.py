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

ms.context.set_context(mode=ms.PYNATIVE_MODE)


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
def test_addcdiv_different_dtypes(mode):
    """
    Feature: Test mint.addcdiv with different dtypes
    Description: Test mint.addcdiv with different dtypes
    Expectation: The results of MindSpore and PyTorch should be consistent across different floating point data types.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE if mode == 'pynative' else ms.GRAPH_MODE, jit_level="O0")
    
    dtypes = [torch.float32, torch.float64]
    
    for dtype in dtypes:    
        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
        tensor1 = torch.tensor([4.0, 5.0, 6.0], dtype=dtype)
        tensor2 = torch.tensor([2.0, 2.0, 2.0], dtype=dtype)
        value = 0.5
        
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


if __name__ == '__main__':
    test_addcdiv_basic()
    test_addcdiv_with_value()
    test_addcdiv_different_dtypes()
    test_addcdiv_edge_cases()
    test_addcdiv_division_by_zero()
    test_addcdiv_shapes()
    print("All tests passed!")

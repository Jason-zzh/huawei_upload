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
""" addcdiv op performance test case """
import time
import numpy as np
import mindspore as ms
from mindspore import mint
from tests.utils.test_op_utils import BACKGROUND_NOISE
from tests.utils.mark_utils import arg_mark
import torch
import pytest


def addcdiv_forward_perf(input_tensor, tensor1, tensor2, value=1.0):
    """Performance test for mint.addcdiv forward pass"""
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    
    # Convert to MindSpore tensors
    ms_input = ms.Tensor(input_tensor)
    ms_tensor1 = ms.Tensor(tensor1)
    ms_tensor2 = ms.Tensor(tensor2)
    
    # Warm up
    for _ in range(5):
        _ = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2, value=value)
    
    # Synchronize before timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    # Run the operation multiple times for more accurate timing
    for _ in range(10):
        result = mint.addcdiv(ms_input, ms_tensor1, ms_tensor2, value=value)
    
    # Synchronize after timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    return (end_time - start_time) / 10  # Return average time


def generate_expect_addcdiv_forward_perf(input_tensor, tensor1, tensor2, value=1.0):
    """Generate expected performance for torch.addcdiv forward pass"""
    # Warm up
    for _ in range(5):
        _ = torch.addcdiv(input_tensor, tensor1, tensor2, value=value)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(10):
        result = torch.addcdiv(input_tensor, tensor1, tensor2, value=value)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    return (end_time - start_time) / 10  # Return average time


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addcdiv_perf_basic(mode):
    """Performance test for mint.addcdiv with basic tensor sizes"""
    ms.context.set_context(mode=ms.PYNATIVE_MODE if mode == 'pynative' else ms.GRAPH_MODE, jit_level="O0")
    
    # Test with medium-sized tensors
    size = (1000, 1000)
    input_tensor = torch.randn(size, dtype=torch.float32)
    tensor1 = torch.randn(size, dtype=torch.float32)
    tensor2 = torch.randn(size, dtype=torch.float32) + 0.1  # Avoid division by zero
    value = 0.5
    
    # Convert to numpy for MindSpore
    input_np = input_tensor.numpy()
    tensor1_np = tensor1.numpy()
    tensor2_np = tensor2.numpy()
    
    # Measure performance
    ms_perf = addcdiv_forward_perf(input_np, tensor1_np, tensor2_np, value=value)
    expect_perf = generate_expect_addcdiv_forward_perf(input_tensor, tensor1, tensor2, value=value)
    
    # Check that MindSpore performance is within 1.1x of PyTorch (allowing for some overhead)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all() if hasattr(np, 'less') else (ms_perf - BACKGROUND_NOISE) < (expect_perf * 1.1)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addcdiv_perf_different_sizes(mode):
    """Performance test for mint.addcdiv with different tensor sizes"""
    ms.context.set_context(mode=ms.PYNATIVE_MODE if mode == 'pynative' else ms.GRAPH_MODE, jit_level="O0")
    
    sizes = [(100, 100), (500, 500), (1000, 1000)]
    value = 0.5
    
    for size in sizes:
        input_tensor = torch.randn(size, dtype=torch.float32)
        tensor1 = torch.randn(size, dtype=torch.float32)
        tensor2 = torch.randn(size, dtype=torch.float32) + 0.1  # Avoid division by zero
        
        # Convert to numpy for MindSpore
        input_np = input_tensor.numpy()
        tensor1_np = tensor1.numpy()
        tensor2_np = tensor2.numpy()
        
        # Measure performance
        ms_perf = addcdiv_forward_perf(input_np, tensor1_np, tensor2_np, value=value)
        expect_perf = generate_expect_addcdiv_forward_perf(input_tensor, tensor1, tensor2, value=value)
        
        # Check that MindSpore performance is reasonable
        assert (ms_perf - BACKGROUND_NOISE) < (expect_perf * 1.5)  # Allow up to 1.5x slower


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_addcdiv_perf_with_default_value(mode):
    """Performance test for mint.addcdiv with default value parameter"""
    ms.context.set_context(mode=ms.PYNATIVE_MODE if mode == 'pynative' else ms.GRAPH_MODE, jit_level="O0")
    
    # Test with default value (1.0)
    size = (800, 800)
    input_tensor = torch.randn(size, dtype=torch.float32)
    tensor1 = torch.randn(size, dtype=torch.float32)
    tensor2 = torch.randn(size, dtype=torch.float32) + 0.1  # Avoid division by zero
    
    # Convert to numpy for MindSpore
    input_np = input_tensor.numpy()
    tensor1_np = tensor1.numpy()
    tensor2_np = tensor2.numpy()
    
    # Measure performance
    ms_perf = addcdiv_forward_perf(input_np, tensor1_np, tensor2_np)  # Default value=1.0
    expect_perf = generate_expect_addcdiv_forward_perf(input_tensor, tensor1, tensor2)  # Default value=1.0
    
    assert (ms_perf - BACKGROUND_NOISE) < (expect_perf * 1.1)  # Allow up to 1.1x slower


if __name__ == '__main__':
    test_addcdiv_perf_basic()
    test_addcdiv_perf_different_sizes()
    test_addcdiv_perf_with_default_value()
    print("Performance tests passed!")

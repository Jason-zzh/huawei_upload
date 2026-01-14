/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/extension.h>
#include <iostream>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {
extern "C" int AvgPool3DGradExt(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                                void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  // Extract inputs
  auto grad_output = tensors[0];        // gradient of the output
  auto input = tensors[1];              // original input tensor from forward pass
  auto kernel_size = input_utils.GetIntVecInput(2);  // kernel_size as vector
  auto stride = input_utils.GetIntVecInput(3);       // stride as vector
  auto padding = input_utils.GetIntVecInput(4);      // padding as vector
  bool ceil_mode = input_utils.GetBoolInput(5);      // ceil_mode
  bool count_include_pad = input_utils.GetBoolInput(6);  // count_include_pad
  auto divisor_override = input_utils.GetOptionalIntInput(7);  // divisor_override

  // Convert vectors to tuples for PyTorch
  auto kernel_sizes = at::IntArrayRef(kernel_size);
  auto strides = at::IntArrayRef(stride);
  auto paddings = at::IntArrayRef(padding);

  // Get output tensor
  auto output = tensors[nparam - 1];

  // Call PyTorch's avg_pool3d_backward implementation
  auto result = at::avg_pool3d_backward(grad_output, input, kernel_sizes, strides, paddings,
                                       ceil_mode, count_include_pad, divisor_override);
  
  // Copy result to output tensor
  output.copy_(result);

  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin

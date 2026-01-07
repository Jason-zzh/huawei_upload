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
#include <string.h>
#include <torch/extension.h>
#include <iostream>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {
extern "C" int AddcdivExt(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                          void *extra) {
  // Convert physical parameters passed by the framework to PyTorch tensor containers
  // Parameter list: [input, tensor1, tensor2, value, output]
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_output = tensors[nparam - 1];  // Output tensor
  auto at_input = tensors[0];            // Input tensor
  auto at_tensor1 = tensors[1];          // First tensor for division
  auto at_tensor2 = tensors[2];          // Second tensor for division

  // Extract value parameter (non-tensor scalar)
  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  c10::Scalar value_scalar = input_utils.GetScalarInput(nparam - 2);

  // Call ATen interface: output = input + value * (tensor1 / tensor2)
  // Using at::addcdiv_out which implements: input + value * (tensor1 / tensor2)
  at::addcdiv_out(at_output, at_input, at_tensor1, at_tensor2, value_scalar);

  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin
// Trigger rebuild

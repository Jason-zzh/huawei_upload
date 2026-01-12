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
extern "C" int FmodScalar(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                          void *extra) {
  // Get tensors with proper memory layout and precision preservation
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  auto at_input = tensors[0];
  auto at_output = tensors[nparam - 1];

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  
  at::Scalar at_other = input_utils.GetScalarInput(1);
  
  // Perform fmod operation with explicit control over precision
  at::fmod_out(at_output, at_input, at_other);

  // Ensure output tensor properties match input tensors exactly
  at_output = at_output.to(at_input.options());

  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin

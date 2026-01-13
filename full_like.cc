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
extern "C" int FullLike(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                        void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  // Parameters: [input_tensor, fill_value, output_tensor] and optional dtype
  auto input_tensor = tensors[0];
  auto at_output = tensors[nparam - 1];  // Last parameter is the output tensor
  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  // Get the fill_value scalar from second parameter
  c10::Scalar fill_value = input_utils.GetScalarInput(1);
  // Call ATen interface: at::full_out to fill the output tensor with the given value
  // Using the input tensor's shape and the output tensor's dtype
  at::full_out(at_output, input_tensor.sizes(), fill_value);
  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin

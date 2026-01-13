/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may not use this file except in compliance with the License.
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
extern "C" int FillScalar(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                          void *extra) {
  // Parameters: [size (list of ints), fill_value (scalar), output]
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_output = tensors[nparam - 1];  // Last parameter is the output tensor

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  
  // Get the size vector from first parameter
  auto size = input_utils.GetIntVecInput(0);
  
  // Get the fill value scalar from second parameter
  c10::Scalar fill_value = input_utils.GetScalarInput(1);

  // Call ATen interface: at::full_out with size and fill_value
  at::full_out(at_output, size, fill_value);
  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin

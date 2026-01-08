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
extern "C" int FloorDivScalar(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  auto at_input1 = tensors[0];  // tensor
  auto at_input2 = tensors[1];  // scalar tensor
  auto at_output = tensors[nparam - 1];  // output tensor

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  
  // Extract scalar value from the scalar tensor
  c10::Scalar scalar_val = at_input2.item();
  
  // Use the tensor-scalar version of floor_divide and copy to output
  at::Tensor result = at::native::floor_divide(at_input1, scalar_val);
  at_output.copy_(result);
  
  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin

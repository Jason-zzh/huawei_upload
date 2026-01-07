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
extern "C" int Addmm(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                     void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  at::Scalar beta = input_utils.GetScalarInput(3);  // beta parameter at index 3
  at::Scalar alpha = input_utils.GetScalarInput(4); // alpha parameter at index 4

  auto input = tensors[0];  // input tensor
  auto mat1 = tensors[1];   // mat1 tensor
  auto mat2 = tensors[2];   // mat2 tensor
  auto output = tensors[nparam - 1];  // output tensor

  at::addmm_out(output, input, mat1, mat2, beta, alpha);
  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin

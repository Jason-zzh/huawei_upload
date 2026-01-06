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
extern "C" int Addbmm(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                      void *extra) {
  // Convert physical parameters passed by the framework to PyTorch tensor containers
  // Parameter list: [input, batch1, batch2, beta, alpha, output]
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];      // input tensor
  auto at_batch1 = tensors[1];     // batch1 tensor
  auto at_batch2 = tensors[2];     // batch2 tensor
  auto at_output = tensors[nparam - 1];  // Output tensor

  // Extract beta and alpha parameters (non-tensor scalars)
  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  c10::Scalar beta_scalar = input_utils.GetScalarInput(nparam - 3);  // beta is the 3rd from last parameter
  c10::Scalar alpha_scalar = input_utils.GetScalarInput(nparam - 2); // alpha is the 2nd from last parameter

  // Call ATen interface: output = beta * input + alpha * sum(batch1 @ batch2)
  at::addbmm_out(at_output, at_input, at_batch1, at_batch2, beta_scalar, alpha_scalar);

  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin

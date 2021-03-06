/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "ml_utils.h"

namespace ML{

void pcaFit(float *input, float *components, float *explained_var,
                    float *explained_var_ratio, float *singular_vals, float *mu,
                    float *noise_vars, paramsPCA prms);
void pcaFit(double *input, double *components, double *explained_var,
                    double *explained_var_ratio, double *singular_vals, double *mu,
                    double *noise_vars, paramsPCA prms);
void pcaFitTransform(float *input, float *trans_input, float *components, float *explained_var,
                    float *explained_var_ratio, float *singular_vals, float *mu,
                    float *noise_vars, paramsPCA prms);
void pcaFitTransform(double *input, double *trans_input, double *components, double *explained_var,
                    double *explained_var_ratio, double *singular_vals, double *mu,
                    double *noise_vars, paramsPCA prms);
void pcaInverseTransform(float *trans_input, float *components, float *singular_vals, float *mu,
                    float *input, paramsPCA prms);
void pcaInverseTransform(double *trans_input, double *components, double *singular_vals, double *mu,
                    double *input, paramsPCA prms);
void pcaTransform(float *input, float *components, float *trans_input, float *singular_vals, float *mu, paramsPCA prms);
void pcaTransform(double *input, double *components, double *trans_input, double *singular_vals, double *mu, paramsPCA prms);
}


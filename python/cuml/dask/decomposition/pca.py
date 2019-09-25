#
# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from cuml.dask.common import extract_ddf_partitions
from cuml.decomposition import PCA as cuPCA
import cudf

from dask.distributed import default_client, wait
import math
import random
import numpy as np

class PCA:
    def __init__(self, copy=True, handle=None, iterated_power=15,
                 n_components=1, random_state=None, svd_solver='auto',
                 tol=1e-7, verbose=False, whiten=False, workers=None):

        c = default_client()
        if workers is None:
            workers = c.has_what().keys()  # Default to all workers
        self.workers = workers

        self.pcas = {
            worker: c.submit(PCA._func_build_pca, n, copy, 
                             handle, iterated_power,
                             n_components, random_state, svd_solver,
                             tol, verbose, whiten, random.random(),
                             workers=[worker])
            for n, worker in enumerate(workers)
        }

        pcas_wait = list()
        for p in self.pcas.values():
            pcas_wait.append(p)

        wait(pcas_wait)

    @staticmethod
    def _func_build_pca(n, copy, handle, iterated_power,
                 n_components, random_state, svd_solver,
                 tol, verbose, whiten, r):

        return cuPCA(copy=copy, handle=handle, iterated_power=iterated_power,
                 n_components=n_components, random_state=random_state, svd_solver=svd_solver,
                 tol=tol, verbose=verbose, whiten=whiten)

    @staticmethod
    def _fit(model, X_df_list, r):
        if len(X_df_list) == 1:
            X_df = X_df_list[0]
        else:
            X_df = cudf.concat(X_df_list)
            
        return model.fit_transform_opg(X_df)

    def fit(self, X):
        c = default_client()

        X_futures = c.sync(extract_ddf_partitions, X)
        
        X_partition_workers = [w for w, xc in X_futures.items()]
        
        if set(X_partition_workers) != set(self.workers):
            raise ValueError("""
              X is not partitioned on the same workers expected by RF\n
              X workers: %s\n
              y workers: %s\n
              RF workers: %s
            """ % (str(X_partition_workers),
                   str(self.workers)))

        f = list()
        for w, xc in X_futures.items():
            f.append(
                c.submit(
                    PCA._fit,
                    self.pcas[w],
                    xc,
                    random.random(),
                    workers=[w],
                )
            )

        wait(f)

        return self

    

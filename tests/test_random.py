# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from stellargraph.random import SeededPerBatch
import numpy as np


def test_seeded_per_batch():
    num_batches = 10
    num_iter = 10
    seed = 0

    def get_batches():
        s = SeededPerBatch(create_with_seed=lambda x: x, seed=seed)
        batches = [0] * num_batches

        # get batches in random order
        for batch_num in np.random.permutation(num_batches):
            batches[batch_num] = s[batch_num]

        return tuple(batches)

    assert len({get_batches() for _ in range(num_iter)}) == 1

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

from contextlib import contextmanager
import numpy as np
from stellargraph.random import set_seed


def models_equals(model1, model2):
    w1 = model1.get_weights()
    w2 = model2.get_weights()
    return all(np.array_equal(w, w_new) for w, w_new in zip(w1, w2))


def assert_reproducible(func, num_iter=20):
    """
    Assert Keras models produced from calling ``func`` are reproducible.

    Args:
        func (callable): Function to check for reproducible model
        num_iter (int, default 20): Number of iterations to run through to validate reproducibility.

    """
    model = func()
    for i in range(num_iter):
        model_new = func()
        assert models_equals(model, model_new), (
            model.get_weights(),
            model_new.get_weights(),
        )


@contextmanager
def use_seed(seed):
    """
    Convenience utility to use a particular global seed value with a context manager

    Args:
        seed: seed value
    """
    set_seed(seed)
    try:
        yield seed
    finally:
        set_seed(None)

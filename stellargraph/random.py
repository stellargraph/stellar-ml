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

import random as rn
import numpy.random as np_rn
from collections import namedtuple


RandomState = namedtuple("RandomState", "random, numpy")


def _global_state():
    return RandomState(rn, np_rn)


def _seeded_state(s):
    return RandomState(rn.Random(s), np_rn.RandomState(s))


_rs = _global_state()


def random_state(seed):
    if seed is None:
        return _rs
    else:
        return _seeded_state(seed)


def set_seed(seed):
    global _rs
    if seed is None:
        _rs = _global_state()
    else:
        _rs = _seeded_state(seed)

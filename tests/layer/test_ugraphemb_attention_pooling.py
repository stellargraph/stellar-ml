# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np
from stellargraph.layer.ugraphemb_attention_pooling import UGraphEmbAttentionPooling


def test_config():
    pool = UGraphEmbAttentionPooling(
        attention_activation="tanh",
        attention_initializer="zeros",
        attention_regularizer="l2",
        attention_constraint="unit_norm",
    )

    conf = pool.get_config()
    assert conf["attention_activation"] == "tanh"
    assert conf["attention_initializer"]["class_name"] == "Zeros"
    assert conf["attention_regularizer"]["class_name"] == "L1L2"
    assert conf["attention_constraint"]["class_name"] == "UnitNorm"


def test_call():
    pool = UGraphEmbAttentionPooling()
    shape = (3, 4, 5)
    sizes = [1, 4, 2]

    nodes = np.random.rand(*shape).astype(np.float32)
    mask = np.full(shape[:2], False)
    for i, size in enumerate(sizes):
        mask[i][:size] = True

    embeddings = pool(nodes, mask=mask)
    assert embeddings.shape == (3, 5)

    kernel = pool.attention_kernel.numpy()

    def sigmoid(x):
        # numerically stable
        return np.exp(-np.logaddexp(0, -x))

    def relu(x):
        return np.maximum(0, x)

    # check each of the graphs
    for emb, nod, msk, size in zip(embeddings, nodes, mask, sizes):
        node_embs = nod[msk]
        # internal consistency of this test:
        assert node_embs.shape[0] == size

        # naive reproduction of the paper's algorithm
        query = relu(np.mean(node_embs, axis=0) @ kernel)
        expected = sum(sigmoid(node @ query) * node for node in node_embs)

        np.testing.assert_allclose(emb, expected, rtol=1e-4)

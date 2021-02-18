# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
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


"""
GCN tests

"""

import stellargraph as sg
from stellargraph.layer.gcn import *
from stellargraph.mapper import FullBatchNodeGenerator, FullBatchLinkGenerator
from stellargraph.core.graph import StellarGraph
from stellargraph.core.utils import GCN_Aadj_feats_op

import networkx as nx
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pytest
from ..test_utils.graphs import create_graph_features
from .. import test_utils


def test_GraphConvolution_config():
    gcn_layer = GraphConvolution(units=16)
    conf = gcn_layer.get_config()

    assert conf["units"] == 16
    assert conf["activation"] == "linear"
    assert conf["use_bias"] == True
    assert conf["kernel_initializer"]["class_name"] == "GlorotUniform"
    assert conf["bias_initializer"]["class_name"] == "Zeros"
    assert conf["kernel_regularizer"] == None
    assert conf["bias_regularizer"] == None
    assert conf["kernel_constraint"] == None
    assert conf["bias_constraint"] == None


def test_GraphConvolution_init():
    gcn_layer = GraphConvolution(units=16, activation="relu")

    assert gcn_layer.units == 16
    assert gcn_layer.use_bias == True
    assert gcn_layer.get_config()["activation"] == "relu"


def test_GraphConvolution_dense():
    G, features = create_graph_features()

    # We need to specify the batch shape as one for the GraphConvolutional logic to work
    x_t = Input(batch_shape=(1,) + features.shape, name="X")
    A_t = Input(batch_shape=(1, 3, 3), name="A")

    # Note we add a batch dimension of 1 to model inputs
    adj = G.to_adjacency_matrix().toarray()[None, :, :]
    x = features[None, :, :]

    out = GraphConvolution(2)([x_t, A_t])
    model = keras.Model(inputs=[x_t, A_t], outputs=out)
    preds = model.predict([x, adj], batch_size=1)
    assert preds.shape == (1, 3, 2)

    # batch dimension > 1 should work with a dense matrix
    x_t = Input(batch_shape=(10,) + features.shape)
    A_t = Input(batch_shape=(10, 3, 3))
    input_data = [np.broadcast_to(x, x_t.shape), np.broadcast_to(adj, A_t.shape)]

    out = GraphConvolution(2)([x_t, A_t])
    model = keras.Model(inputs=[x_t, A_t], outputs=out)

    preds = model.predict(input_data, batch_size=10)
    assert preds.shape == (10, 3, 2)
    for i in range(1, 10):
        # every batch element had the same input data, so the predictions should all be identical
        np.testing.assert_array_equal(preds[i, ...], preds[0, ...])


def test_GraphConvolution_sparse():
    G, features = create_graph_features()
    n_nodes = features.shape[0]

    # We need to specify the batch shape as one for the GraphConvolutional logic to work
    x_t = Input(batch_shape=(1,) + features.shape)
    A_ind = Input(batch_shape=(1, None, 2), dtype="int64")
    A_val = Input(batch_shape=(1, None), dtype="float32")

    A_mat = SqueezedSparseConversion(shape=(n_nodes, n_nodes), dtype=A_val.dtype)(
        [A_ind, A_val]
    )
    out = GraphConvolution(2)([x_t, A_mat])

    # Note we add a batch dimension of 1 to model inputs
    adj = G.to_adjacency_matrix().tocoo()
    A_indices = np.expand_dims(
        np.hstack((adj.row[:, None], adj.col[:, None])).astype(np.int64), 0
    )
    A_values = np.expand_dims(adj.data, 0)
    out_indices = np.array([[0, 1]], dtype="int32")
    x = features[None, :, :]

    model = keras.Model(inputs=[x_t, A_ind, A_val], outputs=out)
    preds = model.predict([x, A_indices, A_values], batch_size=1)
    assert preds.shape == (1, 3, 2)

    x_t_10 = Input(batch_shape=(10,) + features.shape)
    with pytest.raises(
        ValueError,
        match="features: expected batch dimension = 1 .* found features batch dimension 10",
    ):
        GraphConvolution(2)([x_t_10, A_mat])

    A_mat = keras.layers.Lambda(lambda x: tf.sparse.expand_dims(x, axis=0))(A_mat)
    with pytest.raises(
        ValueError,
        match="adjacency: expected a single adjacency matrix .* found adjacency tensor of rank 3",
    ):
        GraphConvolution(2)([x_t, A_mat])


def test_GCN_init():
    G, _ = create_graph_features()

    generator = FullBatchNodeGenerator(G)
    gcnModel = GCN([2], generator, activations=["relu"], dropout=0.5)

    assert gcnModel.layer_sizes == [2]
    assert gcnModel.activations == ["relu"]
    assert gcnModel.dropout == 0.5


def test_GCN_apply_dense():
    G, features = create_graph_features()
    adj = G.to_adjacency_matrix().toarray()[None, :, :]
    n_nodes = features.shape[0]

    generator = FullBatchNodeGenerator(G, sparse=False, method="none")
    gcnModel = GCN([2], generator, activations=["relu"], dropout=0.5)

    x_in, x_out = gcnModel.in_out_tensors()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices, adj])
    assert preds_1.shape == (1, 2, 2)

    # Check fit method
    preds_2 = model.predict(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)


def test_GCN_apply_sparse():

    G, features = create_graph_features()
    adj = G.to_adjacency_matrix()
    features, adj = GCN_Aadj_feats_op(features, adj)
    adj = adj.tocoo()
    A_indices = np.expand_dims(
        np.hstack((adj.row[:, None], adj.col[:, None])).astype(np.int64), 0
    )
    A_values = np.expand_dims(adj.data, 0)

    generator = FullBatchNodeGenerator(G, sparse=True, method="gcn")
    gcnModel = GCN(
        layer_sizes=[2], activations=["relu"], generator=generator, dropout=0.5
    )

    x_in, x_out = gcnModel.in_out_tensors()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices, A_indices, A_values])
    assert preds_1.shape == (1, 2, 2)

    # Check fit method
    preds_2 = model.predict(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)


def test_GCN_linkmodel_apply_dense():
    G, features = create_graph_features()
    adj = G.to_adjacency_matrix().toarray()[None, :, :]
    n_nodes = features.shape[0]

    generator = FullBatchLinkGenerator(G, sparse=False, method="none")
    gcnModel = GCN([3], generator, activations=["relu"], dropout=0.5)

    x_in, x_out = gcnModel.in_out_tensors()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[[0, 1], [1, 2]]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices, adj])
    assert preds_1.shape == (1, 2, 2, 3)

    # Check fit method
    preds_2 = model.predict(generator.flow([("a", "b"), ("b", "c")]))
    assert preds_2.shape == (1, 2, 2, 3)

    assert preds_1 == pytest.approx(preds_2)


def test_GCN_linkmodel_apply_sparse():

    G, features = create_graph_features()
    adj = G.to_adjacency_matrix()
    features, adj = GCN_Aadj_feats_op(features, adj)
    adj = adj.tocoo()
    A_indices = np.expand_dims(
        np.hstack((adj.row[:, None], adj.col[:, None])).astype(np.int64), 0
    )
    A_values = np.expand_dims(adj.data, 0)

    generator = FullBatchLinkGenerator(G, sparse=True, method="gcn")
    gcnModel = GCN(
        layer_sizes=[3], activations=["relu"], generator=generator, dropout=0.5
    )

    x_in, x_out = gcnModel.in_out_tensors()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[[0, 1], [1, 2]]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices, A_indices, A_values])
    assert preds_1.shape == (1, 2, 2, 3)

    # Check fit method
    preds_2 = model.predict(generator.flow([("a", "b"), ("b", "c")]))
    assert preds_2.shape == (1, 2, 2, 3)

    assert preds_1 == pytest.approx(preds_2)


def test_GCN_activations():
    G, features = create_graph_features()
    adj = G.to_adjacency_matrix().toarray()[None, :, :]
    n_nodes = features.shape[0]

    generator = FullBatchNodeGenerator(G, sparse=False, method="none")

    gcn = GCN([2], generator)
    assert gcn.activations == ["relu"]

    gcn = GCN([2, 2], generator)
    assert gcn.activations == ["relu", "relu"]

    gcn = GCN([2], generator, activations=["linear"])
    assert gcn.activations == ["linear"]

    with pytest.raises(ValueError):
        # More regularisers than layers
        gcn = GCN([2], generator, activations=["relu", "linear"])

    with pytest.raises(ValueError):
        # Fewer regularisers than layers
        gcn = GCN([2, 2], generator, activations=["relu"])

    with pytest.raises(ValueError):
        # Unknown regularisers
        gcn = GCN([2], generator, activations=["bleach"])


def test_GCN_regularisers():
    G, features = create_graph_features()
    adj = G.to_adjacency_matrix().toarray()[None, :, :]
    n_nodes = features.shape[0]

    generator = FullBatchNodeGenerator(G, sparse=False, method="none")

    gcn = GCN([2], generator)

    gcn = GCN([2], generator, kernel_initializer="ones")

    gcn = GCN([2], generator, kernel_initializer=initializers.ones())

    with pytest.raises(ValueError):
        gcn = GCN([2], generator, kernel_initializer="fred")

    gcn = GCN([2], generator, bias_initializer="zeros")

    gcn = GCN([2], generator, bias_initializer=initializers.zeros())

    with pytest.raises(ValueError):
        gcn = GCN([2], generator, bias_initializer="barney")


def test_kernel_and_bias_defaults():
    graph, _ = create_graph_features()
    generator = FullBatchNodeGenerator(graph, sparse=False, method="none")
    gcn = GCN([2, 2], generator)

    for layer in gcn._layers:
        if isinstance(layer, GraphConvolution):
            assert isinstance(layer.kernel_initializer, tf.initializers.GlorotUniform)
            assert isinstance(layer.bias_initializer, tf.initializers.Zeros)
            assert layer.kernel_regularizer is None
            assert layer.bias_regularizer is None
            assert layer.kernel_constraint is None
            assert layer.bias_constraint is None


@pytest.mark.parametrize("sparse", [False, True])
def test_gcn_save_load(tmpdir, sparse):
    G, _ = create_graph_features()
    generator = FullBatchNodeGenerator(G, sparse=sparse)
    gcn = GCN([2, 3], generator)
    test_utils.model_save_load(tmpdir, gcn)

# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Data61, CSIRO
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

from stellargraph.layer.appnp import *
from stellargraph.mapper import FullBatchNodeGenerator, FullBatchLinkGenerator
from stellargraph import StellarGraph
from stellargraph.core.utils import GCN_Aadj_feats_op

import networkx as nx
import pandas as pd
import numpy as np
from tensorflow import keras
import pytest
from ..test_utils.graphs import create_graph_features
from .. import test_utils


def test_APPNP_edge_cases():
    G, features = create_graph_features()

    generator = FullBatchNodeGenerator(G, sparse=False, method="gcn")

    try:
        appnpModel = APPNP(
            [2, 2], generator=generator, activations=["relu"], dropout=0.5
        )
    except ValueError as e:
        error = e
    assert str(error) == "The number of layers should equal the number of activations"

    try:
        appnpModel = APPNP([2], generator=[0, 1], activations=["relu"], dropout=0.5)
    except TypeError as e:
        error = e
    assert (
        str(error) == f"Generator should be a instance of FullBatchNodeGenerator, "
        f"FullBatchLinkGenerator or ClusterNodeGenerator"
    )

    try:
        appnpModel = APPNP(
            [2], generator=generator, activations=["relu"], dropout=0.0, approx_iter=-1
        )
    except ValueError as e:
        error = e
    assert str(error) == "approx_iter should be a positive integer"

    try:
        appnpModel = APPNP(
            [2], generator=generator, activations=["relu"], dropout=0.0, approx_iter=1.2
        )
    except ValueError as e:
        error = e
    assert str(error) == "approx_iter should be a positive integer"

    try:
        appnpModel = APPNP(
            [2],
            generator=generator,
            activations=["relu"],
            dropout=0.0,
            teleport_probability=1.2,
        )
    except ValueError as e:
        error = e
    assert str(error) == "teleport_probability should be between 0 and 1 (inclusive)"

    try:
        appnpModel = APPNP(
            [2],
            generator=generator,
            activations=["relu"],
            dropout=0.0,
            teleport_probability=1.2,
        )
    except ValueError as e:
        error = e
    assert str(error) == "teleport_probability should be between 0 and 1 (inclusive)"


def test_APPNP_apply_dense():
    G, features = create_graph_features()
    adj = G.to_adjacency_matrix()
    features, adj = GCN_Aadj_feats_op(features, adj)
    adj = np.array(adj.todense()[None, :, :])

    generator = FullBatchNodeGenerator(G, sparse=False, method="gcn")
    appnpModel = APPNP([2], generator=generator, activations=["relu"], dropout=0.5)

    x_in, x_out = appnpModel.in_out_tensors()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices, adj])
    assert preds_1.shape == (1, 2, 2)

    # Check fit method
    preds_2 = model.predict(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)


def test_APPNP_apply_sparse():

    G, features = create_graph_features()
    adj = G.to_adjacency_matrix()
    features, adj = GCN_Aadj_feats_op(features, adj)
    adj = adj.tocoo()
    A_indices = np.expand_dims(
        np.hstack((adj.row[:, None], adj.col[:, None])).astype(np.int64), 0
    )
    A_values = np.expand_dims(adj.data, 0)

    generator = FullBatchNodeGenerator(G, sparse=True, method="gcn")
    appnpnModel = APPNP([2], generator=generator, activations=["relu"], dropout=0.5)

    x_in, x_out = appnpnModel.in_out_tensors()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices, A_indices, A_values])
    assert preds_1.shape == (1, 2, 2)

    # Check fit method
    preds_2 = model.predict(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)


def test_APPNP_linkmodel_apply_dense():
    G, features = create_graph_features()
    adj = G.to_adjacency_matrix()
    adj = np.array(adj.todense()[None, :, :])

    generator = FullBatchLinkGenerator(G, sparse=False, method="none")
    appnpnModel = APPNP([3], generator, activations=["relu"], dropout=0.5)

    x_in, x_out = appnpnModel.in_out_tensors()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[[0, 1], [1, 2]]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices, adj])
    assert preds_1.shape == (1, 2, 2, 3)

    # Check fit method
    preds_2 = model.predict(generator.flow([("a", "b"), ("b", "c")]))
    assert preds_2.shape == (1, 2, 2, 3)

    assert preds_1 == pytest.approx(preds_2)


def test_APPNP_linkmodel_apply_sparse():

    G, features = create_graph_features()
    adj = G.to_adjacency_matrix()
    features, adj = GCN_Aadj_feats_op(features, adj)
    adj = adj.tocoo()
    A_indices = np.expand_dims(
        np.hstack((adj.row[:, None], adj.col[:, None])).astype(np.int64), 0
    )
    A_values = np.expand_dims(adj.data, 0)

    generator = FullBatchLinkGenerator(G, sparse=True, method="gcn")
    appnpnModel = APPNP(
        layer_sizes=[3], activations=["relu"], generator=generator, dropout=0.5
    )

    x_in, x_out = appnpnModel.in_out_tensors()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[[0, 1], [1, 2]]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices, A_indices, A_values])
    assert preds_1.shape == (1, 2, 2, 3)

    # Check fit method
    preds_2 = model.predict(generator.flow([("a", "b"), ("b", "c")]))
    assert preds_2.shape == (1, 2, 2, 3)

    assert preds_1 == pytest.approx(preds_2)


def test_APPNP_apply_propagate_model_dense():
    G, features = create_graph_features()
    adj = G.to_adjacency_matrix()
    features, adj = GCN_Aadj_feats_op(features, adj)
    adj = np.array(adj.todense()[None, :, :])

    generator = FullBatchNodeGenerator(G, sparse=False, method="gcn")
    appnpnModel = APPNP([2], generator=generator, activations=["relu"], dropout=0.5)

    fully_connected_model = keras.Sequential()
    fully_connected_model.add(Dense(2))

    x_in, x_out = appnpnModel.propagate_model(fully_connected_model)
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices, adj])
    assert preds_1.shape == (1, 2, 2)

    # Check fit method
    preds_2 = model.predict(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)


@pytest.mark.parametrize("model_type", ["sequential", "model"])
def test_APPNP_propagate_model_matches_manual(model_type):
    dense_size = 5

    G, features = create_graph_features()
    adj = G.to_adjacency_matrix()
    features, adj = GCN_Aadj_feats_op(features, adj)
    adj = np.array(adj.todense()[None, :, :])
    out_indices = np.array([[0, 1]], dtype="int32")

    generator = FullBatchNodeGenerator(G, sparse=False, method="gcn")
    appnpnModel = APPNP(
        [dense_size], generator=generator, activations=["relu"], dropout=0.0
    )

    dense = Dense(dense_size)
    if model_type == "sequential":
        fully_connected_model = keras.Sequential()
        fully_connected_model.add(dense)
    else:
        inp = keras.Input(shape=features.shape)
        fully_connected_model = keras.Model(inp, dense(inp))

    x_in, x_out = appnpnModel.propagate_model(fully_connected_model)
    end_to_end_model = keras.Model(inputs=x_in, outputs=x_out)
    preds_1 = end_to_end_model.predict([features[None, :, :], out_indices, adj])

    # run the process manually: transform the features, and then propagate
    float_feats = features[None, :, :].astype("float32")
    manual_preds = manual_inp = fully_connected_model.predict(float_feats)
    propagate = APPNPPropagationLayer(dense_size, teleport_probability=0.1)
    for _ in range(10):
        manual_preds = propagate([manual_preds, manual_inp, adj])

    # select the relevant pieces
    manual_preds = manual_preds.numpy()[:, out_indices.ravel(), :]

    np.testing.assert_allclose(preds_1, manual_preds)


def test_APPNP_apply_propagate_model_sparse():

    G, features = create_graph_features()
    adj = G.to_adjacency_matrix()
    features, adj = GCN_Aadj_feats_op(features, adj)
    adj = adj.tocoo()
    A_indices = np.expand_dims(
        np.hstack((adj.row[:, None], adj.col[:, None])).astype(np.int64), 0
    )
    A_values = np.expand_dims(adj.data, 0)

    generator = FullBatchNodeGenerator(G, sparse=True, method="gcn")
    appnpnModel = APPNP([2], generator=generator, activations=["relu"], dropout=0.5)

    fully_connected_model = keras.Sequential()
    fully_connected_model.add(Dense(2))

    x_in, x_out = appnpnModel.propagate_model(fully_connected_model)
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices, A_indices, A_values])
    assert preds_1.shape == (1, 2, 2)

    # Check fit method
    preds_2 = model.predict(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)


@pytest.mark.parametrize(
    "sparse", [False, True],
)
def test_APPNP_save_load(tmpdir, sparse):
    G, _ = create_graph_features()
    generator = FullBatchNodeGenerator(G, sparse=sparse)
    appnp = APPNP([2, 3], generator, ["relu", "relu"])
    test_utils.model_save_load(tmpdir, appnp)

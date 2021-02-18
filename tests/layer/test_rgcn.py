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

import numpy as np
from stellargraph.layer.rgcn import RelationalGraphConvolution, RGCN
from stellargraph.mapper.full_batch_generators import RelationalFullBatchNodeGenerator
import pytest
from scipy import sparse as sps
from stellargraph.core.utils import normalize_adj
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from stellargraph import StellarDiGraph, StellarGraph
from stellargraph.layer.misc import SqueezedSparseConversion
import pandas as pd
from ..test_utils.graphs import (
    relational_create_graph_features as create_graph_features,
)
from .. import test_utils


def test_RelationalGraphConvolution_config():
    rgcn_layer = RelationalGraphConvolution(units=16, num_relationships=5)
    conf = rgcn_layer.get_config()

    assert conf["units"] == 16
    assert conf["activation"] == "linear"
    assert conf["num_bases"] == 0
    assert conf["num_relationships"] == 5
    assert conf["use_bias"] == True

    assert conf["kernel_initializer"]["class_name"] == "GlorotUniform"
    assert conf["bias_initializer"]["class_name"] == "Zeros"
    assert conf["basis_initializer"]["class_name"] == "GlorotUniform"
    assert conf["coefficient_initializer"]["class_name"] == "GlorotUniform"

    assert conf["kernel_regularizer"] is None
    assert conf["bias_regularizer"] is None
    assert conf["basis_regularizer"] is None
    assert conf["coefficient_regularizer"] is None

    assert conf["kernel_constraint"] is None
    assert conf["bias_constraint"] is None
    assert conf["basis_constraint"] is None
    assert conf["coefficient_constraint"] is None


def test_RelationalGraphConvolution_init():
    rgcn_layer = RelationalGraphConvolution(
        units=16, num_relationships=5, num_bases=0, activation="relu"
    )

    assert rgcn_layer.units == 16
    assert rgcn_layer.use_bias is True
    assert rgcn_layer.num_bases == 0
    assert rgcn_layer.get_config()["activation"] == "relu"

    rgcn_layer = RelationalGraphConvolution(
        units=16, num_relationships=5, num_bases=10, activation="relu"
    )

    assert rgcn_layer.units == 16
    assert rgcn_layer.use_bias is True
    assert rgcn_layer.num_bases == 10
    assert rgcn_layer.get_config()["activation"] == "relu"


def test_RelationalGraphConvolution_sparse():
    G, features = create_graph_features()
    n_edge_types = len(G.edge_types)

    # We need to specify the batch shape as one for the GraphConvolutional logic to work
    n_nodes = features.shape[0]
    n_feat = features.shape[1]

    # Inputs for features
    x_t = Input(batch_shape=(1, n_nodes, n_feat))

    # Create inputs for sparse or dense matrices

    # Placeholders for the sparse adjacency matrix
    As_indices = [
        Input(batch_shape=(1, None, 2), dtype="int64") for i in range(n_edge_types)
    ]
    As_values = [Input(batch_shape=(1, None)) for i in range(n_edge_types)]
    A_placeholders = As_indices + As_values

    Ainput = [
        SqueezedSparseConversion(shape=(n_nodes, n_nodes), dtype=As_values[i].dtype)(
            [As_indices[i], As_values[i]]
        )
        for i in range(n_edge_types)
    ]

    x_inp_model = [x_t] + A_placeholders
    x_inp_conv = [x_t] + Ainput

    out = RelationalGraphConvolution(2, num_relationships=n_edge_types)(x_inp_conv)

    # Note we add a batch dimension of 1 to model inputs
    As = [A.tocoo() for A in get_As(G)]

    A_indices = [
        np.expand_dims(np.hstack((A.row[:, None], A.col[:, None])).astype(np.int64), 0)
        for A in As
    ]
    A_values = [np.expand_dims(A.data, 0) for A in As]

    out_indices = np.array([[0, 1]], dtype="int32")
    x = features[None, :, :]

    model = keras.Model(inputs=x_inp_model, outputs=out)
    preds = model.predict([x] + A_indices + A_values, batch_size=1)
    assert preds.shape == (1, 3, 2)


def test_RelationalGraphConvolution_dense():

    G, features = create_graph_features()
    n_edge_types = len(G.edge_types)

    # We need to specify the batch shape as one for the GraphConvolutional logic to work
    n_nodes = features.shape[0]
    n_feat = features.shape[1]

    # Inputs for features & target indices
    x_t = Input(batch_shape=(1, n_nodes, n_feat))
    out_indices_t = Input(batch_shape=(1, None), dtype="int32")

    # Create inputs for sparse or dense matrices

    # Placeholders for the sparse adjacency matrix
    A_placeholders = [
        Input(batch_shape=(1, n_nodes, n_nodes)) for _ in range(n_edge_types)
    ]

    A_in = [Lambda(lambda A: K.squeeze(A, 0))(A_p) for A_p in A_placeholders]

    x_inp_model = [x_t] + A_placeholders
    x_inp_conv = [x_t] + A_in

    out = RelationalGraphConvolution(2, num_relationships=n_edge_types)(x_inp_conv)

    As = [np.expand_dims(A.todense(), 0) for A in get_As(G)]

    out_indices = np.array([[0, 1]], dtype="int32")
    x = features[None, :, :]

    model = keras.Model(inputs=x_inp_model, outputs=out)
    preds = model.predict([x] + As, batch_size=1)
    assert preds.shape == (1, 3, 2)


def test_RGCN_init():
    G, features = create_graph_features()

    generator = RelationalFullBatchNodeGenerator(G)
    rgcnModel = RGCN([2], generator, num_bases=10, activations=["relu"], dropout=0.5)

    assert rgcnModel.layer_sizes == [2]
    assert rgcnModel.activations == ["relu"]
    assert rgcnModel.dropout == 0.5
    assert rgcnModel.num_bases == 10


def test_RGCN_apply_sparse():
    G, features = create_graph_features(is_directed=True)

    As = get_As(G)
    As = [A.tocoo() for A in As]
    A_indices = [
        np.expand_dims(np.hstack((A.row[:, None], A.col[:, None])).astype(np.int64), 0)
        for A in As
    ]
    A_values = [np.expand_dims(A.data, 0) for A in As]

    generator = RelationalFullBatchNodeGenerator(G, sparse=True)
    rgcnModel = RGCN([2], generator, num_bases=10, activations=["relu"], dropout=0.5)

    x_in, x_out = rgcnModel.in_out_tensors()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices] + A_indices + A_values)
    assert preds_1.shape == (1, 2, 2)

    # Check fit method
    preds_2 = model.predict(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)


def test_RGCN_apply_dense():
    G, features = create_graph_features(is_directed=True)

    As = get_As(G)
    As = [np.expand_dims(A.todense(), 0) for A in As]

    generator = RelationalFullBatchNodeGenerator(G, sparse=False)
    rgcnModel = RGCN([2], generator, num_bases=10, activations=["relu"], dropout=0.5)

    x_in, x_out = rgcnModel.in_out_tensors()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices] + As)
    assert preds_1.shape == (1, 2, 2)

    # Check fit method
    preds_2 = model.predict(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)


def test_RGCN_apply_sparse_directed():
    G, features = create_graph_features(is_directed=True)

    As = get_As(G)
    As = [A.tocoo() for A in As]

    A_indices = [
        np.expand_dims(np.hstack((A.row[:, None], A.col[:, None])).astype(np.int64), 0)
        for A in As
    ]
    A_values = [np.expand_dims(A.data, 0) for A in As]

    generator = RelationalFullBatchNodeGenerator(G, sparse=True)
    rgcnModel = RGCN([2], generator, num_bases=10, activations=["relu"], dropout=0.5)

    x_in, x_out = rgcnModel.in_out_tensors()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices] + A_indices + A_values)
    assert preds_1.shape == (1, 2, 2)

    # Check fit method
    preds_2 = model.predict(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)


def test_RGCN_apply_dense_directed():
    G, features = create_graph_features(is_directed=True)

    As = get_As(G)
    As = [np.expand_dims(A.todense(), 0) for A in As]

    generator = RelationalFullBatchNodeGenerator(G, sparse=False)
    rgcnModel = RGCN([2], generator, num_bases=10, activations=["relu"], dropout=0.5)
    x_in, x_out = rgcnModel.in_out_tensors()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices] + As)
    assert preds_1.shape == (1, 2, 2)

    # Check fit method
    preds_2 = model.predict(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)


def test_RelationalGraphConvolution_edge_cases():

    try:
        rgcn_layer = RelationalGraphConvolution(
            units=16, num_relationships=5, num_bases=0.5, activation="relu"
        )
    except TypeError as e:
        error = e
    assert str(error) == "num_bases should be an int"

    rgcn_layer = RelationalGraphConvolution(
        units=16, num_relationships=5, num_bases=-1, activation="relu"
    )
    rgcn_layer.build(input_shapes=[(1,)])
    assert rgcn_layer.bases is None

    try:
        rgcn_layer = RelationalGraphConvolution(
            units=16, num_relationships=0.5, num_bases=2, activation="relu"
        )
    except TypeError as e:
        error = e
    assert str(error) == "num_relationships should be an int"

    try:
        rgcn_layer = RelationalGraphConvolution(
            units=16, num_relationships=-1, num_bases=2, activation="relu"
        )
    except ValueError as e:
        error = e
    assert str(error) == "num_relationships should be positive"

    try:
        rgcn_layer = RelationalGraphConvolution(
            units=0.5, num_relationships=1, num_bases=2, activation="relu"
        )
    except TypeError as e:
        error = e
    assert str(error) == "units should be an int"

    try:
        rgcn_layer = RelationalGraphConvolution(
            units=-16, num_relationships=1, num_bases=2, activation="relu"
        )
    except ValueError as e:
        error = e
    assert str(error) == "units should be positive"


def get_As(G):

    As = []
    node_list = list(G.nodes())
    node_index = dict(zip(node_list, range(len(node_list))))

    for edge_type in G.edge_types:
        col_index = [
            node_index[n1]
            for n1, n2, etype in G.edges(include_edge_type=True)
            if etype == edge_type
        ]
        row_index = [
            node_index[n2]
            for n1, n2, etype in G.edges(include_edge_type=True)
            if etype == edge_type
        ]
        data = np.ones(len(col_index), np.float64)

        A = sps.coo_matrix(
            (data, (row_index, col_index)), shape=(len(node_list), len(node_list))
        )

        d = sps.diags(np.float_power(np.array(A.sum(1)) + 1e-9, -1).flatten(), 0)
        A = d.dot(A).tocsr()
        As.append(A)

    return As


def test_kernel_and_bias_defaults():
    graph, _ = create_graph_features()
    generator = RelationalFullBatchNodeGenerator(graph, sparse=False)
    rgcn = RGCN([2, 2], generator, num_bases=10)
    for layer in rgcn._layers:
        if isinstance(layer, RelationalGraphConvolution):
            assert isinstance(layer.kernel_initializer, tf.initializers.GlorotUniform)
            assert isinstance(layer.bias_initializer, tf.initializers.Zeros)
            assert layer.kernel_regularizer is None
            assert layer.bias_regularizer is None
            assert layer.kernel_constraint is None
            assert layer.bias_constraint is None


@pytest.mark.parametrize("num_bases", [0, 10])
@pytest.mark.parametrize("sparse", [False, True])
def test_RGCN_save_load(tmpdir, num_bases, sparse):
    graph, _ = create_graph_features()
    generator = RelationalFullBatchNodeGenerator(graph, sparse=sparse)
    rgcn = RGCN([2, 2], generator, num_bases=num_bases)
    test_utils.model_save_load(tmpdir, rgcn)

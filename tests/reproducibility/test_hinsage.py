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


import numpy as np
import pytest
import random
import tensorflow as tf
from stellargraph.mapper.sampled_node_generators import HinSAGENodeGenerator
from stellargraph.mapper.sampled_link_generators import HinSAGELinkGenerator
from stellargraph.layer.hinsage import HinSAGE
from stellargraph.layer.link_inference import link_classification
from stellargraph.random import set_seed
from ..test_utils.graphs import example_hin_1
from .fixtures import assert_reproducible


def hs_nai_model(num_samples, generator, targets, optimizer, bias, dropout, normalize):
    layer_sizes = [50] * len(num_samples)
    hinsage = HinSAGE(
        layer_sizes=layer_sizes,
        generator=generator,
        bias=bias,
        dropout=dropout,
        normalize=normalize,
    )
    # Build the model and expose input and output sockets of hinsage, for node pair inputs:
    x_inp, x_out = hinsage.build()
    pred = tf.keras.layers.Dense(units=targets.shape[1], activation="softmax")(x_out)
    model = tf.keras.Model(inputs=x_inp, outputs=pred)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.categorical_crossentropy)

    return model


def hs_nai(g, head_node_type, targets, num_samples, batch_size, shuffle):
    epochs = 4
    bias = True
    dropout = 0.0
    normalize = "l2"
    optimizer = tf.optimizers.Adam(1e-3)
    seed = 0

    set_seed(seed)
    tf.random.set_seed(seed)
    if shuffle:
        random.seed(seed)

    nodes = list(g.nodes_of_type(head_node_type))
    generator = HinSAGENodeGenerator(g, batch_size, num_samples, head_node_type)
    train_gen = generator.flow(nodes, targets, shuffle=True)

    model = hs_nai_model(
        num_samples, generator, targets, optimizer, bias, dropout, normalize
    )

    model.fit_generator(
        train_gen, epochs=epochs, use_multiprocessing=False, workers=4, shuffle=shuffle,
    )
    return model


def hs_link_pred_model(num_samples, generator, optimizer, bias, dropout, normalize):
    layer_sizes = [50] * len(num_samples)
    hinsage = HinSAGE(
        layer_sizes=layer_sizes,
        generator=generator,
        bias=bias,
        dropout=dropout,
        normalize=normalize,
    )
    # Build the model and expose input and output sockets of hinsage, for node pair inputs:
    x_inp, x_out = hinsage.build()
    pred = link_classification(
        output_dim=1, output_act="relu", edge_embedding_method="ip"
    )(x_out)

    model = tf.keras.Model(inputs=x_inp, outputs=pred)

    model.compile(
        optimizer=optimizer, loss=tf.keras.losses.binary_crossentropy,
    )

    return model


def hs_link_prediction(
    g, head_node_types, edge_ids, edge_labels, num_samples, batch_size, shuffle
):
    epochs = 4
    bias = True
    dropout = 0.0
    normalize = "l2"
    optimizer = tf.optimizers.Adam(1e-3)
    seed = 0

    set_seed(seed)
    tf.random.set_seed(seed)
    if shuffle:
        random.seed(seed)

    generator = HinSAGELinkGenerator(
        g, batch_size, num_samples, head_node_types=head_node_types
    )
    train_gen = generator.flow(edge_ids, edge_labels, shuffle=True)

    model = hs_link_pred_model(
        num_samples, generator, optimizer, bias, dropout, normalize
    )

    model.fit_generator(
        train_gen, epochs=epochs, use_multiprocessing=False, workers=4, shuffle=shuffle,
    )
    return model


@pytest.mark.parametrize("shuffle", [True, False])
def test_nai(shuffle):
    graph = example_hin_1(feature_sizes={"A": 2, "B": 3})
    head_node_type = "A"
    head_nodes = graph.nodes_of_type(head_node_type)

    target_size = 10
    targets = np.random.rand(len(head_nodes), target_size)
    assert_reproducible(
        lambda: hs_nai(
            graph, head_node_type, targets, [2, 2], batch_size=4, shuffle=shuffle
        )
    )


@pytest.mark.parametrize("shuffle", [True, False])
def test_link_prediction(shuffle):
    graph = example_hin_1(feature_sizes={"A": 2, "B": 3})
    num_examples = 10
    head_node_types = ["A", "B"]
    edge_ids = np.stack(
        [
            np.random.choice(graph.nodes_of_type(t), size=num_examples)
            for t in head_node_types
        ],
        axis=-1,
    )
    edge_labels = np.random.choice([0, 1], size=num_examples)
    assert_reproducible(
        lambda: hs_link_prediction(
            graph,
            head_node_types,
            edge_ids,
            edge_labels,
            [2, 2],
            batch_size=4,
            shuffle=shuffle,
        )
    )

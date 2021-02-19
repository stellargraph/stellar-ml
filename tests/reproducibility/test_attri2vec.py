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
from stellargraph.data.unsupervised_sampler import UnsupervisedSampler
from stellargraph.mapper.sampled_link_generators import Attri2VecLinkGenerator
from stellargraph.layer.attri2vec import Attri2Vec
from stellargraph.layer.link_inference import link_classification
from stellargraph.random import set_seed
from ..test_utils.graphs import petersen_graph
from .fixtures import assert_reproducible


def unsup_attri2vec_model(generator, optimizer, bias, normalize):
    layer_sizes = [50]
    attri2vec = Attri2Vec(
        layer_sizes=layer_sizes, generator=generator, bias=bias, normalize=normalize,
    )
    # Build the model and expose input and output sockets of attri2vec, for node pair inputs:
    x_inp, x_out = attri2vec.build()
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)
    model = tf.keras.Model(inputs=x_inp, outputs=prediction)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.binary_crossentropy)

    return model


def unsup_attri2vec(g, batch_size, shuffle):

    epochs = 4
    bias = True
    normalize = "l2"
    seed = 0
    optimizer = tf.optimizers.Adam(1e-3)
    number_of_walks = 1
    walk_length = 5

    set_seed(seed)
    tf.random.set_seed(seed)
    if shuffle:
        random.seed(seed)

    nodes = list(g.nodes())
    unsupervised_samples = UnsupervisedSampler(
        g, nodes=nodes, length=walk_length, number_of_walks=number_of_walks
    )
    generator = Attri2VecLinkGenerator(g, batch_size)
    train_gen = generator.flow(unsupervised_samples)

    model = unsup_attri2vec_model(generator, optimizer, bias, normalize)

    model.fit_generator(
        train_gen, epochs=epochs, use_multiprocessing=False, workers=4, shuffle=shuffle,
    )
    return model


def attri2vec_link_pred_model(generator, optimizer, bias, normalize):
    layer_sizes = [50]
    attri2vec = Attri2Vec(
        layer_sizes=layer_sizes, generator=generator, bias=bias, normalize=normalize,
    )
    # Build the model and expose input and output sockets of attri2vec, for node pair inputs:
    x_inp, x_out = attri2vec.build()
    pred = link_classification(
        output_dim=1, output_act="relu", edge_embedding_method="ip"
    )(x_out)

    model = tf.keras.Model(inputs=x_inp, outputs=pred)

    model.compile(
        optimizer=optimizer, loss=tf.keras.losses.binary_crossentropy,
    )

    return model


def attri2vec_link_prediction(g, edge_ids, edge_labels, batch_size, shuffle):
    epochs = 4
    bias = True
    normalize = "l2"
    seed = 0
    optimizer = tf.optimizers.Adam(1e-3)

    set_seed(seed)
    tf.random.set_seed(seed)
    if shuffle:
        random.seed(seed)

    generator = Attri2VecLinkGenerator(g, batch_size)
    train_gen = generator.flow(edge_ids, edge_labels, shuffle=True)

    model = attri2vec_link_pred_model(generator, optimizer, bias, normalize)

    model.fit_generator(
        train_gen, epochs=epochs, use_multiprocessing=False, workers=4, shuffle=shuffle,
    )
    return model


@pytest.mark.parametrize("shuffle", [True, False])
def test_unsup(petersen_graph, shuffle):
    assert_reproducible(
        lambda: unsup_attri2vec(petersen_graph, batch_size=4, shuffle=shuffle)
    )


# FIXME (#970): This test fails intermittently with shuffle=True
@pytest.mark.parametrize("shuffle", [False])
def test_link_prediction(petersen_graph, shuffle):
    num_examples = 10
    edge_ids = np.random.choice(petersen_graph.nodes(), size=(num_examples, 2))
    edge_labels = np.random.choice([0, 1], size=num_examples)
    assert_reproducible(
        lambda: attri2vec_link_prediction(
            petersen_graph, edge_ids, edge_labels, batch_size=4, shuffle=shuffle
        )
    )

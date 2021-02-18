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
GraphSAGE tests

"""
from tensorflow import keras
from tensorflow.keras import initializers, regularizers
import tensorflow as tf

import numpy as np
import pytest

from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer.graphsage import (
    GraphSAGE,
    MeanAggregator,
    MaxPoolingAggregator,
    MeanPoolingAggregator,
    AttentionalAggregator,
)
from ..test_utils.graphs import example_graph
from .. import test_utils


pytestmark = test_utils.ignore_stellargraph_experimental_mark


# Mean aggregator tests
def test_mean_agg_constructor():
    agg = MeanAggregator(2)
    assert agg.output_dim == 2
    assert not agg.has_bias

    # Check config
    config = agg.get_config()
    assert config["output_dim"] == 2
    assert config["bias"] is False
    assert config["act"] == "relu"


def test_mean_agg_constructor_1():
    agg = MeanAggregator(output_dim=4, bias=True, act=lambda x: x + 1)
    assert agg.output_dim == 4
    assert agg.has_bias
    assert agg.act(2) == 3


def test_mean_agg_apply():
    agg = MeanAggregator(5, bias=True, act=lambda x: x, kernel_initializer="ones")
    inp1 = keras.Input(shape=(1, 2))
    inp2 = keras.Input(shape=(1, 2, 2))
    out = agg([inp1, inp2])

    assert agg.weight_dims == [5, 5]

    model = keras.Model(inputs=[inp1, inp2], outputs=out)
    x1 = np.array([[[1, 1]]])
    x2 = np.array([[[[2, 2], [3, 3]]]])
    actual = model.predict([x1, x2])
    expected = np.array([[[7, 7, 7, 7, 7]]])
    assert expected == pytest.approx(actual)


def test_mean_agg_apply_groups():
    agg = MeanAggregator(11, bias=True, act=lambda x: x, kernel_initializer="ones")
    inp1 = keras.Input(shape=(1, 2))
    inp2 = keras.Input(shape=(1, 2, 2))
    inp3 = keras.Input(shape=(1, 2, 2))
    out = agg([inp1, inp2, inp3])

    assert agg.weight_dims == [11, 11, 11]

    model = keras.Model(inputs=[inp1, inp2, inp3], outputs=out)
    x1 = np.array([[[1, 1]]])
    x2 = np.array([[[[2, 2], [3, 3]]]])
    x3 = np.array([[[[5, 5], [4, 4]]]])

    actual = model.predict([x1, x2, x3])
    print(actual)

    expected = np.array([[[16] * 11]])
    assert expected == pytest.approx(actual)


def test_mean_agg_zero_neighbours():
    agg = MeanAggregator(4, bias=False, act=lambda x: x, kernel_initializer="ones")

    inp1 = keras.Input(shape=(1, 2))
    inp2 = keras.Input(shape=(1, 0, 2))

    out = agg([inp1, inp2])
    model = keras.Model(inputs=[inp1, inp2], outputs=out)

    x1 = np.array([[[1, 1]]])
    x2 = np.zeros((1, 1, 0, 2))

    actual = model.predict([x1, x2])
    expected = np.array([[[2, 2, 2, 2]]])
    assert expected == pytest.approx(actual)


# MaxPooling aggregator tests
def test_maxpool_agg_constructor():
    agg = MaxPoolingAggregator(2, bias=False)
    assert agg.output_dim == 2
    assert agg.hidden_dim == 2
    assert not agg.has_bias
    assert agg.act.__name__ == "relu"
    assert agg.hidden_act.__name__ == "relu"

    # Check config
    config = agg.get_config()
    assert config["output_dim"] == 2
    assert config["bias"] == False
    assert config["act"] == "relu"


def test_maxpool_agg_constructor_1():
    agg = MaxPoolingAggregator(output_dim=4, bias=True, act=lambda x: x + 1)
    assert agg.output_dim == 4
    assert agg.hidden_dim == 4
    assert agg.has_bias
    assert agg.act(2) == 3


def test_maxpool_agg_apply_hidden_bias():
    # Specifying bias_initializer="ones" initialises all bias terms to ones;
    # using bias=False turns of outer bias but retains hidden bias.
    agg = MaxPoolingAggregator(
        2, bias=False, act="linear", kernel_initializer="ones", bias_initializer="ones"
    )
    assert agg.get_config()["kernel_initializer"]["class_name"] == "Ones"
    assert agg.get_config()["bias_initializer"]["class_name"] == "Ones"

    # Self features
    inp1 = keras.Input(shape=(1, 2))
    # Neighbour features
    inp2 = keras.Input(shape=(1, 2, 2))
    out = agg([inp1, inp2])

    # Check sizes
    assert agg.weight_dims == [2, 2]

    # Numerical test values
    x1 = np.array([[[1, 1]]])
    x2 = np.array([[[[2, 2], [3, 3]]]])

    # Agg output:
    # neigh_agg = max(relu(x2 · ones(2x2)) + ones(2)), axis=1) = max([[5,5],[7,7]]) = [[7,7]]
    # from_self = K.dot(x1, ones) = [[2]]
    # from_neigh = K.dot(neigh_agg, ones) = [[14]]
    model = keras.Model(inputs=[inp1, inp2], outputs=out)
    actual = model.predict([x1, x2])
    expected = np.array([[[16, 16]]])

    assert expected == pytest.approx(actual)


def test_maxpool_agg_apply_no_bias():
    # By default, bias_initializers="zeros", so all bias terms are initialised to zeros.
    agg = MaxPoolingAggregator(2, act="linear", kernel_initializer="ones")
    assert agg.get_config()["kernel_initializer"]["class_name"] == "Ones"
    assert agg.get_config()["bias_initializer"]["class_name"] == "Zeros"

    # Self features
    inp1 = keras.Input(shape=(1, 2))
    # Neighbour features
    inp2 = keras.Input(shape=(1, 2, 2))
    out = agg([inp1, inp2])

    # Check sizes
    assert agg.weight_dims == [2, 2]

    # Numerical test values
    x1 = np.array([[[1, 1]]])
    x2 = np.array([[[[2, 2], [3, 3]]]])

    # Agg output:
    # neigh_agg = max(relu(x2 · ones(2x2)) + zeros(2)), axis=1) = max([[4,4],[6,6]]) = [[6,6]]
    # from_self = K.dot(x1, ones) = [[2]]
    # from_neigh = K.dot(neigh_agg, ones) = [[12]]
    model = keras.Model(inputs=[inp1, inp2], outputs=out)
    actual = model.predict([x1, x2])
    expected = np.array([[[14, 14]]])

    assert expected == pytest.approx(actual)


def test_maxpool_agg_zero_neighbours():
    agg = MaxPoolingAggregator(4, bias=False, act="linear", kernel_initializer="ones")

    inp1 = keras.Input(shape=(1, 2))
    inp2 = keras.Input(shape=(1, 0, 2))

    out = agg([inp1, inp2])
    model = keras.Model(inputs=[inp1, inp2], outputs=out)

    x1 = np.array([[[1, 1]]])
    x2 = np.zeros((1, 1, 0, 2))

    actual = model.predict([x1, x2])
    expected = np.array([[[2, 2, 2, 2]]])
    assert expected == pytest.approx(actual)


# MeanPooling aggregator tests
def test_meanpool_agg_constructor():
    agg = MeanPoolingAggregator(2, bias=False)
    assert agg.output_dim == 2
    assert agg.hidden_dim == 2
    assert not agg.has_bias
    assert agg.act.__name__ == "relu"
    assert agg.hidden_act.__name__ == "relu"

    # Check config
    config = agg.get_config()
    assert config["output_dim"] == 2
    assert config["bias"] is False
    assert config["act"] == "relu"


def test_meanpool_agg_constructor_1():
    agg = MeanPoolingAggregator(output_dim=4, bias=True, act=lambda x: x + 1)
    assert agg.output_dim == 4
    assert agg.hidden_dim == 4
    assert agg.has_bias
    assert agg.act(2) == 3


def test_meanpool_agg_apply_hidden_bias():
    # Specifying bias_initializer="ones" initialises all bias terms to ones;
    # using bias=False turns of outer bias but retains hidden bias.
    agg = MeanPoolingAggregator(
        2, bias=False, act="linear", kernel_initializer="ones", bias_initializer="ones"
    )
    assert agg.get_config()["kernel_initializer"]["class_name"] == "Ones"
    assert agg.get_config()["bias_initializer"]["class_name"] == "Ones"

    # Self features
    inp1 = keras.Input(shape=(1, 2))
    # Neighbour features
    inp2 = keras.Input(shape=(1, 2, 2))

    out = agg([inp1, inp2])

    # Check sizes
    assert agg.weight_dims == [2, 2]

    # Numerical test values
    x1 = np.array([[[1, 1]]])
    x2 = np.array([[[[2, 2], [3, 3]]]])

    # Agg output:
    # neigh_agg = mean(relu(x2 · ones(2x2) + ones(2)), axis=1)
    #   = mean([[5,5],[7,7]]) = [[6,6]]
    # from_self = K.dot(x1, ones) = [[2]]
    # from_neigh = K.dot(neigh_agg, ones(2x1)) = [[12]]

    model = keras.Model(inputs=[inp1, inp2], outputs=out)
    actual = model.predict([x1, x2])
    expected = np.array([[[14, 14]]])

    assert expected == pytest.approx(actual)


def test_meanpool_agg_apply_no_bias():
    # By default, bias_initializers="zeros", so all bias terms are initialised to zeros.
    agg = MeanPoolingAggregator(2, act="linear", kernel_initializer="ones")
    assert agg.get_config()["kernel_initializer"]["class_name"] == "Ones"
    assert agg.get_config()["bias_initializer"]["class_name"] == "Zeros"

    # Self features
    inp1 = keras.Input(shape=(1, 2))
    # Neighbour features
    inp2 = keras.Input(shape=(1, 2, 2))

    out = agg([inp1, inp2])

    # Check sizes
    assert agg.weight_dims == [2, 2]

    # Numerical test values
    x1 = np.array([[[1, 1]]])
    x2 = np.array([[[[2, 2], [3, 3]]]])

    # Agg output:
    # neigh_agg = mean(relu(x2 · ones(2x2) + zeros(2)), axis=1)
    #   = mean([[4,4],[6,6]]) = [[5,5]]
    # from_self = K.dot(x1, ones) = [[2]]
    # from_neigh = K.dot(neigh_agg, ones) = [[10]]

    model = keras.Model(inputs=[inp1, inp2], outputs=out)
    actual = model.predict([x1, x2])
    expected = np.array([[[12, 12]]])

    assert expected == pytest.approx(actual)


def test_meanpool_agg_zero_neighbours():
    agg = MeanPoolingAggregator(4, bias=False, act="linear", kernel_initializer="ones")

    inp1 = keras.Input(shape=(1, 2))
    inp2 = keras.Input(shape=(1, 0, 2))
    out = agg([inp1, inp2])

    # Now we have an input shape with a 0, the attention model switches to
    # a MLP and the first group will have non-zero output size.
    assert agg.weight_dims == [4, 0]

    model = keras.Model(inputs=[inp1, inp2], outputs=out)
    x1 = np.array([[[1, 1]]])
    x2 = np.zeros((1, 1, 0, 2))

    actual = model.predict([x1, x2])
    expected = np.array([[[2, 2, 2, 2]]])
    assert expected == pytest.approx(actual)


# Attentional aggregator tests
def test_attn_agg_constructor():
    agg = AttentionalAggregator(2, bias=False)
    assert agg.output_dim == 2
    assert not agg.has_bias
    assert agg.act.__name__ == "relu"
    # assert agg.attn_act.__name__ == "relu"

    # Check config
    config = agg.get_config()
    assert config["output_dim"] == 2
    assert config["bias"] is False
    assert config["act"] == "relu"


def test_attn_agg_constructor_1():
    agg = AttentionalAggregator(output_dim=4, bias=True, act=lambda x: x + 1)
    assert agg.output_dim == 4
    assert agg.has_bias
    assert agg.act(2) == 3


def test_attn_agg_apply():
    agg = AttentionalAggregator(2, bias=False, act="linear", kernel_initializer="ones")
    agg.attn_act = keras.activations.get("linear")

    # Self features
    inp1 = keras.Input(shape=(1, 2))
    # Neighbour features
    inp2 = keras.Input(shape=(1, 2, 2))
    out = agg([inp1, inp2])

    # The AttentionalAggregator implementation is a hack at the moment, it doesn't
    # assign any dimensions in the output to head-node features.
    assert agg.weight_dims == [0, 2]

    # Numerical test values
    x1 = np.array([[[1, 1]]])
    x2 = np.array([[[[2, 2], [3, 3]]]])

    # Agg output:
    # hs = relu(x1 · ones(2x2)) = [2,2]
    # hn = relu(x2 · ones(2x2)) =  [[2,2], [4,4], [6,6]]
    # attn_u = ones(2) · hs +  ones(2) · hn = [8, 12, 16]
    # attn = softmax(attn_u) = [3.3e-4, 1.8e-4, 9.81e-1]
    # hout =  attn · hn = [5.96, 5.96]
    model = keras.Model(inputs=[inp1, inp2], outputs=out)
    actual = model.predict([x1, x2])

    expected = np.array([[[5.963, 5.963]]])

    assert expected == pytest.approx(actual, rel=1e-4)


def test_attn_agg_zero_neighbours():
    agg = AttentionalAggregator(4, bias=False, act="linear", kernel_initializer="ones")

    inp1 = keras.Input(shape=(1, 2))
    inp2 = keras.Input(shape=(1, 0, 2))

    out = agg([inp1, inp2])
    model = keras.Model(inputs=[inp1, inp2], outputs=out)

    x1 = np.array([[[1, 1]]])
    x2 = np.zeros((1, 1, 0, 2))

    actual = model.predict([x1, x2])
    expected = np.array([[[2, 2, 2, 2]]])
    assert expected == pytest.approx(actual)


def test_graphsage_constructor():
    gs = GraphSAGE(
        layer_sizes=[4], n_samples=[2], input_dim=2, normalize="l2", multiplicity=1
    )
    assert gs.dims == [2, 4]
    assert gs.n_samples == [2]
    assert gs.max_hops == 1
    assert gs.bias
    assert len(gs._aggs) == 1

    # Check incorrect normalization flag
    with pytest.raises(ValueError):
        GraphSAGE(
            layer_sizes=[4],
            n_samples=[2],
            input_dim=2,
            normalize=lambda x: x,
            multiplicity=1,
        )

    with pytest.raises(ValueError):
        GraphSAGE(
            layer_sizes=[4],
            n_samples=[2],
            input_dim=2,
            normalize="unknown",
            multiplicity=1,
        )

    # Check requirement for generator or n_samples
    with pytest.raises(ValueError):
        GraphSAGE(layer_sizes=[4])

    # Construction from generator
    G = example_graph(feature_size=3)
    gen = GraphSAGENodeGenerator(G, batch_size=2, num_samples=[2, 2])
    gs = GraphSAGE(layer_sizes=[4, 8], generator=gen, bias=True)

    # The GraphSAGE should no longer accept a Sequence
    t_gen = gen.flow([1, 2])
    with pytest.raises(TypeError):
        gs = GraphSAGE(layer_sizes=[4, 8], generator=t_gen, bias=True)

    assert gs.dims == [3, 4, 8]
    assert gs.n_samples == [2, 2]
    assert gs.max_hops == 2
    assert gs.bias
    assert len(gs._aggs) == 2


def test_graphsage_constructor_passing_aggregator():
    gs = GraphSAGE(
        layer_sizes=[4],
        n_samples=[2],
        input_dim=2,
        multiplicity=1,
        aggregator=MeanAggregator,
    )
    assert gs.dims == [2, 4]
    assert gs.n_samples == [2]
    assert gs.max_hops == 1
    assert gs.bias
    assert len(gs._aggs) == 1

    with pytest.raises(TypeError):
        GraphSAGE(
            layer_sizes=[4], n_samples=[2], input_dim=2, multiplicity=1, aggregator=1
        )


def test_graphsage_constructor_1():
    gs = GraphSAGE(
        layer_sizes=[4, 6, 8],
        n_samples=[2, 4, 6],
        input_dim=2,
        multiplicity=1,
        bias=True,
        dropout=0.5,
    )
    assert gs.dims == [2, 4, 6, 8]
    assert gs.n_samples == [2, 4, 6]
    assert gs.max_hops == 3
    assert gs.bias
    assert len(gs._aggs) == 3


def test_graphsage_apply():
    gs = GraphSAGE(
        layer_sizes=[4],
        n_samples=[2],
        bias=False,
        input_dim=2,
        multiplicity=1,
        normalize=None,
        kernel_initializer="ones",
    )

    inp1 = keras.Input(shape=(1, 2))
    inp2 = keras.Input(shape=(2, 2))
    out = gs([inp1, inp2])
    model = keras.Model(inputs=[inp1, inp2], outputs=out)


def test_graphsage_apply_1():
    gs = GraphSAGE(
        layer_sizes=[2, 2, 2],
        n_samples=[2, 2, 2],
        bias=False,
        input_dim=2,
        multiplicity=1,
        normalize=None,
        kernel_initializer="ones",
    )

    inp = [keras.Input(shape=(i, 2)) for i in [1, 2, 4, 8]]
    out = gs(inp)
    model = keras.Model(inputs=inp, outputs=out)

    x = [
        np.array([[[1, 1]]]),
        np.array([[[2, 2], [2, 2]]]),
        np.array([[[3, 3], [3, 3], [3, 3], [3, 3]]]),
        np.array([[[4, 4], [4, 4], [4, 4], [4, 4], [5, 5], [5, 5], [5, 5], [5, 5]]]),
    ]
    expected = np.array([[164, 164]])

    actual = model.predict(x)
    assert expected == pytest.approx(actual)

    # Use the node model:
    xinp, xout = gs.in_out_tensors()
    model2 = keras.Model(inputs=xinp, outputs=xout)
    assert pytest.approx(expected) == model2.predict(x)


def test_graphsage_serialize():
    gs = GraphSAGE(
        layer_sizes=[4],
        n_samples=[2],
        bias=False,
        input_dim=2,
        multiplicity=1,
        normalize=None,
    )

    inp1 = keras.Input(shape=(1, 2))
    inp2 = keras.Input(shape=(2, 2))
    out = gs([inp1, inp2])
    model = keras.Model(inputs=[inp1, inp2], outputs=out)

    # Save model
    model_json = model.to_json()

    # Set all weights to one
    model_weights = [np.ones_like(w) for w in model.get_weights()]

    # Load model from json & set all weights
    model2 = keras.models.model_from_json(
        model_json, custom_objects={"MeanAggregator": MeanAggregator}
    )
    model2.set_weights(model_weights)

    # Test loaded model
    x1 = np.array([[[1, 1]]])
    x2 = np.array([[[2, 2], [3, 3]]])
    expected = np.array([[7, 7, 7, 7]])

    actual = model2.predict([x1, x2])
    assert expected == pytest.approx(actual)


def test_graphsage_zero_neighbours():
    gs = GraphSAGE(
        layer_sizes=[2, 2],
        n_samples=[0, 0],
        bias=False,
        input_dim=2,
        multiplicity=1,
        normalize="none",
        kernel_initializer="ones",
    )

    inp = [keras.Input(shape=(i, 2)) for i in [1, 0, 0]]
    out = gs(inp)
    model = keras.Model(inputs=inp, outputs=out)

    x = [np.array([[[1.5, 1]]]), np.zeros((1, 0, 2)), np.zeros((1, 0, 2))]

    actual = model.predict(x)
    expected = np.array([[5, 5]])
    assert actual == pytest.approx(expected)


def test_graphsage_passing_activations():
    gs = GraphSAGE(layer_sizes=[4], n_samples=[2], input_dim=2, multiplicity=1)
    assert gs.activations == ["linear"]

    gs = GraphSAGE(layer_sizes=[4, 4], n_samples=[2, 2], input_dim=2, multiplicity=1)
    assert gs.activations == ["relu", "linear"]

    gs = GraphSAGE(
        layer_sizes=[4, 4, 4], n_samples=[2, 2, 2], input_dim=2, multiplicity=1
    )
    assert gs.activations == ["relu", "relu", "linear"]

    with pytest.raises(ValueError):
        GraphSAGE(
            layer_sizes=[4, 4, 4],
            n_samples=[2, 2, 2],
            input_dim=2,
            multiplicity=1,
            activations=["relu"],
        )

    with pytest.raises(ValueError):
        GraphSAGE(
            layer_sizes=[4, 4, 4],
            n_samples=[2, 2, 2],
            input_dim=2,
            multiplicity=1,
            activations=["relu"] * 2,
        )

    with pytest.raises(ValueError):
        GraphSAGE(
            layer_sizes=[4, 4, 4],
            n_samples=[2, 2, 2],
            input_dim=2,
            multiplicity=1,
            activations=["fred", "wilma", "barney"],
        )

    gs = GraphSAGE(
        layer_sizes=[4, 4, 4],
        n_samples=[2, 2, 2],
        input_dim=2,
        multiplicity=1,
        activations=["linear"] * 3,
    )
    assert gs.activations == ["linear"] * 3


def test_graphsage_passing_regularisers():
    with pytest.raises(ValueError):
        GraphSAGE(
            layer_sizes=[4],
            n_samples=[2],
            input_dim=2,
            multiplicity=1,
            kernel_initializer="fred",
        )

    GraphSAGE(
        layer_sizes=[4],
        n_samples=[2],
        input_dim=2,
        multiplicity=1,
        kernel_initializer="ones",
    )

    GraphSAGE(
        layer_sizes=[4],
        n_samples=[2],
        input_dim=2,
        multiplicity=1,
        kernel_initializer=initializers.ones(),
    )

    GraphSAGE(
        layer_sizes=[4],
        n_samples=[2],
        input_dim=2,
        multiplicity=1,
        kernel_regularizer=regularizers.l2(0.01),
    )

    with pytest.raises(ValueError):
        GraphSAGE(
            layer_sizes=[4],
            n_samples=[2],
            input_dim=2,
            multiplicity=1,
            kernel_regularizer="wilma",
        )


def test_kernel_and_bias_defaults():
    gs = GraphSAGE(layer_sizes=[4, 4], n_samples=[2, 2], input_dim=2, multiplicity=1)
    for layer in gs._aggs:
        assert isinstance(layer.kernel_initializer, tf.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, tf.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None
        assert layer.kernel_constraint is None
        assert layer.bias_constraint is None


def test_graphsage_save_load(tmpdir):
    gs = GraphSAGE(layer_sizes=[4, 4], n_samples=[2, 2], input_dim=2, multiplicity=1)
    test_utils.model_save_load(tmpdir, gs)

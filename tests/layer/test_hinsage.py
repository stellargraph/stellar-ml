# -*- coding: utf-8 -*-
#
# Copyright 2018 Data61, CSIRO
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
HinSAGE tests

"""


from stellargraph.layer.hinsage import *
from tensorflow import keras
from tensorflow.keras import regularizers
import numpy as np
import pytest


def test_mean_hin_agg_constructor():
    agg = MeanHinAggregator(output_dim=2)
    assert agg.output_dim == 2
    assert agg.half_output_dim == 1
    assert not agg.has_bias
    assert agg.act.__name__ == "relu"


def test_mean_hin_agg_constructor_1():
    agg = MeanHinAggregator(output_dim=2, bias=True, act=lambda x: x + 1)
    assert agg.output_dim == 2
    assert agg.half_output_dim == 1
    assert agg.has_bias
    assert agg.act(2) == 3

    # Check config
    config = agg.get_config()
    assert config["output_dim"] == 2
    assert config["bias"] == True
    assert config["act"] == "<lambda>"

    agg = MeanHinAggregator(output_dim=2, bias=True, act="relu")
    assert agg.output_dim == 2
    assert agg.half_output_dim == 1
    assert agg.has_bias

    # Check config
    config = agg.get_config()
    assert config["output_dim"] == 2
    assert config["bias"] == True
    assert config["act"] == "relu"


def test_mean_hin_agg_apply():
    agg = MeanHinAggregator(2, act=lambda z: z, kernel_initializer="ones")
    inp = [
        keras.Input(shape=(1, 2)),
        keras.Input(shape=(1, 2, 2)),
        keras.Input(shape=(1, 2, 4)),
    ]
    out = agg(inp)
    model = keras.Model(inputs=inp, outputs=out)

    x = [
        np.array([[[1, 1]]]),
        np.array([[[[2, 2], [2, 2]]]]),
        np.array([[[[3, 3, 3, 3], [3, 3, 3, 3]]]]),
    ]

    actual = model.predict(x)
    expected = np.array([[[2, 8]]])
    assert actual == pytest.approx(expected)


def test_mean_hin_agg_apply_2():
    agg1 = MeanHinAggregator(2, act=lambda z: z, kernel_initializer="ones")
    agg2 = MeanHinAggregator(2, act=lambda z: z + 1, kernel_initializer="ones")

    inp = [
        keras.Input(shape=(1, 2)),
        keras.Input(shape=(1, 2, 2)),
        keras.Input(shape=(1, 2, 4)),
    ]
    out1 = agg1(inp, name="test")
    out2 = agg2(inp, name="test")

    model = keras.Model(inputs=inp, outputs=[out1, out2])

    x = [
        np.array([[[1, 1]]]),
        np.array([[[[2, 2], [2, 2]]]]),
        np.array([[[[3, 3, 3, 3], [3, 3, 3, 3]]]]),
    ]

    actual = model.predict(x)
    expected = [np.array([[[2, 8]]]), np.array([[[3, 9]]])]
    assert all(a == pytest.approx(e) for a, e in zip(actual, expected))


def test_mean_hin_zero_neighbours():
    agg = MeanHinAggregator(2, bias=False, act=lambda z: z, kernel_initializer="ones")
    inp = [
        keras.Input(shape=(1, 2)),
        keras.Input(shape=(1, 0, 2)),
        keras.Input(shape=(1, 0, 4)),
    ]
    out = agg(inp)

    # Check weights added only for first input
    assert all(w is None for w in agg.w_neigh)

    model = keras.Model(inputs=inp, outputs=out)

    x = [np.array([[[1, 1]]]), np.zeros((1, 1, 0, 2)), np.zeros((1, 1, 0, 4))]

    actual = model.predict(x)
    expected = np.array([[[2, 0]]])
    assert actual == pytest.approx(expected)


def test_hinsage_constructor():
    hs = HinSAGE(
        layer_sizes=[{"1": 2, "2": 2}, {"1": 2}],
        n_samples=[2, 2],
        input_neighbor_tree=[
            ("1", [1, 2]),
            ("1", [3, 4]),
            ("2", [5]),
            ("1", []),
            ("2", []),
            ("2", []),
        ],
        input_dim={"1": 2, "2": 2},
    )
    assert hs.n_layers == 2
    assert hs.n_samples == [2, 2]
    assert hs.bias

    hs = HinSAGE(
        layer_sizes=[2, 2],
        n_samples=[2, 2],
        input_neighbor_tree=[
            ("1", [1, 2]),
            ("1", [3, 4]),
            ("2", [5]),
            ("1", []),
            ("2", []),
            ("2", []),
        ],
        input_dim={"1": 2, "2": 2},
    )
    assert hs.n_layers == 2
    assert hs.n_samples == [2, 2]
    assert hs.bias


def test_hinsage_constructor_with_agg():
    hs = HinSAGE(
        layer_sizes=[{"1": 2, "2": 2}, {"1": 2}],
        n_samples=[2, 2],
        input_neighbor_tree=[
            ("1", [1, 2]),
            ("1", [3, 4]),
            ("2", [5]),
            ("1", []),
            ("2", []),
            ("2", []),
        ],
        input_dim={"1": 2, "2": 2},
        aggregator=MeanHinAggregator,
    )
    assert hs.n_layers == 2
    assert hs.n_samples == [2, 2]
    assert hs.bias


def test_hinsage_input_shapes():
    hs = HinSAGE(
        layer_sizes=[{"1": 2, "2": 2}, 2],
        n_samples=[2, 2],
        input_neighbor_tree=[
            ("1", [1, 2]),
            ("1", [3, 4]),
            ("2", [5]),
            ("1", []),
            ("2", []),
            ("2", []),
        ],
        input_dim={"1": 2, "2": 4},
    )
    assert hs._input_shapes() == [(1, 2), (2, 2), (2, 4), (4, 2), (4, 4), (4, 4)]


def test_hinsage_constructor_wrong_normalisation():
    with pytest.raises(ValueError):
        hs = HinSAGE(
            layer_sizes=[{"1": 2, "2": 2}, {"1": 2}],
            n_samples=[2, 2],
            input_neighbor_tree=[
                ("1", [1, 2]),
                ("1", [3, 4]),
                ("2", [5]),
                ("1", []),
                ("2", []),
                ("2", []),
            ],
            input_dim={"1": 2, "2": 2},
            normalize="unknown",
        )


def test_hinsage_apply():
    hs = HinSAGE(
        layer_sizes=[{"1": 2, "2": 2}, 2],
        n_samples=[2, 2],
        input_neighbor_tree=[
            ("1", [1, 2]),
            ("1", [3, 4]),
            ("2", [5]),
            ("1", []),
            ("2", []),
            ("2", []),
        ],
        input_dim={"1": 2, "2": 4},
        normalize="none",
        kernel_initializer="ones",
    )

    inp = [
        keras.Input(shape=(1, 2)),
        keras.Input(shape=(2, 2)),
        keras.Input(shape=(2, 4)),
        keras.Input(shape=(4, 2)),
        keras.Input(shape=(4, 4)),
        keras.Input(shape=(4, 4)),
    ]

    out = hs(inp)
    model = keras.Model(inputs=inp, outputs=out)

    x = [
        np.array([[[1, 1]]]),
        np.array([[[2, 2], [2, 2]]]),
        np.array([[[4, 4, 4, 4], [4, 4, 4, 4]]]),
        np.array([[[3, 3], [3, 3], [3, 3], [3, 3]]]),
        np.array([[[6, 6, 6, 6], [6, 6, 6, 6], [6, 6, 6, 6], [6, 6, 6, 6]]]),
        np.array([[[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]]]),
    ]

    actual = model.predict(x)
    expected = np.array([[12, 35.5]])
    assert actual == pytest.approx(expected)


def test_hinsage_default_model():
    hs = HinSAGE(
        layer_sizes=[2, 2],
        n_samples=[2, 2],
        input_neighbor_tree=[
            ("1", [1, 2]),
            ("1", [3, 4]),
            ("2", [5]),
            ("1", []),
            ("2", []),
            ("2", []),
        ],
        input_dim={"1": 2, "2": 4},
        normalize="none",
        kernel_initializer="ones",
    )

    xin, xout = hs.build()
    model = keras.Model(inputs=xin, outputs=xout)

    x = [
        np.array([[[1, 1]]]),
        np.array([[[2, 2], [2, 2]]]),
        np.array([[[4, 4, 4, 4], [4, 4, 4, 4]]]),
        np.array([[[3, 3], [3, 3], [3, 3], [3, 3]]]),
        np.array([[[6, 6, 6, 6], [6, 6, 6, 6], [6, 6, 6, 6], [6, 6, 6, 6]]]),
        np.array([[[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]]]),
    ]

    actual = model.predict(x)
    expected = np.array([[12, 35.5]])
    assert actual == pytest.approx(expected)


def test_hinsage_serialize():
    hs = HinSAGE(
        layer_sizes=[2, 2],
        n_samples=[2, 2],
        input_neighbor_tree=[
            ("1", [1, 2]),
            ("1", [3, 4]),
            ("2", [5]),
            ("1", []),
            ("2", []),
            ("2", []),
        ],
        input_dim={"1": 2, "2": 4},
        normalize="none",
        bias=False,
    )
    xin, xout = hs.build()
    model = keras.Model(inputs=xin, outputs=xout)

    # Save model
    model_json = model.to_json()

    # Set all weights to one
    model_weights = [np.ones_like(w) for w in model.get_weights()]

    # Load model from json & set all weights
    model2 = keras.models.model_from_json(
        model_json, custom_objects={"MeanHinAggregator": MeanHinAggregator}
    )
    model2.set_weights(model_weights)

    # Test loaded model
    x = [
        np.array([[[1, 1]]]),
        np.array([[[2, 2], [2, 2]]]),
        np.array([[[4, 4, 4, 4], [4, 4, 4, 4]]]),
        np.array([[[3, 3], [3, 3], [3, 3], [3, 3]]]),
        np.array([[[6, 6, 6, 6], [6, 6, 6, 6], [6, 6, 6, 6], [6, 6, 6, 6]]]),
        np.array([[[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]]]),
    ]
    actual = model2.predict(x)
    expected = np.array([[12, 35.5]])
    assert actual == pytest.approx(expected)


def test_hinsage_zero_neighbours():
    hs = HinSAGE(
        layer_sizes=[2, 2],
        n_samples=[0, 0],
        input_neighbor_tree=[
            ("1", [1, 2]),
            ("1", [3, 4]),
            ("2", [5]),
            ("1", []),
            ("2", []),
            ("2", []),
        ],
        input_dim={"1": 2, "2": 4},
        normalize="none",
        kernel_initializer="ones",
    )

    xin, xout = hs.build()
    model = keras.Model(inputs=xin, outputs=xout)

    x = [
        np.array([[[1.5, 1]]]),
        np.zeros((1, 0, 2)),
        np.zeros((1, 0, 4)),
        np.zeros((1, 0, 2)),
        np.zeros((1, 0, 4)),
        np.zeros((1, 0, 4)),
    ]

    actual = model.predict(x)
    expected = np.array([[2.5, 0]])
    assert actual == pytest.approx(expected)


def test_hinsage_aggregators():
    hs = HinSAGE(
        layer_sizes=[2, 2],
        n_samples=[2, 2],
        input_neighbor_tree=[
            ("1", [1, 2]),
            ("1", [3, 4]),
            ("2", [5]),
            ("1", []),
            ("2", []),
            ("2", []),
        ],
        input_dim={"1": 2, "2": 4},
        aggregator=MeanHinAggregator,
        normalize="none",
        kernel_initializer="ones",
    )

    xin, xout = hs.build()
    model = keras.Model(inputs=xin, outputs=xout)

    x = [
        np.array([[[1, 1]]]),
        np.array([[[2, 2], [2, 2]]]),
        np.array([[[4, 4, 4, 4], [4, 4, 4, 4]]]),
        np.array([[[3, 3], [3, 3], [3, 3], [3, 3]]]),
        np.array([[[6, 6, 6, 6], [6, 6, 6, 6], [6, 6, 6, 6], [6, 6, 6, 6]]]),
        np.array([[[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]]]),
    ]

    actual = model.predict(x)
    expected = np.array([[12, 35.5]])
    assert actual == pytest.approx(expected)


def test_hinsage_passing_activations():
    hs = HinSAGE(
        layer_sizes=[2, 2],
        n_samples=[2, 2],
        input_neighbor_tree=[
            ("1", [1, 2]),
            ("1", [3, 4]),
            ("2", [5]),
            ("1", []),
            ("2", []),
            ("2", []),
        ],
        input_dim={"1": 2, "2": 2},
    )
    assert hs.activations == ["relu", "linear"]

    with pytest.raises(ValueError):
        hs = HinSAGE(
            layer_sizes=[2, 2],
            n_samples=[2, 2],
            input_neighbor_tree=[
                ("1", [1, 2]),
                ("1", [3, 4]),
                ("2", [5]),
                ("1", []),
                ("2", []),
                ("2", []),
            ],
            input_dim={"1": 2, "2": 2},
            activations=["fred", "wilma"],
        )

    with pytest.raises(ValueError):
        hs = HinSAGE(
            layer_sizes=[2, 2],
            n_samples=[2, 2],
            input_neighbor_tree=[
                ("1", [1, 2]),
                ("1", [3, 4]),
                ("2", [5]),
                ("1", []),
                ("2", []),
                ("2", []),
            ],
            input_dim={"1": 2, "2": 2},
            activations=["relu"],
        )

    hs = HinSAGE(
        layer_sizes=[2, 2],
        n_samples=[2, 2],
        input_neighbor_tree=[
            ("1", [1, 2]),
            ("1", [3, 4]),
            ("2", [5]),
            ("1", []),
            ("2", []),
            ("2", []),
        ],
        input_dim={"1": 2, "2": 2},
        activations=["linear"] * 2,
    )
    assert hs.activations == ["linear"] * 2


def test_hinsage_regularisers():
    hs = HinSAGE(
        layer_sizes=[2, 2],
        n_samples=[2, 2],
        input_neighbor_tree=[
            ("1", [1, 2]),
            ("1", [3, 4]),
            ("2", [5]),
            ("1", []),
            ("2", []),
            ("2", []),
        ],
        input_dim={"1": 2, "2": 4},
        normalize="none",
        kernel_initializer="ones",
        kernel_regularizer=regularizers.l2(0.01),
    )

    with pytest.raises(ValueError):
        hs = HinSAGE(
            layer_sizes=[2, 2],
            n_samples=[2, 2],
            input_neighbor_tree=[
                ("1", [1, 2]),
                ("1", [3, 4]),
                ("2", [5]),
                ("1", []),
                ("2", []),
                ("2", []),
            ],
            input_dim={"1": 2, "2": 4},
            normalize="none",
            kernel_initializer="ones",
            kernel_regularizer="fred",
        )

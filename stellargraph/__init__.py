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


__all__ = [
    "data",
    "datasets",
    "calibration",
    "ensemble",
    "interpretability",
    "losses",
    "layer",
    "mapper",
    "utils",
    "custom_keras_layers",
    "StellarDiGraph",
    "StellarGraph",
    "GraphSchema",
    "__version__",
]

# Version
from .version import __version__

# Import modules
from stellargraph import (
    data,
    calibration,
    datasets,
    ensemble,
    interpretability,
    losses,
    layer,
    mapper,
    utils,
)

# Top-level imports
from stellargraph.core.graph import StellarGraph, StellarDiGraph
from stellargraph.core.indexed_array import IndexedArray
from stellargraph.core.schema import GraphSchema
import warnings

# Custom layers for keras deserialization (this is computed from a manual list to make it clear
# what's included)

# the `link_inference` module is shadowed in `sg.layer` by the `link_inference` function, so these
# layers need to be manually imported
from .layer.link_inference import (
    LinkEmbedding as _LinkEmbedding,
    LeakyClippedLinear as _LeakyClippedLinear,
)

custom_keras_layers = {
    class_.__name__: class_
    for class_ in [
        layer.GraphConvolution,
        layer.ClusterGraphConvolution,
        layer.GraphAttention,
        layer.GraphAttentionSparse,
        layer.SqueezedSparseConversion,
        layer.graphsage.MeanAggregator,
        layer.graphsage.MaxPoolingAggregator,
        layer.graphsage.MeanPoolingAggregator,
        layer.graphsage.AttentionalAggregator,
        layer.hinsage.MeanHinAggregator,
        layer.rgcn.RelationalGraphConvolution,
        layer.ppnp.PPNPPropagationLayer,
        layer.appnp.APPNPPropagationLayer,
        layer.misc.GatherIndices,
        layer.deep_graph_infomax.DGIDiscriminator,
        layer.deep_graph_infomax.DGIReadout,
        layer.graphsage.GraphSAGEAggregator,
        layer.knowledge_graph.ComplExScore,
        layer.knowledge_graph.DistMultScore,
        layer.knowledge_graph.RotatEScore,
        layer.knowledge_graph.RotHEScore,
        layer.preprocessing_layer.GraphPreProcessingLayer,
        layer.preprocessing_layer.SymmetricGraphPreProcessingLayer,
        layer.watch_your_step.AttentiveWalk,
        layer.sort_pooling.SortPooling,
        layer.gcn_lstm.FixedAdjacencyGraphConvolution,
        _LinkEmbedding,
        _LeakyClippedLinear,
        layer.ugraphemb_attention_pooling.UGraphEmbAttentionPooling,
    ]
}
"""
A dictionary of the ``tensorflow.keras`` layers defined by StellarGraph.

When Keras models using StellarGraph layers are saved, they can be loaded by passing this value to
the ``custom_objects`` parameter to model loading functions like
``tensorflow.keras.models.load_model``.

Example::

    import stellargraph as sg
    from tensorflow import keras
    keras.models.load_model("/path/to/model", custom_objects=sg.custom_keras_layers)
"""


def _top_level_deprecation_warning(name, path):
    warnings.warn(
        f"'{name}' is no longer available at the top-level. "
        f"Please use 'stellargraph.{path}.{name}' instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def expected_calibration_error(*args, **kwargs):
    _top_level_deprecation_warning("expected_calibration_error", "calibration")
    return calibration.expected_calibration_error(*args, **kwargs)


def plot_reliability_diagram(*args, **kwargs):
    _top_level_deprecation_warning("plot_reliability_diagram", "calibration")
    return calibration.plot_reliability_diagram(*args, **kwargs)


def Ensemble(*args, **kwargs):
    _top_level_deprecation_warning("Ensemble", "ensemble")
    return ensemble.Ensemble(*args, **kwargs)


def BaggingEnsemble(*args, **kwargs):
    _top_level_deprecation_warning("BaggingEnsemble", "ensemble")
    return ensemble.BaggingEnsemble(*args, **kwargs)


def TemperatureCalibration(*args, **kwargs):
    _top_level_deprecation_warning("TemperatureCalibration", "calibration")
    return calibration.TemperatureCalibration(*args, **kwargs)


def IsotonicCalibration(*args, **kwargs):
    _top_level_deprecation_warning("IsotonicCalibration", "calibration")
    return calibration.IsotonicCalibration(*args, **kwargs)

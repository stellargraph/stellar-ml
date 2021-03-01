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
GraphSAGE and compatible aggregator layers

"""
__all__ = [
    "GraphSAGE",
    "MeanAggregator",
    "MaxPoolingAggregator",
    "MeanPoolingAggregator",
    "AttentionalAggregator",
    "DirectedGraphSAGE",
]

import warnings
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Dropout, Reshape, LeakyReLU
from tensorflow.keras.utils import Sequence
from tensorflow.keras import activations, initializers, constraints, regularizers
from typing import List, Tuple, Callable, AnyStr, Union
from ..mapper import (
    GraphSAGENodeGenerator,
    GraphSAGELinkGenerator,
    DirectedGraphSAGENodeGenerator,
    DirectedGraphSAGELinkGenerator,
    NodeSequence,
    LinkSequence,
)

from .misc import deprecated_model_function
from ..connector.neo4j.mapper import (
    Neo4jGraphSAGENodeGenerator,
    Neo4jDirectedGraphSAGENodeGenerator,
    Neo4jGraphSAGELinkGenerator,
    Neo4jDirectedGraphSAGELinkGenerator,
)


class GraphSAGEAggregator(Layer):
    """
    Base class for GraphSAGE aggregators

    Args:
        output_dim (int): Output dimension
        bias (bool): Optional flag indicating whether (True) or not (False; default)
            a bias term should be included.
        act (Callable or str): name of the activation function to use (must be a
            Keras activation function), or alternatively, a TensorFlow operation.
        kernel_initializer (str or func): The initialiser to use for the weights
        kernel_regularizer (str or func): The regulariser to use for the weights
        kernel_constraint (str or func): The constraint to use for the weights
        bias_initializer (str or func): The initialiser to use for the bias
        bias_regularizer (str or func): The regulariser to use for the bias
        bias_constraint (str or func): The constraint to use for the bias
    """

    def __init__(
        self,
        output_dim: int = 0,
        bias: bool = False,
        act: Union[Callable, AnyStr] = "relu",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        **kwargs,
    ):
        self.output_dim = output_dim
        self.has_bias = bias
        self.act = activations.get(act)
        super().__init__(**kwargs)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        # These will be filled in at build time
        self.bias = None
        self.w_self = None
        self.w_group = None
        self.weight_dims = None
        self.included_weight_groups = None

    def get_config(self):
        """
        Gets class configuration for Keras serialization

        """
        config = {
            "output_dim": self.output_dim,
            "bias": self.has_bias,
            "act": activations.serialize(self.act),
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def calculate_group_sizes(self, input_shape):
        """
        Calculates the output size for each input group. The results are stored in two variables:
            self.included_weight_groups: if the corresponding entry is True then the input group
                is valid and should be used.
            self.weight_sizes: the size of the output from this group.

        Args:
            input_shape (list of list of int): Shape of input tensors for self
                and neighbour features
        """
        # If the neighbours are zero-dimensional for any of the shapes
        # in the input, do not use the input group in the model.
        # XXX Ignore batch size, since test dim != 0 evaluates to None!!
        self.included_weight_groups = [
            all(dim != 0 for dim in group_shape[1:]) for group_shape in input_shape
        ]

        # The total number of enabled input groups
        num_groups = np.sum(self.included_weight_groups)
        if num_groups < 1:
            raise ValueError(
                "There must be at least one input with a non-zero neighbourhood dimension"
            )

        # Calculate the dimensionality of each group, and put remainder into the first group
        # with non-zero dimensions, which should be the head node group.
        group_output_dim = self.output_dim // num_groups
        remainder_dim = self.output_dim - num_groups * group_output_dim
        weight_dims = []
        for g in self.included_weight_groups:
            if g:
                group_dim = group_output_dim + remainder_dim
                remainder_dim = 0
            else:
                group_dim = 0
            weight_dims.append(group_dim)
        self.weight_dims = weight_dims

    def build(self, input_shape):
        """
        Builds the weight tensor corresponding to the features
        of the initial nodes in sampled random walks.
        Optionally builds the weight tensor(s) corresponding
        to sampled neighbourhoods, if required.
        Optionally builds the bias tensor, if requested.

        Args:
            input_shape (list of list of int): Shape of input tensors for self
                and neighbour features

        """
        if not isinstance(input_shape, list):
            raise ValueError(
                "Expected a list of inputs, not {}".format(type(input_shape).__name__)
            )

        # Configure bias vector, if used.
        if self.has_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )

        # Calculate weight size for each input group
        self.calculate_group_sizes(input_shape)

        # Configure weights for input groups, if used.
        w_group = [None] * len(input_shape)
        for ii, g_shape in enumerate(input_shape):
            if self.included_weight_groups[ii]:
                weight = self._build_group_weights(
                    g_shape, self.weight_dims[ii], group_idx=ii
                )
                w_group[ii] = weight
        self.w_group = w_group

        # Signal that the build has completed.
        super().build(input_shape)

    def _build_group_weights(self, in_shape, out_size, group_idx=0):
        """
        Builds the weight tensor(s) corresponding to the features of the input groups.

        Args:
            in_shape (list of int): Shape of input tensor for single group
            out_size (int): The size of the output vector for this group
            group_idx (int): The index of the input group

        """
        weight = self.add_weight(
            shape=(int(in_shape[-1]), out_size),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            name=f"weight_g{group_idx}",
        )
        return weight

    def aggregate_neighbours(self, x_neigh, group_idx: int = 0):
        """
        Override with a method to aggregate tensors over neighbourhood.

        Args:
            x_neigh: The input tensor representing the sampled neighbour nodes.
            group_idx: Optional neighbourhood index used for multi-dimensional hops.

        Returns:
            A tensor aggregation of the input nodes features.
        """
        raise NotImplementedError(
            "The GraphSAGEAggregator base class should not be directly instantiated"
        )

    def call(self, inputs, **kwargs):
        """
        Apply aggregator on the input tensors, `inputs`

        Args:
          inputs: List of Keras tensors

        Returns:
            Keras Tensor representing the aggregated embeddings in the input.

        """
        # If a neighbourhood dimension exists for the group, aggregate over the neighbours
        # otherwise create a simple layer.
        sources = []
        for ii, x in enumerate(inputs):
            # If the group is included, apply aggregation and collect the output tensor
            # otherwise, this group is ignored
            if self.included_weight_groups[ii]:
                x_agg = self.group_aggregate(x, group_idx=ii)
                sources.append(x_agg)

        # Concatenate outputs from all groups
        # TODO: Generalize to sum a subset of groups.
        h_out = K.concatenate(sources, axis=2)

        # Optionally add bias
        if self.has_bias:
            h_out = h_out + self.bias

        # Finally, apply activation
        return self.act(h_out)

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.
        Assumes that the layer will be built to match that input shape provided.

        Args:
            input_shape (tuple of int)
                Shape tuples can include None for free dimensions, instead of an integer.

        Returns:
            The output shape calculated from the input shape, this is of the form
                (batch_num, head_num, output_dim)
        """
        return input_shape[0][0], input_shape[0][1], self.output_dim

    def group_aggregate(self, x_neigh, group_idx=0):
        """
        Override with a method to aggregate tensors over the neighbourhood for each group.

        Args:
            x_neigh (tf.Tensor): : The input tensor representing the sampled neighbour nodes.
            group_idx (int, optional): Group index.

        Returns:
            :class:`tensorflow.Tensor`: A tensor aggregation of the input nodes features.
        """
        raise NotImplementedError(
            "The GraphSAGEAggregator base class should not be directly instantiated"
        )


class MeanAggregator(GraphSAGEAggregator):
    """
    Mean Aggregator for GraphSAGE implemented with Keras base layer

    Args:
        output_dim (int): Output dimension
        bias (bool): Optional bias
        act (Callable or str): name of the activation function to use (must be a
            Keras activation function), or alternatively, a TensorFlow operation.

    """

    def group_aggregate(self, x_group, group_idx=0):
        """
        Mean aggregator for tensors over the neighbourhood for each group.

        Args:
            x_group (tf.Tensor): : The input tensor representing the sampled neighbour nodes.
            group_idx (int, optional): Group index.

        Returns:
            :class:`tensorflow.Tensor`: A tensor aggregation of the input nodes features.
        """
        # The first group is assumed to be the self-tensor and we do not aggregate over it
        if group_idx == 0:
            x_agg = x_group
        else:
            x_agg = K.mean(x_group, axis=2)

        return K.dot(x_agg, self.w_group[group_idx])


class MaxPoolingAggregator(GraphSAGEAggregator):
    """
    Max Pooling Aggregator for GraphSAGE implemented with Keras base layer

    Implements the aggregator of Eq. (3) in Hamilton et al. (2017)

    Args:
        output_dim (int): Output dimension
        bias (bool): Optional bias
        act (Callable or str): name of the activation function to use (must be a
            Keras activation function), or alternatively, a TensorFlow operation.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: These should be user parameters
        self.hidden_dim = self.output_dim
        self.hidden_act = activations.get("relu")

    def _build_group_weights(self, in_shape, out_size, group_idx=0):
        """
        Builds the weight tensor(s) corresponding to the features of the input groups.

        Args:
            in_shape (list of int): Shape of input tensor for single group
            out_size (int): The size of the output vector for this group
            group_idx (int): The index of the input group

        """
        if group_idx == 0:
            weights = self.add_weight(
                name=f"w_g{group_idx}",
                shape=(int(in_shape[-1]), out_size),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )
        else:
            w_group = self.add_weight(
                name=f"w_g{group_idx}",
                shape=(self.hidden_dim, out_size),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )
            w_pool = self.add_weight(
                name=f"w_pool_g{group_idx}",
                shape=(int(in_shape[-1]), self.hidden_dim),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )
            b_pool = self.add_weight(
                name=f"b_pool_g{group_idx}",
                shape=(self.hidden_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )
            weights = [w_group, w_pool, b_pool]
        return weights

    def group_aggregate(self, x_group, group_idx=0):
        """
        Aggregates the group tensors by max-pooling of neighbours

        Args:
            x_group (tf.Tensor): : The input tensor representing the sampled neighbour nodes.
            group_idx (int, optional): Group index.

        Returns:
            :class:`tensorflow.Tensor`: A tensor aggregation of the input nodes features.
        """
        if group_idx == 0:
            # Do not aggregate features for head nodes
            x_agg = K.dot(x_group, self.w_group[0])

        else:
            w_g, w_pool, b_pool = self.w_group[group_idx]

            # Pass neighbour features through a dense layer with w_pool, b_pool
            xw_neigh = self.hidden_act(K.dot(x_group, w_pool) + b_pool)

            # Take max of this tensor over neighbour dimension
            x_agg = K.max(xw_neigh, axis=2)

            # Final output is a dense layer over the aggregated tensor
            x_agg = K.dot(x_agg, w_g)
        return x_agg


class MeanPoolingAggregator(GraphSAGEAggregator):
    """
    Mean Pooling Aggregator for GraphSAGE implemented with Keras base layer

    Implements the aggregator of Eq. (3) in Hamilton et al. (2017), with max pooling replaced with mean pooling

    Args:
        output_dim (int): Output dimension
        bias (bool): Optional bias
        act (Callable or str): name of the activation function to use (must be a
            Keras activation function), or alternatively, a TensorFlow operation.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: These should be user parameters
        self.hidden_dim = self.output_dim
        self.hidden_act = activations.get("relu")

    def _build_group_weights(self, in_shape, out_size, group_idx=0):
        """
        Builds the weight tensor(s) corresponding to the features of the input groups.

        Args:
            in_shape (list of int): Shape of input tensor for single group
            out_size (int): The size of the output vector for this group
            group_idx (int): The index of the input group

        """
        if group_idx == 0:
            weights = self.add_weight(
                name=f"w_g{group_idx}",
                shape=(int(in_shape[-1]), out_size),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )
        else:
            w_group = self.add_weight(
                name=f"w_g{group_idx}",
                shape=(self.hidden_dim, out_size),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )
            w_pool = self.add_weight(
                name=f"w_pool_g{group_idx}",
                shape=(int(in_shape[-1]), self.hidden_dim),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )
            b_pool = self.add_weight(
                name=f"b_pool_g{group_idx}",
                shape=(self.hidden_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )
            weights = [w_group, w_pool, b_pool]
        return weights

    def group_aggregate(self, x_group, group_idx=0):
        """
        Aggregates the group tensors by mean-pooling of neighbours

        Args:
            x_group (tf.Tensor): : The input tensor representing the sampled neighbour nodes.
            group_idx (int, optional): Group index.

        Returns:
            :class:`tensorflow.Tensor`: A tensor aggregation of the input nodes features.
        """
        if group_idx == 0:
            # Do not aggregate features for head nodes
            x_agg = K.dot(x_group, self.w_group[0])

        else:
            w_g, w_pool, b_pool = self.w_group[group_idx]

            # Pass neighbour features through a dense layer with w_pool, b_pool
            xw_neigh = self.hidden_act(K.dot(x_group, w_pool) + b_pool)

            # Take max of this tensor over neighbour dimension
            x_agg = K.mean(xw_neigh, axis=2)

            # Final output is a dense layer over the aggregated tensor
            x_agg = K.dot(x_agg, w_g)
        return x_agg


class AttentionalAggregator(GraphSAGEAggregator):
    """
    Attentional Aggregator for GraphSAGE implemented with Keras base layer

    Implements the aggregator of Veličković et al. "Graph Attention Networks" ICLR 2018

    Args:
        output_dim (int): Output dimension
        bias (bool): Optional bias
        act (Callable or str): name of the activation function to use (must be a
            Keras activation function), or alternatively, a TensorFlow operation.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: How can we expose these options to the user?
        self.hidden_dim = self.output_dim
        self.attn_act = LeakyReLU(0.2)

    def _build_group_weights(self, in_shape, out_size, group_idx=0):
        """
        Builds the weight tensor(s) corresponding to the features of the input groups.

        Args:
            in_shape (list of int): Shape of input tensor for single group
            out_size (int): The size of the output vector for this group
            group_idx (int): The index of the input group

        """
        if group_idx == 0:
            if out_size > 0:
                weights = self.add_weight(
                    name=f"w_self",
                    shape=(int(in_shape[-1]), out_size),
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                    trainable=True,
                )
            else:
                weights = None

        else:
            w_g = self.add_weight(
                name=f"w_g{group_idx}",
                shape=(int(in_shape[-1]), out_size),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )
            w_attn_s = self.add_weight(
                name=f"w_attn_s{group_idx}",
                shape=(out_size, 1),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )
            w_attn_g = self.add_weight(
                name=f"w_attn_g{group_idx}",
                shape=(out_size, 1),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )
            weights = [w_g, w_attn_s, w_attn_g]
        return weights

    def calculate_group_sizes(self, input_shape):
        """
        Calculates the output size for each input group.

        The results are stored in two variables:
            * self.included_weight_groups: if the corresponding entry is True then the input group
              is valid and should be used.
            * self.weight_sizes: the size of the output from this group.

        The AttentionalAggregator is implemented to not use the first (head node) group. This makes
        the implementation different from other aggregators.

        Args:
            input_shape (list of list of int): Shape of input tensors for self
                and neighbour features
        """
        # If the neighbours are zero-dimensional for any of the shapes
        # in the input, do not use the input group in the model.
        # XXX Ignore batch size, since dim != 0 results in None!!
        self.included_weight_groups = [
            all(dim != 0 for dim in group_shape[1:]) for group_shape in input_shape
        ]

        # The total number of enabled input groups
        num_groups = np.sum(self.included_weight_groups) - 1

        # We do not assign any features to the head node group, unless this is the only group.
        if num_groups == 0:
            weight_dims = [self.output_dim] + [0] * (len(input_shape) - 1)

        else:
            # Calculate the dimensionality of each group, and put remainder into the first group
            # with non-zero dimensions.
            group_output_dim = self.output_dim // num_groups
            remainder_dim = self.output_dim - num_groups * group_output_dim
            weight_dims = [0]
            for g in self.included_weight_groups[1:]:
                if g:
                    group_dim = group_output_dim + remainder_dim
                    remainder_dim = 0
                else:
                    group_dim = 0
                weight_dims.append(group_dim)

        self.weight_dims = weight_dims

    def call(self, inputs, **kwargs):
        """
        Apply aggregator on the input tensors, `inputs`

        Args:
          inputs (List[Tensor]): Tensors giving self and neighbour features
                x[0]: self Tensor (batch_size, head size, feature_size)
                x[k>0]: group Tensors for neighbourhood (batch_size, head size, neighbours, feature_size)

        Returns:
            Keras Tensor representing the aggregated embeddings in the input.

        """
        # We require the self group to be included to calculate attention
        if not self.included_weight_groups[0]:
            raise ValueError("The head node group must have non-zero dimension")

        # If a neighbourhood dimension exists for the group, aggregate over the neighbours
        # otherwise create a simple layer.
        x_self = inputs[0]
        group_sources = []
        for ii, x_g in enumerate(inputs[1:]):
            group_idx = ii + 1
            if not self.included_weight_groups[group_idx]:
                continue

            # Get the weights for this group
            w_g, w_attn_s, w_attn_g = self.w_group[group_idx]

            # Group transform for self & neighbours
            xw_self = K.expand_dims(K.dot(x_self, w_g), axis=2)
            xw_neigh = K.dot(x_g, w_g)

            # Concatenate self vector to neighbour vectors
            # Shape is (n_b, n_h, n_neigh+1, n_out[ii])
            xw_all = K.concatenate([xw_self, xw_neigh], axis=2)

            # Calculate group attention
            attn_self = K.dot(xw_self, w_attn_s)  # (n_b, n_h, 1)
            attn_neigh = K.dot(xw_all, w_attn_g)  # (n_b, n_h, n_neigh+1, 1)

            # Add self and neighbour attn and apply activation
            # Note: This broadcasts to (n_b, n_h, n_neigh + 1, 1)
            attn_u = self.attn_act(attn_self + attn_neigh)

            # Attn coefficients, softmax over the neighbours
            attn = K.softmax(attn_u, axis=2)

            # Multiply attn coefficients by neighbours (and self) and aggregate
            h_out = K.sum(attn * xw_all, axis=2)
            group_sources.append(h_out)

        # If there are no groups with features built, fallback to a MLP on the head node features
        if not group_sources:
            group_sources = [K.dot(x_self, self.w_group[0])]

        # Concatenate or sum the outputs from all groups
        h_out = K.concatenate(group_sources, axis=2)

        if self.has_bias:
            h_out = h_out + self.bias

        return self.act(h_out)


def _require_without_generator(value, name):
    if value is not None:
        return value
    else:
        raise ValueError(
            f"{name}: expected a value for 'n_samples', 'input_dim', and 'multiplicity' when "
            f"'generator' is not provided, found {name}=None."
        )


class GraphSAGE:
    """
    Implementation of the GraphSAGE algorithm of Hamilton et al. with Keras layers.
    see: http://snap.stanford.edu/graphsage/

    The model minimally requires specification of the layer sizes as a list of int
    corresponding to the feature dimensions for each hidden layer and a generator object.

    Different neighbour node aggregators can also be specified with the ``aggregator``
    argument, which should be the aggregator class,
    either :class:`.MeanAggregator`, :class:`.MeanPoolingAggregator`,
    :class:`.MaxPoolingAggregator`, or :class:`.AttentionalAggregator`.

    To use this class as a Keras model, the features and graph should be supplied using the
    :class:`.GraphSAGENodeGenerator` class for node inference models or the
    :class:`.GraphSAGELinkGenerator` class for link inference models.  The `.in_out_tensors` method should
    be used to create a Keras model from the `GraphSAGE` object.

    Examples:
        Creating a two-level GrapSAGE node classification model with hidden node sizes of 8 and 4
        and 10 neighbours sampled at each layer using an existing :class:`.StellarGraph` object `G`
        containing the graph and node features::

            generator = GraphSAGENodeGenerator(G, batch_size=50, num_samples=[10,10])
            gat = GraphSAGE(
                    layer_sizes=[8, 4],
                    activations=["relu","softmax"],
                    generator=generator,
                )
            x_inp, predictions = gat.in_out_tensors()

    Note that passing a `NodeSequence` or `LinkSequence` object from the `generator.flow(...)` method
    as the `generator=` argument is now deprecated and the base generator object should be passed instead.

    .. seealso::

       Examples using GraphSAGE:

       - node classification: `natively <https://stellargraph.readthedocs.io/en/stable/demos/node-classification/graphsage-node-classification.html>`__, `via Neo4j <https://stellargraph.readthedocs.io/en/stable/demos/connector/neo4j/undirected-graphsage-on-cora-neo4j-example.html>`__
       - `link prediction <https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/graphsage-link-prediction.html>`__
       - unsupervised representation learning: `via random walks <https://stellargraph.readthedocs.io/en/stable/demos/embeddings/graphsage-unsupervised-sampler-embeddings.html>`__, `via Deep Graph Infomax <https://stellargraph.readthedocs.io/en/stable/demos/embeddings/deep-graph-infomax-embeddings.html>`__
       - calibrating models: `node classification <https://stellargraph.readthedocs.io/en/stable/demos/calibration/calibration-node-classification.html>`__, `link prediction <https://stellargraph.readthedocs.io/en/stable/demos/calibration/calibration-link-prediction.html>`__
       - ensemble models: `node classification <https://stellargraph.readthedocs.io/en/stable/demos/ensembles/ensemble-node-classification-example.html>`__, `link prediction <https://stellargraph.readthedocs.io/en/stable/demos/ensembles/ensemble-link-prediction-example.html>`__
       - `comparison of link prediction algorithms <https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/homogeneous-comparison-link-prediction.html>`__

       Appropriate data generators: :class:`.GraphSAGENodeGenerator`, :class:`.Neo4jGraphSAGENodeGenerator`, :class:`.GraphSAGELinkGenerator`.

       Related models:

       - :class:`.DirectedGraphSAGE` for a generalisation to directed graphs
       - :class:`.HinSAGE` for a generalisation to heterogeneous graphs
       - :class:`.DeepGraphInfomax` for unsupervised training

       Aggregators: :class:`.MeanAggregator`, :class:`.MeanPoolingAggregator`, :class:`.MaxPoolingAggregator`, :class:`.AttentionalAggregator`.

    Args:
        layer_sizes (list): Hidden feature dimensions for each layer.
        generator (GraphSAGENodeGenerator or GraphSAGELinkGenerator):
            If specified `n_samples` and `input_dim` will be extracted from this object.
        aggregator (class): The GraphSAGE aggregator to use; defaults to the `MeanAggregator`.
        bias (bool): If True (default), a bias vector is learnt for each layer.
        dropout (float): The dropout supplied to each layer; defaults to no dropout.
        normalize (str or None): The normalization used after each layer; defaults to L2 normalization.
        activations (list): Activations applied to each layer's output;
            defaults to ``['relu', ..., 'relu', 'linear']``.
        kernel_initializer (str or func, optional): The initialiser to use for the weights of each layer.
        kernel_regularizer (str or func, optional): The regulariser to use for the weights of each layer.
        kernel_constraint (str or func, optional): The constraint to use for the weights of each layer.
        bias_initializer (str or func, optional): The initialiser to use for the bias of each layer.
        bias_regularizer (str or func, optional): The regulariser to use for the bias of each layer.
        bias_constraint (str or func, optional): The constraint to use for the bias of each layer.
        n_samples (list, optional): The number of samples per layer in the model.
        input_dim (int, optional): The dimensions of the node features used as input to the model.
        multiplicity (int, optional): The number of nodes to process at a time. This is 1 for a node inference
          and 2 for link inference (currently no others are supported).

    .. note::
        The values for ``n_samples``, ``input_dim``, and ``multiplicity`` are obtained from the provided
        ``generator`` by default. The additional keyword arguments for these parameters provide an
        alternative way to specify them if a generator cannot be supplied.

    """

    def __init__(
        self,
        layer_sizes,
        generator=None,
        aggregator=None,
        bias=True,
        dropout=0.0,
        normalize="l2",
        activations=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        n_samples=None,
        input_dim=None,
        multiplicity=None,
    ):
        # Model parameters
        self.layer_sizes = layer_sizes
        self.max_hops = len(layer_sizes)
        self.bias = bias
        self.dropout = dropout

        # Set the normalization layer used in the model
        if normalize == "l2":
            self._normalization = Lambda(lambda x: K.l2_normalize(x, axis=-1))

        elif normalize is None or normalize == "none" or normalize == "None":
            self._normalization = Lambda(lambda x: x)

        else:
            raise ValueError(
                "Normalization should be either 'l2' or 'none'; received '{}'".format(
                    normalize
                )
            )

        # Get the input_dim and num_samples
        if generator is not None:
            self._get_sizes_from_generator(generator)
        else:
            self.n_samples = _require_without_generator(n_samples, "n_samples")
            self.input_feature_size = _require_without_generator(input_dim, "input_dim")
            self.multiplicity = _require_without_generator(multiplicity, "multiplicity")
            # Check the number of samples and the layer sizes are consistent
            if len(self.n_samples) != self.max_hops:
                raise ValueError(
                    f"n_samples: expected one sample size for each of the {self.max_hops} layers, "
                    f"found {len(self.n_samples)} sample sizes"
                )

        # Feature dimensions for each layer
        self.dims = [self.input_feature_size] + layer_sizes

        # Compute size of each sampled neighbourhood
        self._compute_neighbourhood_sizes()

        # Set the aggregator layer used in the model
        if aggregator is None:
            self._aggregator = MeanAggregator
        elif issubclass(aggregator, Layer):
            self._aggregator = aggregator
        else:
            raise TypeError("Aggregator should be a subclass of Keras Layer")

        # Activation function for each layer
        if activations is None:
            activations = ["relu"] * (self.max_hops - 1) + ["linear"]
        elif len(activations) != self.max_hops:
            raise ValueError(
                "Invalid number of activations; require one function per layer"
            )
        self.activations = activations

        # Aggregator functions for each layer
        self._aggs = [
            self._aggregator(
                output_dim=self.layer_sizes[layer],
                bias=self.bias,
                act=self.activations[layer],
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                kernel_constraint=kernel_constraint,
                bias_initializer=bias_initializer,
                bias_regularizer=bias_regularizer,
                bias_constraint=bias_constraint,
            )
            for layer in range(self.max_hops)
        ]

    def _get_sizes_from_generator(self, generator):
        """
        Sets n_samples and input_feature_size from the generator.
        Args:
             generator: The supplied generator.
        """
        if not isinstance(
            generator,
            (
                GraphSAGENodeGenerator,
                GraphSAGELinkGenerator,
                Neo4jGraphSAGENodeGenerator,
                Neo4jGraphSAGELinkGenerator,
            ),
        ):
            errmsg = "Generator should be an instance of GraphSAGENodeGenerator or GraphSAGELinkGenerator"
            if isinstance(generator, (NodeSequence, LinkSequence)):
                errmsg = (
                    "Passing a Sequence object as the generator to GraphSAGE is no longer supported. "
                    + errmsg
                )
            raise TypeError(errmsg)

        self.n_samples = generator.num_samples
        # Check the number of samples and the layer sizes are consistent
        if len(self.n_samples) != self.max_hops:
            raise ValueError(
                "Mismatched lengths: neighbourhood sample sizes {} versus layer sizes {}".format(
                    self.n_samples, self.layer_sizes
                )
            )

        self.multiplicity = generator.multiplicity
        feature_sizes = generator.graph.node_feature_sizes()
        if len(feature_sizes) > 1:
            raise RuntimeError(
                "GraphSAGE called on graph with more than one node type."
            )
        self.input_feature_size = feature_sizes.popitem()[1]

    def _compute_neighbourhood_sizes(self):
        """
        Computes the total (cumulative product) number of nodes
        sampled at each neighbourhood.

        Each hop samples from the neighbours of the previous nodes.
        """

        def size_at(i):
            return np.product(self.n_samples[:i], dtype=int)

        self.neighbourhood_sizes = [size_at(i) for i in range(self.max_hops + 1)]

    def __call__(self, xin: List):
        """
        Apply aggregator layers

        Args:
            xin (list of Tensor): Batch input features

        Returns:
            Output tensor
        """

        def apply_layer(x: List, num_hops: int):
            """
            Compute the list of output tensors for a single GraphSAGE layer

            Args:
                x (List[Tensor]): Inputs to the layer
                num_hops (int): Layer index to construct

            Returns:
                Outputs of applying the aggregators as a list of Tensors

            """
            layer_out = []
            for i in range(self.max_hops - num_hops):
                head_shape = K.int_shape(x[i])[1]

                # Reshape neighbours per node per layer
                neigh_in = Dropout(self.dropout)(
                    Reshape((head_shape, self.n_samples[i], self.dims[num_hops]))(
                        x[i + 1]
                    )
                )

                # Apply aggregator to head node and neighbour nodes
                layer_out.append(
                    self._aggs[num_hops]([Dropout(self.dropout)(x[i]), neigh_in])
                )

            return layer_out

        if not isinstance(xin, list):
            raise TypeError("Input features to GraphSAGE must be a list")

        if len(xin) != self.max_hops + 1:
            raise ValueError(
                "Length of input features should equal the number of GraphSAGE layers plus one"
            )

        # Form GraphSAGE layers iteratively
        h_layer = xin
        for layer in range(0, self.max_hops):
            h_layer = apply_layer(h_layer, layer)

        # Remove neighbourhood dimension from output tensors of the stack
        # note that at this point h_layer contains the output tensor of the top (last applied) layer of the stack
        h_layer = [
            Reshape(K.int_shape(x)[2:])(x) if K.int_shape(x)[1] == 1 else x
            for x in h_layer
        ]

        return (
            self._normalization(h_layer[0])
            if len(h_layer) == 1
            else [self._normalization(xi) for xi in h_layer]
        )

    def _node_model(self):
        """
        Builds a GraphSAGE model for node prediction

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a list of Keras input tensors
            for the specified GraphSAGE model and ``x_out`` is the Keras tensor
            for the GraphSAGE model output.

        """
        # Create tensor inputs for neighbourhood sampling
        x_inp = [
            Input(shape=(s, self.input_feature_size)) for s in self.neighbourhood_sizes
        ]

        # Output from GraphSAGE model
        x_out = self(x_inp)

        # Returns inputs and outputs
        return x_inp, x_out

    def _link_model(self):
        """
        Builds a GraphSAGE model for link or node pair prediction

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a list of Keras input tensors for (src, dst) node pairs
            (where (src, dst) node inputs alternate),
            and ``x_out`` is a list of output tensors for (src, dst) nodes in the node pairs

        """
        # Expose input and output sockets of the model, for source and destination nodes:
        x_inp_src, x_out_src = self._node_model()
        x_inp_dst, x_out_dst = self._node_model()
        # re-pack into a list where (source, target) inputs alternate, for link inputs:
        x_inp = [x for ab in zip(x_inp_src, x_inp_dst) for x in ab]
        # same for outputs:
        x_out = [x_out_src, x_out_dst]
        return x_inp, x_out

    def in_out_tensors(self, multiplicity=None):
        """
        Builds a GraphSAGE model for node or link/node pair prediction, depending on the generator used to construct
        the model (whether it is a node or link/node pair generator).

        Returns:
            tuple: ``(x_inp, x_out)``, where ``x_inp`` is a list of Keras input tensors
                for the specified GraphSAGE model (either node or link/node pair model) and ``x_out`` contains
                model output tensor(s) of shape (batch_size, layer_sizes[-1])

        """
        if multiplicity is None:
            multiplicity = self.multiplicity

        if multiplicity == 1:
            return self._node_model()
        elif multiplicity == 2:
            return self._link_model()
        else:
            raise RuntimeError(
                "Currently only multiplicities of 1 and 2 are supported. Consider using node_model or "
                "link_model method explicitly to build node or link prediction model, respectively."
            )

    def default_model(self, flatten_output=True):
        warnings.warn(
            "The .default_model() method is deprecated. Please use .in_out_tensors() method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.in_out_tensors()

    node_model = deprecated_model_function(_node_model, "node_model")
    link_model = deprecated_model_function(_link_model, "link_model")
    build = deprecated_model_function(in_out_tensors, "build")


class DirectedGraphSAGE(GraphSAGE):
    """
    Implementation of a directed version of the GraphSAGE algorithm of Hamilton et al. with Keras layers.
    see: http://snap.stanford.edu/graphsage/

    The model minimally requires specification of the layer sizes as a list of int
    corresponding to the feature dimensions for each hidden layer and a generator object.

    Different neighbour node aggregators can also be specified with the ``aggregator``
    argument, which should be the aggregator class,
    either :class:`.MeanAggregator`, :class:`.MeanPoolingAggregator`,
    :class:`.MaxPoolingAggregator`, or :class:`.AttentionalAggregator`.

    .. seealso::

       Examples using Directed GraphSAGE:

       - `node classification <https://stellargraph.readthedocs.io/en/stable/demos/node-classification/directed-graphsage-node-classification.html>`__
       - `node classification with Neo4j <https://stellargraph.readthedocs.io/en/stable/demos/connector/neo4j/directed-graphsage-on-cora-neo4j-example.html>`__

       Appropriate data generators: :class:`.DirectedGraphSAGENodeGenerator`, :class:`.Neo4jDirectedGraphSAGENodeGenerator`, :class:`.DirectedGraphSAGELinkGenerator`.

       Related models:

       - :class:`.GraphSAGE` for undirected graphs
       - :class:`.HinSAGE` for undirected heterogeneous graphs

       Aggregators: :class:`.MeanAggregator`, :class:`.MeanPoolingAggregator`, :class:`.MaxPoolingAggregator`, :class:`.AttentionalAggregator`.

    Args:
        layer_sizes (list): Hidden feature dimensions for each layer.
        generator (DirectedGraphSAGENodeGenerator):
            If specified `n_samples` and `input_dim` will be extracted from this object.
        aggregator (class, optional): The GraphSAGE aggregator to use; defaults to the `MeanAggregator`.
        bias (bool, optional): If True (default), a bias vector is learnt for each layer.
        dropout (float, optional): The dropout supplied to each layer; defaults to no dropout.
        normalize (str, optional): The normalization used after each layer; defaults to L2 normalization.
        kernel_initializer (str or func, optional): The initialiser to use for the weights of each layer.
        kernel_regularizer (str or func, optional): The regulariser to use for the weights of each layer.
        kernel_constraint (str or func, optional): The constraint to use for the weights of each layer.
        bias_initializer (str or func, optional): The initialiser to use for the bias of each layer.
        bias_regularizer (str or func, optional): The regulariser to use for the bias of each layer.
        bias_constraint (str or func, optional): The constraint to use for the bias of each layer.

    Notes::
        If a generator is not specified, then additional keyword arguments must be supplied:

        * in_samples (list): The number of in-node samples per layer in the model.

        * out_samples (list): The number of out-node samples per layer in the model.

        * input_dim (int): The dimensions of the node features used as input to the model.

        * multiplicity (int): The number of nodes to process at a time. This is 1 for a node inference
          and 2 for link inference (currently no others are supported).

        Passing a `NodeSequence` or `LinkSequence` object from the `generator.flow(...)` method
        as the `generator=` argument is now deprecated and the base generator object should be passed instead.

        """

    def _get_sizes_from_generator(self, generator):
        """
        Sets in_samples, out_samples and input_feature_size from the generator.
        Args:
             generator: The supplied generator.
        """
        if not isinstance(
            generator,
            (
                DirectedGraphSAGENodeGenerator,
                DirectedGraphSAGELinkGenerator,
                Neo4jDirectedGraphSAGENodeGenerator,
            ),
        ):
            errmsg = "Generator should be an instance of DirectedGraphSAGENodeGenerator"
            if isinstance(generator, (NodeSequence, LinkSequence)):
                errmsg = (
                    "Passing a Sequence object as the generator to DirectedGraphSAGE is no longer supported. "
                    + errmsg
                )
            raise TypeError(errmsg)

        self.in_samples = generator.in_samples
        if len(self.in_samples) != self.max_hops:
            raise ValueError(
                "Mismatched lengths: in-node sample sizes {} versus layer sizes {}".format(
                    self.in_samples, self.layer_sizes
                )
            )
        self.out_samples = generator.out_samples
        if len(self.out_samples) != self.max_hops:
            raise ValueError(
                "Mismatched lengths: out-node sample sizes {} versus layer sizes {}".format(
                    self.out_samples, self.layer_sizes
                )
            )
        feature_sizes = generator.graph.node_feature_sizes()
        if len(feature_sizes) > 1:
            raise RuntimeError(
                "DirectedGraphSAGE called on graph with more than one node type."
            )
        self.input_feature_size = feature_sizes.popitem()[1]
        self.multiplicity = generator.multiplicity

    def _get_sizes_from_keywords(self, **kwargs):
        """
        Sets in_samples, out_samples and input_feature_size from the keywords.
        Args:
             kwargs: The additional keyword arguments.
        """
        try:
            self.in_samples = kwargs["in_samples"]
            self.out_samples = kwargs["out_samples"]
            self.input_feature_size = kwargs["input_dim"]
            self.multiplicity = kwargs["multiplicity"]

        except KeyError:
            raise KeyError(
                "If generator is not provided, in_samples, out_samples, "
                "input_dim, and multiplicity must be specified."
            )

        if len(self.in_samples) != self.max_hops:
            raise ValueError(
                "Mismatched lengths: in-node sample sizes {} versus layer sizes {}".format(
                    self.in_samples, self.layer_sizes
                )
            )
        if len(self.out_samples) != self.max_hops:
            raise ValueError(
                "Mismatched lengths: out-node sample sizes {} versus layer sizes {}".format(
                    self.out_samples, self.layer_sizes
                )
            )

    def _compute_neighbourhood_sizes(self):
        """
        Computes the total (cumulative product) number of nodes
        sampled at each neighbourhood.

        Each hop has to sample separately from both the in-nodes
        and the out-nodes of the previous nodes.
        This gives rise to a binary tree of directed neighbourhoods.
        """
        self.max_slots = 2 ** (self.max_hops + 1) - 1
        self.neighbourhood_sizes = [1] + [
            np.product(
                [
                    self.in_samples[kk] if d == "0" else self.out_samples[kk]
                    for kk, d in enumerate(np.binary_repr(ii + 1)[1:])
                ]
            )
            for ii in range(1, self.max_slots)
        ]

    def __call__(self, xin: List):
        """
        Apply aggregator layers

        Args:
            xin (list of Tensor): Batch input features

        Returns:
            Output tensor
        """

        def aggregate_neighbours(tree: List, stage: int):
            # compute the number of slots with children in the binary tree
            num_slots = (len(tree) - 1) // 2
            new_tree = [None] * num_slots
            for slot in range(num_slots):
                # get parent nodes
                num_head_nodes = K.int_shape(tree[slot])[1]
                parent = Dropout(self.dropout)(tree[slot])
                # find in-nodes
                child_slot = 2 * slot + 1
                size = (
                    self.neighbourhood_sizes[child_slot] // num_head_nodes
                    if num_head_nodes > 0
                    else 0
                )
                in_child = Dropout(self.dropout)(
                    Reshape((num_head_nodes, size, self.dims[stage]))(tree[child_slot])
                )
                # find out-nodes
                child_slot = child_slot + 1
                size = (
                    self.neighbourhood_sizes[child_slot] // num_head_nodes
                    if num_head_nodes > 0
                    else 0
                )
                out_child = Dropout(self.dropout)(
                    Reshape((num_head_nodes, size, self.dims[stage]))(tree[child_slot])
                )
                # aggregate neighbourhoods
                new_tree[slot] = self._aggs[stage]([parent, in_child, out_child])
            return new_tree

        if not isinstance(xin, list):
            raise TypeError("Input features to GraphSAGE must be a list")

        if len(xin) != self.max_slots:
            raise ValueError(
                "Number of input tensors does not match number of GraphSAGE layers"
            )

        # Combine GraphSAGE layers in stages
        stage_tree = xin
        for stage in range(self.max_hops):
            stage_tree = aggregate_neighbours(stage_tree, stage)
        out_layer = stage_tree[0]

        # Remove neighbourhood dimension from output tensors of the stack
        if K.int_shape(out_layer)[1] == 1:
            out_layer = Reshape(K.int_shape(out_layer)[2:])(out_layer)
        return self._normalization(out_layer)

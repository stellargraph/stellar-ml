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
Heterogeneous GraphSAGE and compatible aggregator layers

"""
__all__ = ["HinSAGE", "MeanHinAggregator"]

from keras.engine.topology import Layer
from keras import backend as K, Input
from keras.layers import Lambda, Dropout, Reshape
from keras.utils import Sequence
from keras import activations
from typing import List, Callable, Tuple, Dict, Union, AnyStr
import itertools as it
import operator as op


class MeanHinAggregator(Layer):
    """Mean Aggregator for HinSAGE implemented with Keras base layer

    Args:
        output_dim (int): Output dimension
        bias (bool): Use bias in layer or not (Default False)
        act (Callable or str): name of the activation function to use (must be a Keras
            activation function), or alternatively, a TensorFlow operation.
    """

    def __init__(
        self,
        output_dim: int = 0,
        bias: bool = False,
        act: Union[Callable, AnyStr] = "relu",
        **kwargs
    ):
        self.output_dim = output_dim
        assert output_dim % 2 == 0
        self.half_output_dim = int(output_dim / 2)
        self.has_bias = bias
        self.act = activations.get(act)
        self.nr = None
        self.w_neigh = []
        self.w_self = None
        self.bias = None
        self._initializer = "glorot_uniform"
        super().__init__(**kwargs)

    def get_config(self):
        """
        Gets class configuration for Keras serialization

        """
        config = {
            "output_dim": self.output_dim,
            "bias": self.has_bias,
            "act": activations.serialize(self.act),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def build(self, input_shape):
        """
        Builds layer

        Args:
            input_shape (list of list of int): Shape of input per neighbour type.

        """
        # Weight matrix for each type of neighbour
        # If there are no neighbours (input_shape[x][2]) for an input
        # then do not create weights as they are not used.
        self.nr = len(input_shape) - 1
        self.w_neigh = [
            self.add_weight(
                name="w_neigh_" + str(r),
                shape=(input_shape[1 + r][3], self.half_output_dim),
                initializer=self._initializer,
                trainable=True,
            )
            if input_shape[1 + r][2] > 0
            else None
            for r in range(self.nr)
        ]

        # Weight matrix for self
        self.w_self = self.add_weight(
            name="w_self",
            shape=(input_shape[0][2], self.half_output_dim),
            initializer=self._initializer,
            trainable=True,
        )

        # Optional bias
        if self.has_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.output_dim],
                initializer="zeros",
                trainable=True,
            )

        super().build(input_shape)

    def call(self, x, **kwargs):
        """
        Apply MeanAggregation on input tensors, x

        Args:
          x: List of Keras Tensors with the following elements

            - x[0]: tensor of self features shape (n_batch, n_head, n_feat)
            - x[1+r]: tensors of neighbour features each of shape (n_batch, n_head, n_neighbour[r], n_feat[r])

        Returns:
            Keras Tensor representing the aggregated embeddings in the input.

        """
        # Calculate the mean vectors over the neigbours of each relation (edge) type
        neigh_agg_by_relation = []
        for r in range(self.nr):
            # The neighbour input tensors for relation r
            z = x[1 + r]

            # If there are neighbours aggregate over them
            if z.shape[2] > 0:
                z_agg = K.dot(K.mean(z, axis=2), self.w_neigh[r])

            # Otherwise add a synthetic zero vector
            else:
                z_shape = K.shape(z)
                w_shape = self.half_output_dim
                z_agg = K.zeros((z_shape[0], z_shape[1], w_shape))

            neigh_agg_by_relation.append(z_agg)

        # Calculate the self vector shape (n_batch, n_head, n_out_self)
        from_self = K.dot(x[0], self.w_self)

        # Sum the contributions from all neighbour averages shape (n_batch, n_head, n_out_neigh)
        from_neigh = sum(neigh_agg_by_relation) / self.nr

        # Concatenate self + neighbour features, shape (n_batch, n_head, n_out)
        total = K.concatenate(
            [from_self, from_neigh], axis=2
        )  # YT: this corresponds to concat=Partial
        # TODO: implement concat=Full and concat=False

        return self.act((total + self.bias) if self.has_bias else total)

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.
        Assumes that the layer will be built to match that input shape provided.

        Args:
            input_shape (tuple of ints)
                Shape tuples can include `None` for free dimensions, instead of an integer.

        Returns:
            An input shape tuple.
        """
        return input_shape[0][0], input_shape[0][1], self.output_dim


class HinSAGE:
    """
    Implementation of the GraphSAGE algorithm extended for heterogeneous graphs with Keras layers.

    """

    def __init__(
        self,
        layer_sizes,
        generator=None,
        n_samples=None,
        input_neighbor_tree=None,
        input_dim=None,
        aggregator=None,
        bias=True,
        dropout=0.0,
        normalize="l2",
    ):
        """
        Args:
            layer_sizes (list): Hidden feature dimensions for each layer
            mapper (Sequence): A HinSAGENodeMapper or HinSAGELinkMapper. If specified the n_samples,
                input_neighbour_tree and input_dim will be taken from this object.
            n_samples: (Optional: needs to be specified if no mapper is provided.)
                The number of samples per layer in the model.
            input_neighbor_tree: A list of (node_type, [children]) tuples that specify the
                subtree to be created by the HinSAGE model.
            input_dim: The input dimensions for each node type as a dictionary of the form
                {node_type: feature_size}.
            aggregator: The HinSAGE aggregator to use. Defaults to the `MeanHinAggregator`.
            bias: If True a bias vector is learnt for each layer in the HinSAGE model
            dropout: The dropout supplied to each layer in the HinSAGE model.
            normalize: The normalization used after each layer, defaults to L2 normalization.
        """

        def eval_neigh_tree_per_layer(input_tree):
            """
            Function to evaluate the neighbourhood tree structure for every layer. The tree
            structure at each layer is a truncated version of the previous layer.

            Args:
              input_tree: Neighbourhood tree for the input batch

            Returns:
              List of neighbourhood trees

            """
            reduced = [
                li
                for li in input_tree
                if all(li_neigh < len(input_tree) for li_neigh in li[1])
            ]
            return (
                [input_tree]
                if len(reduced) == 0
                else [input_tree] + eval_neigh_tree_per_layer(reduced)
            )

        # Set the aggregator layer used in the model
        if aggregator is None:
            self._aggregator = MeanHinAggregator
        elif issubclass(aggregator, Layer):
            self._aggregator = aggregator
        else:
            raise TypeError("Aggregator should be a subclass of Keras Layer")

        # Set the normalization layer used in the model
        if normalize == "l2":
            self._normalization = Lambda(lambda x: K.l2_normalize(x, axis=2))

        elif normalize is None or normalize == "none" or normalize == "None":
            self._normalization = Lambda(lambda x: x)

        else:
            raise ValueError(
                "Normalization should be either 'l2' or 'none'; received '{}'".format(
                    normalize
                )
            )

        # Get the sampling tree, input_dim, and num_samples from the mapper if it is given
        # Use both the schema and head node type from the mapper
        # TODO: Refactor the horror of generator.generator.graph...
        if generator is not None:
            self.n_samples = generator.generator.num_samples
            self.subtree_schema = generator.generator.schema.type_adjacency_list(
                generator.head_node_types, len(self.n_samples)
            )
            self.input_dims = generator.generator.graph.node_feature_sizes()

        elif (
            input_neighbor_tree is not None
            and n_samples is not None
            and input_dim is not None
        ):
            self.subtree_schema = input_neighbor_tree
            self.n_samples = n_samples
            self.input_dims = input_dim

        else:
            raise RuntimeError(
                "If mapper is not provided, input_neighbour_tree, n_samples,"
                " and input_dim must be specified."
            )

        # Set parameters for the model
        self.n_layers = len(self.n_samples)
        self.bias = bias
        self.dropout = dropout

        # Neighbourhood info per layer
        self.neigh_trees = eval_neigh_tree_per_layer(
            [li for li in self.subtree_schema if len(li[1]) > 0]
        )

        # Depth of each input tensor i.e. number of hops from root nodes
        self._depths = [
            self.n_layers
            + 1
            - sum([1 for li in [self.subtree_schema] + self.neigh_trees if i < len(li)])
            for i in range(len(self.subtree_schema))
        ]

        # Dict of {node type: dimension} per layer
        self.dims = [
            dim
            if isinstance(dim, dict)
            else {k: dim for k, _ in ([self.subtree_schema] + self.neigh_trees)[layer]}
            for layer, dim in enumerate([self.input_dims] + layer_sizes)
        ]

        # Dict of {node type: aggregator} per layer
        self._aggs = [
            {
                node_type: self._aggregator(
                    output_dim,
                    bias=self.bias,
                    act="relu" if layer < self.n_layers - 1 else "linear",
                )
                for node_type, output_dim in self.dims[layer + 1].items()
            }
            for layer in range(self.n_layers)
        ]

    def __call__(self, xin: List):
        """
        Apply aggregator layers

        Args:
            x (list of Tensor): Batch input features

        Returns:
            Output tensor
        """

        def apply_layer(x: List, layer: int):
            """
            Compute the list of output tensors for a single HinSAGE layer

            Args:
                x (List[Tensor]): Inputs to the layer
                layer (int): Layer index

            Returns:
                Outputs of applying the aggregators as a list of Tensors

            """
            layer_out = []
            for i, (node_type, neigh_indices) in enumerate(self.neigh_trees[layer]):
                # The shape of the head node is used for reshaping the neighbour inputs
                head_shape = K.int_shape(x[i])[1]

                # Aplly dropout and reshape neighbours per node per layer
                neigh_list = [
                    Dropout(self.dropout)(
                        Reshape(
                            (
                                head_shape,
                                self.n_samples[self._depths[i]],
                                self.dims[layer][self.subtree_schema[neigh_index][0]],
                            )
                        )(x[neigh_index])
                    )
                    for neigh_index in neigh_indices
                ]

                # Apply dropout to head inputs
                x_head = Dropout(self.dropout)(x[i])

                # Apply aggregator to head node and reshaped neighbour nodes
                layer_out.append(self._aggs[layer][node_type]([x_head] + neigh_list))

            return layer_out

        # Form HinSAGE layers iteratively
        self.layer_tensors = []
        h_layer = xin
        for layer in range(0, self.n_layers):
            h_layer = apply_layer(h_layer, layer)
            self.layer_tensors.append(h_layer)

        # Return final layer output tensor with optional normalization
        return (
            self._normalization(h_layer[0])
            if len(h_layer) == 1
            else [self._normalization(xi) for xi in h_layer]
        )

    def _input_shapes(self) -> List[Tuple[int, int]]:
        """
        Returns the input shapes for the tensors of the supplied neighbourhood type tree

        Returns:
            A list of tuples giving the shape (number of nodes, feature size) for
            the corresponding item in the neighbourhood type tree (self.subtree_schema)
        """
        neighbor_sizes = list(it.accumulate([1] + self.n_samples, op.mul))

        def get_shape(stree, cnode, level=0):
            adj = stree[cnode][1]
            size_dict = {
                cnode: (neighbor_sizes[level], self.input_dims[stree[cnode][0]])
            }
            if len(adj) > 0:
                size_dict.update(
                    {
                        k: s
                        for a in adj
                        for k, s in get_shape(stree, a, level + 1).items()
                    }
                )
            return size_dict

        input_shapes = dict()
        for ii in range(len(self.subtree_schema)):
            input_shapes_ii = get_shape(self.subtree_schema, ii)
            # Update input_shapes if input_shapes_ii.keys() are not already in input_shapes.keys():
            if (
                len(set(input_shapes_ii.keys()).intersection(set(input_shapes.keys())))
                == 0
            ):
                input_shapes.update(input_shapes_ii)

        return [input_shapes[ii] for ii in range(len(self.subtree_schema))]

    def default_model(self, flatten_output=False):
        """
        Return model with default inputs

        Args:
            flatten_output (bool): The HinSAGE model returns an output tensor
                of form (batch_size, 1, feature_size) -
                if this flag is True, the output will be resized to
                (batch_size, feature_size)

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a list of Keras input tensors
            for the specified HinSAGE model and ``x_out`` is tne Keras tensor
            for the HinSAGE model output.

        """
        # Create tensor inputs
        x_inp = [Input(shape=s) for s in self._input_shapes()]

        # Output from GraphSAGE model
        x_out = self(x_inp)

        if flatten_output:
            x_out = Reshape((-1,))(x_out)

        return x_inp, x_out

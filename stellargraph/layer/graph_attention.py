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
Definition of Graph Attention Network (GAT) layer, and GAT class that is a stack of GAT layers
"""
__all__ = ["GraphAttention", "GraphAttentionSparse", "GAT"]

import warnings
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Input, Layer, Dropout, LeakyReLU, Lambda, Reshape

from ..mapper import FullBatchNodeGenerator, FullBatchGenerator
from .misc import SqueezedSparseConversion


class GraphAttention(Layer):
    """
    Graph Attention (GAT) layer. The base implementation is taken from
    https://github.com/danielegrattarola/keras-gat,
    with some modifications added for ease of use.

    Based on the original paper: Graph Attention Networks. P. Velickovic et al. ICLR 2018 https://arxiv.org/abs/1710.10903

    Notes:
      - The inputs are tensors with a batch dimension of 1:
        Keras requires this batch dimension, and for full-batch methods
        we only have a single "batch".

      - There are three inputs required, the node features, the output
        indices (the nodes that are to be selected in the final layer)
        and the graph adjacency matrix

      - This does not add self loops to the adjacency matrix, you should preprocess
        the adjacency matrix to add self-loops

      - The output indices are used when ``final_layer=True`` and the returned outputs
        are the final-layer features for the nodes indexed by output indices.

      - If ``final_layer=False`` all the node features are output in the same ordering as
        given by the adjacency matrix.

    Args:
        F_out (int): dimensionality of output feature vectors
        attn_heads (int or list of int): number of attention heads
        attn_heads_reduction (str): reduction applied to output features of each attention head, 'concat' or 'average'.
            'Average' should be applied in the final prediction layer of the model (Eq. 6 of the paper).
        in_dropout_rate (float): dropout rate applied to features
        attn_dropout_rate (float): dropout rate applied to attention coefficients
        activation (str): nonlinear activation applied to layer's output to obtain output features (eq. 4 of the GAT paper)
        final_layer (bool): If False the layer returns output for all nodes,
                            if True it returns the subset specified by the indices passed to it.
        use_bias (bool): toggles an optional bias
        saliency_map_support (bool): If calculating saliency maps using the tools in
            stellargraph.interpretability.saliency_maps this should be True. Otherwise this should be False (default).
        kernel_initializer (str or func): The initialiser to use for the head weights;
            defaults to 'glorot_uniform'.
        kernel_regularizer (str or func): The regulariser to use for the head weights;
            defaults to None.
        kernel_constraint (str or func): The constraint to use for the head weights;
            defaults to None.
        bias_initializer (str or func): The initialiser to use for the head bias;
            defaults to 'zeros'.
        bias_regularizer (str or func): The regulariser to use for the head bias;
            defaults to None.
        bias_constraint (str or func): The constraint to use for the head bias;
            defaults to None.
        attn_kernel_initializer (str or func): The initialiser to use for the attention weights;
            defaults to 'glorot_uniform'.
        attn_kernel_regularizer (str or func): The regulariser to use for the attention weights;
            defaults to None.
        attn_kernel_constraint (str or func): The constraint to use for the attention weights;
            defaults to None.
    """

    def __init__(
        self,
        units,
        attn_heads=1,
        attn_heads_reduction="concat",  # {'concat', 'average'}
        in_dropout_rate=0.0,
        attn_dropout_rate=0.0,
        activation="relu",
        use_bias=True,
        final_layer=False,
        saliency_map_support=False,
        **kwargs,
    ):

        if attn_heads_reduction not in {"concat", "average"}:
            raise ValueError(
                "{}: Possible heads reduction methods: concat, average; received {}".format(
                    type(self).__name__, attn_heads_reduction
                )
            )

        self.units = units  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.in_dropout_rate = in_dropout_rate  # dropout rate for node features
        self.attn_dropout_rate = attn_dropout_rate  # dropout rate for attention coefs
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias
        self.final_layer = final_layer

        self.saliency_map_support = saliency_map_support
        # Populated by build()
        self.kernels = []  # Layer kernels for attention heads
        self.biases = []  # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == "concat":
            # Output will have shape (..., K * F')
            self.output_dim = self.units * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.units

        self._get_regularisers_from_keywords(kwargs)

        super().__init__(**kwargs)

    def _get_regularisers_from_keywords(self, kwargs):
        self.kernel_initializer = initializers.get(
            kwargs.pop("kernel_initializer", "glorot_uniform")
        )
        self.kernel_regularizer = regularizers.get(
            kwargs.pop("kernel_regularizer", None)
        )
        self.kernel_constraint = constraints.get(kwargs.pop("kernel_constraint", None))

        self.bias_initializer = initializers.get(
            kwargs.pop("bias_initializer", "zeros")
        )
        self.bias_regularizer = regularizers.get(kwargs.pop("bias_regularizer", None))
        self.bias_constraint = constraints.get(kwargs.pop("bias_constraint", None))

        self.attn_kernel_initializer = initializers.get(
            kwargs.pop("attn_kernel_initializer", "glorot_uniform")
        )
        self.attn_kernel_regularizer = regularizers.get(
            kwargs.pop("attn_kernel_regularizer", None)
        )
        self.attn_kernel_constraint = constraints.get(
            kwargs.pop("attn_kernel_constraint", None)
        )

    def get_config(self):
        """
        Gets class configuration for Keras serialization

        """
        config = {
            "units": self.units,
            "attn_heads": self.attn_heads,
            "attn_heads_reduction": self.attn_heads_reduction,
            "in_dropout_rate": self.in_dropout_rate,
            "attn_dropout_rate": self.attn_dropout_rate,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "final_layer": self.final_layer,
            "saliency_map_support": self.saliency_map_support,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "attn_kernel_initializer": initializers.serialize(
                self.attn_kernel_initializer
            ),
            "attn_kernel_regularizer": regularizers.serialize(
                self.attn_kernel_regularizer
            ),
            "attn_kernel_constraint": constraints.serialize(
                self.attn_kernel_constraint
            ),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        """
        Computes the output shape of the layer.
        Assumes the following inputs:

        Args:
            input_shapes (tuple of ints)
                Shape tuples can include None for free dimensions, instead of an integer.

        Returns:
            An input shape tuple.
        """
        feature_shape, out_shape, *As_shapes = input_shapes

        batch_dim = feature_shape[0]
        if self.final_layer:
            out_dim = out_shape[1]
        else:
            out_dim = feature_shape[1]

        return batch_dim, out_dim, self.output_dim

    def build(self, input_shapes):
        """
        Builds the layer

        Args:
            input_shapes (list of int): shapes of the layer's inputs (node features and adjacency matrix)

        """
        feat_shape = input_shapes[0]
        input_dim = int(feat_shape[-1])

        # Variables to support integrated gradients
        self.delta = self.add_weight(
            name="ig_delta", shape=(), trainable=False, initializer=initializers.ones()
        )
        self.non_exist_edge = self.add_weight(
            name="ig_non_exist_edge",
            shape=(),
            trainable=False,
            initializer=initializers.zeros(),
        )

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(
                shape=(input_dim, self.units),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name="kernel_{}".format(head),
            )
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(
                    shape=(self.units,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    name="bias_{}".format(head),
                )
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(
                shape=(self.units, 1),
                initializer=self.attn_kernel_initializer,
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint,
                name="attn_kernel_self_{}".format(head),
            )
            attn_kernel_neighs = self.add_weight(
                shape=(self.units, 1),
                initializer=self.attn_kernel_initializer,
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint,
                name="attn_kernel_neigh_{}".format(head),
            )
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs):
        """
        Creates the layer as a Keras graph.

        Note that the inputs are tensors with a batch dimension of 1:
        Keras requires this batch dimension, and for full-batch methods
        we only have a single "batch".

        There are three inputs required, the node features, the output
        indices (the nodes that are to be selected in the final layer)
        and the graph adjacency matrix

        Notes:
            This does not add self loops to the adjacency matrix.
            The output indices are only used when ``final_layer=True``

        Args:
            inputs (list): list of inputs with 3 items:
            node features (size 1 x N x F),
            output indices (size 1 x M),
            graph adjacency matrix (size N x N),
            where N is the number of nodes in the graph,
                  F is the dimensionality of node features
                  M is the number of output nodes
        """
        X = inputs[0]  # Node features (1 x N x F)
        out_indices = inputs[1]  # output indices (1 x K)
        A = inputs[2]  # Adjacency matrix (N x N)
        N = K.int_shape(A)[-1]

        batch_dim, n_nodes, _ = K.int_shape(X)
        if batch_dim != 1:
            raise ValueError(
                "Currently full-batch methods only support a batch dimension of one"
            )

        else:
            # Remove singleton batch dimension
            X = K.squeeze(X, 0)
            out_indices = K.squeeze(out_indices, 0)

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[
                head
            ]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(
                features, attention_kernel[0]
            )  # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(
                features, attention_kernel[1]
            )  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            dense = attn_for_self + K.transpose(
                attn_for_neighs
            )  # (N x N) via broadcasting

            # Add nonlinearity
            dense = LeakyReLU(alpha=0.2)(dense)

            # Mask values before activation (Vaswani et al., 2017)
            # YT: this only works for 'binary' A, not for 'weighted' A!
            # YT: if A does not have self-loops, the node itself will be masked, so A should have self-loops
            # YT: this is ensured by setting the diagonal elements of A tensor to 1 above
            if not self.saliency_map_support:
                mask = -10e9 * (1.0 - A)
                dense += mask
                dense = K.softmax(dense)  # (N x N), Eq. 3 of the paper

            else:
                # dense = dense - tf.reduce_max(dense)
                # GAT with support for saliency calculations
                W = (self.delta * A) * K.exp(
                    dense - K.max(dense, axis=1, keepdims=True)
                ) * (1 - self.non_exist_edge) + self.non_exist_edge * (
                    A + self.delta * (tf.ones((N, N)) - A) + tf.eye(N)
                ) * K.exp(
                    dense - K.max(dense, axis=1, keepdims=True)
                )
                dense = W / K.sum(W, axis=1, keepdims=True)

            # Apply dropout to features and attention coefficients
            dropout_feat = Dropout(self.in_dropout_rate)(features)  # (N x F')
            dropout_attn = Dropout(self.attn_dropout_rate)(dense)  # (N x N)

            # Linear combination with neighbors' features [YT: see Eq. 4]
            node_features = K.dot(dropout_attn, dropout_feat)  # (N x F')

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == "concat":
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')

        # Nonlinear activation function
        output = self.activation(output)

        # On the final layer we gather the nodes referenced by the indices
        if self.final_layer:
            output = K.gather(output, out_indices)

        # Add batch dimension back if we removed it
        if batch_dim == 1:
            output = K.expand_dims(output, 0)

        return output


class GraphAttentionSparse(GraphAttention):
    """
    Graph Attention (GAT) layer, base implementation taken from https://github.com/danielegrattarola/keras-gat,
    some modifications added for ease of use.

    Based on the original paper: Graph Attention Networks. P. Velickovic et al. ICLR 2018 https://arxiv.org/abs/1710.10903

    Notes:
      - The inputs are tensors with a batch dimension of 1:
        Keras requires this batch dimension, and for full-batch methods
        we only have a single "batch".

      - There are three inputs required, the node features, the output
        indices (the nodes that are to be selected in the final layer),
        and the graph adjacency matrix

      - This does not add self loops to the adjacency matrix, you should preprocess
        the adjacency matrix to add self-loops

      - The output indices are used when `final_layer=True` and the returned outputs
        are the final-layer features for the nodes indexed by output indices.

      - If `final_layer=False` all the node features are output in the same ordering as
        given by the adjacency matrix.


    Args:
        F_out (int): dimensionality of output feature vectors
        attn_heads (int or list of int): number of attention heads
        attn_heads_reduction (str): reduction applied to output features of each attention head, 'concat' or 'average'.
            'Average' should be applied in the final prediction layer of the model (Eq. 6 of the paper).
        in_dropout_rate (float): dropout rate applied to features
        attn_dropout_rate (float): dropout rate applied to attention coefficients
        activation (str): nonlinear activation applied to layer's output to obtain output features (eq. 4 of the GAT paper)
        final_layer (bool): If False the layer returns output for all nodes,
                            if True it returns the subset specified by the indices passed to it.
        use_bias (bool): toggles an optional bias
        saliency_map_support (bool): If calculating saliency maps using the tools in
            stellargraph.interpretability.saliency_maps this should be True. Otherwise this should be False (default).
        kernel_initializer (str or func): The initialiser to use for the head weights;
            defaults to 'glorot_uniform'.
        kernel_regularizer (str or func): The regulariser to use for the head weights;
            defaults to None.
        kernel_constraint (str or func): The constraint to use for the head weights;
            defaults to None.
        bias_initializer (str or func): The initialiser to use for the head bias;
            defaults to 'zeros'.
        bias_regularizer (str or func): The regulariser to use for the head bias;
            defaults to None.
        bias_constraint (str or func): The constraint to use for the head bias;
            defaults to None.
        attn_kernel_initializer (str or func): The initialiser to use for the attention weights;
            defaults to 'glorot_uniform'.
        attn_kernel_regularizer (str or func): The regulariser to use for the attention weights;
            defaults to None.
        attn_kernel_constraint (str or func): The constraint to use for the attention weights;
            defaults to None.
    """

    def call(self, inputs, **kwargs):
        """
        Creates the layer as a Keras graph

        Notes:
            This does not add self loops to the adjacency matrix.
            The output indices are only used when `final_layer=True`

        Args:
            inputs (list): list of inputs with 4 items:
            node features (size b x N x F),
            output indices (size b x M),
            sparse graph adjacency matrix (size N x N),
            where N is the number of nodes in the graph,
                  F is the dimensionality of node features
                  M is the number of output nodes
        """
        X = inputs[0]  # Node features (1 x N x F)
        out_indices = inputs[1]  # output indices (1 x K)
        A_sparse = inputs[2]  # Adjacency matrix (1 x N x N)

        if not isinstance(A_sparse, tf.SparseTensor):
            raise TypeError("A is not sparse")

        # Get undirected graph edges (E x 2)
        A_indices = A_sparse.indices

        batch_dim, n_nodes, _ = K.int_shape(X)
        if batch_dim != 1:
            raise ValueError(
                "Currently full-batch methods only support a batch dimension of one"
            )
        else:
            # Remove singleton batch dimension
            out_indices = K.squeeze(out_indices, 0)
            X = K.squeeze(X, 0)

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[
                head
            ]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_j]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(
                features, attention_kernel[0]
            )  # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(
                features, attention_kernel[1]
            )  # (N x 1), [a_2]^T [Wh_j]

            # Create sparse attention vector (All non-zero values of the matrix)
            sparse_attn_self = tf.gather(
                K.reshape(attn_for_self, [-1]), A_indices[:, 0], axis=0
            )
            sparse_attn_neighs = tf.gather(
                K.reshape(attn_for_neighs, [-1]), A_indices[:, 1], axis=0
            )
            attn_values = sparse_attn_self + sparse_attn_neighs

            # Add nonlinearity
            attn_values = LeakyReLU(alpha=0.2)(attn_values)

            # Apply dropout to features and attention coefficients
            dropout_feat = Dropout(self.in_dropout_rate)(features)  # (N x F')
            dropout_attn = Dropout(self.attn_dropout_rate)(attn_values)  # (N x N)

            # Convert to sparse matrix
            sparse_attn = tf.sparse.SparseTensor(
                A_indices, values=dropout_attn, dense_shape=[n_nodes, n_nodes]
            )

            # Apply softmax to get attention coefficients
            sparse_attn = tf.sparse.softmax(sparse_attn)  # (N x N), Eq. 3 of the paper

            # Linear combination with neighbors' features [YT: see Eq. 4]
            node_features = tf.sparse.sparse_dense_matmul(
                sparse_attn, dropout_feat
            )  # (N x F')

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == "concat":
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')

        output = self.activation(output)

        # On the final layer we gather the nodes referenced by the indices
        if self.final_layer:
            output = K.gather(output, out_indices)

        # Add batch dimension back if we removed it
        if batch_dim == 1:
            output = K.expand_dims(output, 0)
        return output


class GAT:
    """
    A stack of Graph Attention (GAT) layers with aggregation of multiple attention heads,
    Eqs 5-6 of the GAT paper https://arxiv.org/abs/1710.10903

    To use this class as a Keras model, the features and pre-processed adjacency matrix
    should be supplied using either the :class:`FullBatchNodeGenerator` class for node inference
    or the :class:`FullBatchLinkGenerator` class for link inference.

    To have the appropriate pre-processing the generator object should be instanciated
    with the `method='gat'` argument.

    Examples:
        Creating a GAT node classification model from an existing :class:`StellarGraph` object `G`::

            generator = FullBatchNodeGenerator(G, method="gat")
            gat = GAT(
                    layer_sizes=[8, 4],
                    activations=["elu","softmax"],
                    attn_heads=8,
                    generator=generator,
                    in_dropout=0.5,
                    attn_dropout=0.5,
                )
            x_inp, predictions = gat.node_model()

    For more details, please see the GAT demo notebook:
    demos/node-classification/gat/gat-cora-node-classification-example.ipynb

    Notes:
      - The inputs are tensors with a batch dimension of 1. These are provided by the \
        :class:`FullBatchNodeGenerator` object.

      - This does not add self loops to the adjacency matrix, you should preprocess
        the adjacency matrix to add self-loops, using the ``method='gat'`` argument
        of the :class:`FullBatchNodeGenerator`.

      - The nodes provided to the :class:`FullBatchNodeGenerator.flow` method are
        used by the final layer to select the predictions for those nodes in order.
        However, the intermediate layers before the final layer order the nodes
        in the same way as the adjacency matrix.

    Args:
        layer_sizes (list of int): list of output sizes of GAT layers in the stack. The length of this list defines
            the number of GraphAttention layers in the stack.
        generator (FullBatchNodeGenerator): an instance of FullBatchNodeGenerator class constructed on the graph of interest
        attn_heads (int or list of int): number of attention heads in GraphAttention layers. The options are:

            - a single integer: the passed value of ``attn_heads`` will be applied to all GraphAttention layers in the stack, except the last layer (for which the number of attn_heads will be set to 1).
            - a list of integers: elements of the list define the number of attention heads in the corresponding layers in the stack.

        attn_heads_reduction (list of str or None): reductions applied to output features of each attention head,
            for all layers in the stack. Valid entries in the list are {'concat', 'average'}.
            If None is passed, the default reductions are applied: 'concat' reduction to all layers in the stack
            except the final layer, 'average' reduction to the last layer (Eqs. 5-6 of the GAT paper).
        bias (bool): toggles an optional bias in GAT layers
        in_dropout (float): dropout rate applied to input features of each GAT layer
        attn_dropout (float): dropout rate applied to attention maps
        normalize (str or None): normalization applied to the final output features of the GAT layers stack. Default is None.
        activations (list of str): list of activations applied to each layer's output; defaults to ['elu', ..., 'elu'].
        saliency_map_support (bool): If calculating saliency maps using the tools in
            stellargraph.interpretability.saliency_maps this should be True. Otherwise this should be False (default).
        kernel_regularizer (str or func): The regulariser to use for the head weights;
            defaults to None.
        attn_kernel_regularizer (str or func): The regulariser to use for the attention weights;
            defaults to None.
    """

    def __init__(
        self,
        layer_sizes,
        generator=None,
        attn_heads=1,
        attn_heads_reduction=None,
        bias=True,
        in_dropout=0.0,
        attn_dropout=0.0,
        normalize=None,
        activations=None,
        saliency_map_support=False,
        **kwargs,
    ):
        self.bias = bias
        self.in_dropout = in_dropout
        self.attn_dropout = attn_dropout
        self.generator = generator
        self.saliency_map_support = saliency_map_support

        # Check layer_sizes (must be list of int):
        # check type:
        if not isinstance(layer_sizes, list):
            raise TypeError(
                "{}: layer_sizes should be a list of integers; received type {} instead.".format(
                    type(self).__name__, type(layer_sizes).__name__
                )
            )
        # check that values are valid:
        elif not all([isinstance(s, int) and s > 0 for s in layer_sizes]):
            raise ValueError(
                "{}: all elements in layer_sizes should be positive integers!".format(
                    type(self).__name__
                )
            )
        self.layer_sizes = layer_sizes
        n_layers = len(layer_sizes)

        # Check attn_heads (must be int or list of int):
        if isinstance(attn_heads, list):
            # check the length
            if not len(attn_heads) == n_layers:
                raise ValueError(
                    "{}: length of attn_heads list ({}) should match the number of GAT layers ({})".format(
                        type(self).__name__, len(attn_heads), n_layers
                    )
                )
            # check that values in the list are valid
            if not all([isinstance(a, int) and a > 0 for a in attn_heads]):
                raise ValueError(
                    "{}: all elements in attn_heads should be positive integers!".format(
                        type(self).__name__
                    )
                )
            self.attn_heads = attn_heads  # (list of int as passed by the user)

        elif isinstance(attn_heads, int):
            self.attn_heads = list()
            for l, _ in enumerate(layer_sizes):
                # number of attention heads for layer l: attn_heads (int) for all but the last layer (for which it's set to 1)
                self.attn_heads.append(attn_heads if l < n_layers - 1 else 1)

        else:
            raise TypeError(
                "{}: attn_heads should be an integer or a list of integers!".format(
                    type(self).__name__
                )
            )

        # Check attn_heads_reduction (list of str, or None):
        if attn_heads_reduction is None:
            # set default head reductions, see eqs 5-6 of the GAT paper
            self.attn_heads_reduction = ["concat"] * (n_layers - 1) + ["average"]
        else:
            # user-specified list of head reductions (valid entries are 'concat' and 'average')
            # check type (must be a list of str):
            if not isinstance(attn_heads_reduction, list):
                raise TypeError(
                    "{}: attn_heads_reduction should be a string; received type {} instead.".format(
                        type(self).__name__, type(attn_heads_reduction).__name__
                    )
                )

            # check length of attn_heads_reduction list:
            if not len(attn_heads_reduction) == len(layer_sizes):
                raise ValueError(
                    "{}: length of attn_heads_reduction list ({}) should match the number of GAT layers ({})".format(
                        type(self).__name__, len(attn_heads_reduction), n_layers
                    )
                )

            # check that list elements are valid:
            if all(
                [ahr.lower() in {"concat", "average"} for ahr in attn_heads_reduction]
            ):
                self.attn_heads_reduction = attn_heads_reduction
            else:
                raise ValueError(
                    "{}: elements of attn_heads_reduction list should be either 'concat' or 'average'!".format(
                        type(self).__name__
                    )
                )

        # Check activations (list of str):
        # check type:
        if activations is None:
            activations = ["elu"] * n_layers
        if not isinstance(activations, list):
            raise TypeError(
                "{}: activations should be a list of strings; received {} instead".format(
                    type(self).__name__, type(activations)
                )
            )
        # check length:
        if not len(activations) == n_layers:
            raise ValueError(
                "{}: length of activations list ({}) should match the number of GAT layers ({})".format(
                    type(self).__name__, len(activations), n_layers
                )
            )
        self.activations = activations

        # Check generator and configure sparse adjacency matrix
        if generator is None:
            self.use_sparse = False
            self.multiplicity = kwargs.get("multiplicity", 1)
            self.n_nodes = kwargs.get("num_nodes", None)
            self.n_features = kwargs.get("num_features", None)

        else:
            if not isinstance(generator, FullBatchGenerator):
                raise TypeError(
                    "Generator should be a instance of FullBatchNodeGenerator or FullBatchLinkGenerator"
                )

            # Copy required information from generator
            self.use_sparse = generator.use_sparse
            self.multiplicity = generator.multiplicity
            self.n_nodes = generator.features.shape[0]
            self.n_features = generator.features.shape[1]

        if self.n_nodes is None or self.n_features is None:
            raise RuntimeError(
                "node_model: if generator is not provided to object constructor, num_nodes and feature_size must be specified."
            )

        # Set the normalization layer used in the model
        if normalize == "l2":
            self._normalization = Lambda(lambda x: K.l2_normalize(x, axis=2))

        elif normalize is None or str(normalize).lower() in {"none", "linear"}:
            self._normalization = Lambda(lambda x: x)

        else:
            raise ValueError(
                "Normalization should be either 'l2' or None (also allowed as 'none'); received '{}'".format(
                    normalize
                )
            )

        # Switch between sparse or dense model
        if self.use_sparse:
            self._gat_layer = GraphAttentionSparse
        else:
            self._gat_layer = GraphAttention

        # Optional regulariser, etc. for weights and biases
        self._get_regularisers_from_keywords(kwargs)

        # Initialize a stack of GAT layers
        self._layers = []
        n_layers = len(self.layer_sizes)
        for ii in range(n_layers):
            # Dropout on input node features before each GAT layer
            self._layers.append(Dropout(self.in_dropout))

            # GraphAttention layer
            self._layers.append(
                self._gat_layer(
                    units=self.layer_sizes[ii],
                    attn_heads=self.attn_heads[ii],
                    attn_heads_reduction=self.attn_heads_reduction[ii],
                    in_dropout_rate=self.in_dropout,
                    attn_dropout_rate=self.attn_dropout,
                    activation=self.activations[ii],
                    use_bias=self.bias,
                    final_layer=ii == (n_layers - 1),
                    saliency_map_support=self.saliency_map_support,
                    **self._regularisers,
                )
            )

    def _get_regularisers_from_keywords(self, kwargs):
        regularisers = {}
        for param_name in [
            "kernel_initializer",
            "kernel_regularizer",
            "kernel_constraint",
            "bias_initializer",
            "bias_regularizer",
            "bias_constraint",
            "attn_kernel_initializer",
            "attn_kernel_regularizer",
            "attn_kernel_constraint",
        ]:
            param_value = kwargs.pop(param_name, None)
            if param_value is not None:
                regularisers[param_name] = param_value
        self._regularisers = regularisers

    def __call__(self, inputs):
        """
        Apply a stack of GAT layers to the input x_inp

        Args:
            x_inp (Tensor): input of the 1st GAT layer in the stack

        Returns: Output tensor of the GAT layers stack

        """
        if not isinstance(inputs, list):
            raise TypeError(f"inputs: expected list, found {type(inputs).__name__}")

        x_in, out_indices, *As = inputs

        # Currently we require the batch dimension to be one for full-batch methods
        batch_dim, n_nodes, _ = K.int_shape(x_in)

        if batch_dim != 1:
            raise ValueError(
                "Currently full-batch methods only support a batch dimension of one"
            )

        # Convert input indices & values to a sparse matrix
        if self.use_sparse:
            A_indices, A_values = As
            Ainput = [
                SqueezedSparseConversion(shape=(n_nodes, n_nodes))(
                    [A_indices, A_values]
                )
            ]

        # Otherwise, create dense matrix from input tensor
        else:
            Ainput = [Lambda(lambda A: K.squeeze(A, 0))(A) for A in As]

        # TODO: Support multiple matrices?
        if len(Ainput) != 1:
            raise NotImplementedError(
                "The GAT method currently only accepts a single matrix"
            )

        # Remove singleton batch dimension
        h_layer = x_in
        for layer in self._layers:
            if isinstance(layer, self._gat_layer):
                # For a GAT layer add the matrix and output indices
                # Note that the output indices are only used if `final_layer=True`
                h_layer = layer([h_layer, out_indices] + Ainput)

            else:
                # For other (non-graph) layers only supply the input tensor
                h_layer = layer(h_layer)

            # print("Hlayer:", h_layer)

        return self._normalization(h_layer)

    def build(self, multiplicity=None):
        """
        Builds a GAT model for node or link prediction

        Returns:
            tuple: `(x_inp, x_out)`, where `x_inp` is a list of Keras/TensorFlow
            input tensors for the model and `x_out` is a tensor of the model output.
        """

        # Inputs for features
        x_t = Input(batch_shape=(1, self.n_nodes, self.n_features))

        # If not specified use multiplicity from instanciation
        if multiplicity is None:
            multiplicity = self.multiplicity

        # Indices to gather for model output
        if multiplicity == 1:
            out_indices_t = Input(batch_shape=(1, None), dtype="int32")
        else:
            out_indices_t = Input(batch_shape=(1, None, multiplicity), dtype="int32")

        # Create inputs for sparse or dense matrices
        if self.use_sparse:
            # Placeholders for the sparse adjacency matrix
            A_indices_t = Input(batch_shape=(1, None, 2), dtype="int64")
            A_values_t = Input(batch_shape=(1, None))
            A_placeholders = [A_indices_t, A_values_t]

        else:
            # Placeholders for the dense adjacency matrix
            A_m = Input(batch_shape=(1, self.n_nodes, self.n_nodes))
            A_placeholders = [A_m]

        # TODO: Support multiple matrices
        x_inp = [x_t, out_indices_t] + A_placeholders
        x_out = self(x_inp)

        # Flatten output by removing singleton batch dimension
        if x_out.shape[0] == 1:
            self.x_out_flat = Lambda(lambda x: K.squeeze(x, 0))(x_out)
        else:
            self.x_out_flat = x_out

        return x_inp, x_out

    def link_model(self):
        if self.multiplicity != 2:
            warnings.warn(
                "Link model requested but a generator not supporting links was supplied."
            )
        return self.build(multiplicity=2)

    def node_model(self):
        if self.multiplicity != 1:
            warnings.warn(
                "Node model requested but a generator not supporting nodes was supplied."
            )
        return self.build(multiplicity=1)

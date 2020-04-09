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

from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, constraints, regularizers
from tensorflow.keras.layers import Input, Layer, Lambda, Dropout, Reshape
from .misc import deprecated_model_function, GatherIndices
from ..mapper import ClusterNodeGenerator


class ClusterGraphConvolution(Layer):

    """
    Cluster Graph Convolution (GCN) Keras layer. A stack of such layers can be used to create a Cluster-GCN model.

    The implementation is based on the GCN Keras layer of keras-gcn github
    repo https://github.com/tkipf/keras-gcn

    Original paper: Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph
    Convolutional Networks, W. Chiang, X. Liu, S. Si, Y. Li, S. Bengio, and C. Hsieh,
    KDD, 2019, https://arxiv.org/abs/1905.07953

    Notes:
      - The inputs are tensors with a batch dimension of 1:
        Keras requires this batch dimension.

      - There are two inputs required, the node features,
        and the normalized graph adjacency matrix.

      - This class assumes that the normalized graph adjacency matrix is passed as
        input to the Keras methods.

    Args:
        units (int): dimensionality of output feature vectors
        activation (str): nonlinear activation applied to layer's output to obtain output features
        use_bias (bool): toggles an optional bias
        final_layer (bool): Deprecated, use ``tf.gather`` or :class:`GatherIndices`
        kernel_initializer (str): name of layer bias f the initializer for kernel parameters (weights)
        bias_initializer (str): name of the initializer for bias
        kernel_regularizer (str): name of regularizer to be applied to layer kernel. Must be a Keras regularizer.
        bias_regularizer (str): name of regularizer to be applied to layer bias. Must be a Keras regularizer.
        activity_regularizer (str): not used in the current implementation
        kernel_constraint (str): constraint applied to layer's kernel
        bias_constraint (str): constraint applied to layer's bias
        input_dim (int, optional): the size of the input shape, if known.
        kwargs: any additional arguments to pass to :class:`tensorflow.keras.layers.Layer`
    """

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        final_layer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        input_dim=None,
        **kwargs,
    ):
        if "input_shape" not in kwargs and input_dim is not None:
            kwargs["input_shape"] = (input_dim,)

        super().__init__(**kwargs)

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        if final_layer is not None:
            raise ValueError(
                "'final_layer' is not longer supported, use 'tf.gather' or 'GatherIndices' separately"
            )

    def get_config(self):
        """
        Gets class configuration for Keras serialization.
        Used by keras model serialization.

        Returns:
            A dictionary that contains the config of the layer
        """

        config = {
            "units": self.units,
            "use_bias": self.use_bias,
            "activation": activations.serialize(self.activation),
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
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
        feature_shape, *As_shapes = input_shapes

        batch_dim = feature_shape[0]
        out_dim = feature_shape[1]

        return batch_dim, out_dim, self.units

    def build(self, input_shapes):
        """
        Builds the layer

        Args:
            input_shapes (list of int): shapes of the layer's inputs (node features and adjacency matrix)

        """
        feat_shape = input_shapes[0]
        input_dim = int(feat_shape[-1])

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        """
        Applies the layer.

        Args:
            inputs (list): a list of 3 input tensors that includes
                node features (size 1 x N x F),
                graph adjacency matrix (size 1 x N x N),
                where N is the number of nodes in the graph, and
                F is the dimensionality of node features.

        Returns:
            Keras Tensor that represents the output of the layer.
        """
        features, *As = inputs

        # Remove singleton batch dimension
        features = K.squeeze(features, 0)

        # Calculate the layer operation of GCN multiplying the normalized adjacency matrix
        # width the node features matrix.
        A = As[0]
        h_graph = K.dot(A, features)
        output = K.dot(h_graph, self.kernel)

        # Add optional bias & apply activation
        if self.bias is not None:
            output += self.bias
        output = self.activation(output)
        output = K.expand_dims(output, 0)
        return output


class ClusterGCN:
    """
    A stack of Cluster Graph Convolutional layers that implement a cluster graph convolution network
    model as in https://arxiv.org/abs/1905.07953

    The model minimally requires specification of the layer sizes as a list of ints
    corresponding to the feature dimensions for each hidden layer,
    activation functions for each hidden layers, and a generator object.

    To use this class as a Keras model, the features and pre-processed adjacency matrix
    should be supplied using the :class:`ClusterNodeGenerator` class.

    For more details, please see the Cluster-GCN demo notebook:
    demos/node-classification/clustergcn/cluster-gcn-node-classification.ipynb

    Notes:
      - The inputs are tensors with a batch dimension of 1. These are provided by the \
        :class:`ClusterNodeGenerator` object.

      - The nodes provided to the :class:`ClusterNodeGenerator.flow` method are
        used by the final layer to select the predictions for those nodes in order.
        However, the intermediate layers before the final layer order the nodes
        in the same way as the adjacency matrix.

    Examples:
        Creating a Cluster-GCN node classification model from an existing :class:`StellarGraph`
        object ``G``::

            generator = ClusterNodeGenerator(G, clusters=10, q=2)
            cluster_gcn = ClusterGCN(
                             layer_sizes=[32, 4],
                             activations=["elu","softmax"],
                             generator=generator,
                             dropout=0.5
                )
            x_inp, predictions = cluster_gcn.in_out_tensors()

    Args:
        layer_sizes (list of int): list of output sizes of the graph convolutional layers in the stack
        activations (list of str): list of activations applied to each layer's output
        generator (ClusterNodeGenerator): an instance of ClusterNodeGenerator class constructed on the graph of interest
        bias (bool): toggles an optional bias in graph convolutional layers
        dropout (float): dropout rate applied to input features of each graph convolutional layer
        kernel_initializer (str or func, optional): The initialiser to use for the weights of each layer.
        kernel_regularizer (str or func, optional): The regulariser to use for the weights of each layer.
        kernel_constraint (str or func, optional): The constraint to use for the weights of each layer.
        bias_initializer (str or func, optional): The initialiser to use for the bias of each layer.
        bias_regularizer (str or func, optional): The regulariser to use for the bias of each layer.
        bias_constraint (str or func, optional): The constraint to use for the bias of each layer.
    """

    def __init__(
        self,
        layer_sizes,
        activations,
        generator,
        bias=True,
        dropout=0.0,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
    ):
        if not isinstance(generator, ClusterNodeGenerator):
            raise TypeError("Generator should be a instance of ClusterNodeGenerator")

        if len(layer_sizes) != len(activations):
            raise AssertionError(
                "The number of given layers should be the same as the number of activations."
                "However given len(layer_sizes): {} vs len(activations): {}".format(
                    len(layer_sizes), len(activations)
                )
            )

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.bias = bias
        self.dropout = dropout
        self.generator = generator
        self.support = 1

        # Initialize a stack of Cluster GCN layers
        n_layers = len(self.layer_sizes)
        self._layers = []
        for ii in range(n_layers):
            l = self.layer_sizes[ii]
            a = self.activations[ii]
            self._layers.append(Dropout(self.dropout))
            self._layers.append(
                ClusterGraphConvolution(
                    l,
                    activation=a,
                    use_bias=self.bias,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_initializer=bias_initializer,
                    bias_regularizer=bias_regularizer,
                    bias_constraint=bias_constraint,
                )
            )

    def __call__(self, x):
        """
        Apply a stack of Cluster GCN-layers to the inputs.
        The input tensors are expected to be a list of the following:
        [
            Node features shape (1, N, F),
            Adjacency indices (1, E, 2),
            Adjacency values (1, E),
            Output indices (1, O)
        ]
        where N is the number of nodes, F the number of input features,
              E is the number of edges, O the number of output nodes.

        Args:
            x (Tensor): input tensors

        Returns:
            Output tensor
        """
        x_in, out_indices, *As = x

        Ainput = [Lambda(lambda A: K.squeeze(A, 0))(A) for A in As]

        h_layer = x_in

        for layer in self._layers:
            if isinstance(layer, ClusterGraphConvolution):
                # For a GCN layer add the matrix
                h_layer = layer([h_layer] + Ainput)
            else:
                # For other (non-graph) layers only supply the input tensor
                h_layer = layer(h_layer)

        # only return data for the requested nodes
        h_layer = GatherIndices(batch_dims=1)([h_layer, out_indices])

        return h_layer

    def in_out_tensors(self):
        """
        Builds a Cluster-GCN model for node prediction.

        Returns:
            tuple: `(x_inp, x_out)`, where `x_inp` is a list of two input tensors for the
            Cluster-GCN model (containing node features and normalized adjacency matrix),
            and `x_out` is a tensor for the Cluster-GCN model output.
        """
        # Placeholder for node features
        N_feat = self.generator.feature_size

        # Inputs for features & target indices
        x_t = Input(batch_shape=(1, None, N_feat))
        out_indices_t = Input(batch_shape=(1, None), dtype="int32")

        # Placeholders for the dense adjacency matrix
        A_m = Input(batch_shape=(1, None, None))
        A_placeholders = [A_m]

        x_inp = [x_t, out_indices_t] + A_placeholders
        x_out = self(x_inp)

        return x_inp, x_out

    build = deprecated_model_function(in_out_tensors, "build")

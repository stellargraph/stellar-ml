# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
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
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from ..core import StellarGraph
from ..core.validation import require_integer_in_range
from .base import Generator
from scipy.sparse.linalg import eigs
from scipy.sparse import diags


class GraphWaveGenerator(Generator):
    """
    Implementation of the GraphWave structural embedding algorithm from the paper: "Learning Structural Node Embeddings
    via Diffusion Wavelets" (https://arxiv.org/pdf/1710.10321.pdf)

    This class is minimally with a StellarGraph object. Calling the flow function will return a TensorFlow
    DataSet that contains the GraphWave embeddings.

    This implementation differs from the paper by removing the automatic method of calculating scales. This method was
    found to not work well in practice, and replicating the results of the paper requires manually specifying much
    larger scales than those automatically calculated.

    .. seealso:: Example using this generator: `unsupervised representation learning <https://stellargraph.readthedocs.io/en/stable/demos/embeddings/graphwave-embeddings.html>`__.

    Args:
        G (StellarGraph): the StellarGraph object.
        scales (iterable of floats): the wavelet scales to use. Smaller values embed smaller scale structural
            features, and larger values embed larger structural features.
        degree: the degree of the Chebyshev polynomial to use. Higher degrees yield more accurate results but at a
            higher computational cost. According to [1], the default value of 20 is accurate enough for most
            applications.
        weighted (bool, optional): if True, use the edge weights from ``G``; if False, treat the
            graph as unweighted.

    [1] D. I. Shuman, P. Vandergheynst, and P. Frossard, “Chebyshev Polynomial Approximation for Distributed Signal
    Processing,” https://arxiv.org/abs/1105.1891
    """

    def __init__(self, G, scales=(5, 10), degree=20, weighted=False):

        if not isinstance(G, StellarGraph):
            raise TypeError("G must be a StellarGraph object.")

        # Check that there is only a single node type
        _ = G.unique_node_type(
            "G: expected a graph with a single node type, found a graph with node types: %(found)s"
        )

        require_integer_in_range(degree, "degree", min_val=1)

        # Create sparse adjacency matrix:
        adj = G.to_adjacency_matrix(weighted=weighted).tocoo()

        # Function to map node IDs to indices for quicker node index lookups
        self._node_lookup = G.node_ids_to_ilocs

        degree_mat = diags(np.asarray(adj.sum(1)).ravel())
        laplacian = degree_mat - adj
        laplacian = laplacian.tocoo()

        self.scales = np.array(scales).astype(np.float32)

        max_eig = eigs(laplacian, k=1, return_eigenvectors=False)
        self.max_eig = np.real(max_eig).astype(np.float32)[0]

        coeffs = [
            np.polynomial.chebyshev.Chebyshev.interpolate(
                lambda x: np.exp(-s * x), domain=[0, self.max_eig], deg=degree
            ).coef.astype(np.float32)
            for s in scales
        ]

        self.coeffs = tf.convert_to_tensor(np.stack(coeffs, axis=0))

        self.laplacian = tf.sparse.SparseTensor(
            indices=np.column_stack((laplacian.row, laplacian.col)),
            values=laplacian.data.astype(np.float32),
            dense_shape=laplacian.shape,
        )

    def num_batch_dims(self):
        return 1

    def flow(
        self,
        node_ids,
        sample_points,
        batch_size,
        targets=None,
        shuffle=False,
        seed=None,
        repeat=False,
        num_parallel_calls=1,
    ):
        """
        Creates a TensorFlow DataSet object of GraphWave embeddings.

        The dimension of the embeddings are `2 * len(scales) * len(sample_points)`.

        Args:
            node_ids: an iterable of node ids for the nodes of interest
                (e.g., training, validation, or test set nodes)
            sample_points: a 1D array of points at which to sample the characteristic function. This should be of the
                form: `sample_points=np.linspace(0, max_val, number_of_samples)` and is graph dependent.
            batch_size (int): the number of node embeddings to include in a batch.
            targets: a 1D or 2D array of numeric node targets with shape ``(len(node_ids),)``
                or ``(len(node_ids), target_size)``
            shuffle (bool): indicates whether to shuffle the dataset after each epoch
            seed (int,optional): the random seed to use for shuffling the dataset
            repeat (bool): indicates whether iterating through the DataSet will continue infinitely or stop after one
                full pass.
            num_parallel_calls (int): number of threads to use.
        """

        require_integer_in_range(batch_size, "batch_size", min_val=1)

        require_integer_in_range(num_parallel_calls, "num_parallel_calls", min_val=1)

        if not isinstance(shuffle, bool):
            raise TypeError(f"shuffle: expected bool, found {type(shuffle).__name__}")

        if not isinstance(repeat, bool):
            raise TypeError(f"repeat: expected bool, found {type(repeat).__name__}")

        ts = tf.convert_to_tensor(sample_points.astype(np.float32))

        def _map_func(x):
            return _empirical_characteristic_function(
                _chebyshev(x, self.laplacian, self.coeffs, self.max_eig), ts,
            )

        node_idxs = self._node_lookup(node_ids)

        # calculates the columns of U exp(-scale * eigenvalues) U^T on the fly
        # empirically calculate the characteristic function for each column of U exp(-scale * eigenvalues) U^T

        dataset = tf.data.Dataset.from_tensor_slices(
            tf.sparse.SparseTensor(
                indices=np.stack([np.arange(len(node_ids)), node_idxs], axis=1),
                dense_shape=(len(node_ids), self.laplacian.shape[0]),
                values=np.ones(len(node_ids), dtype=np.float32),
            )
        ).map(_map_func, num_parallel_calls=num_parallel_calls)

        if targets is not None:

            target_dataset = tf.data.Dataset.from_tensor_slices(targets)
            dataset = tf.data.Dataset.zip((dataset, target_dataset))

        # cache embeddings in memory for performance
        dataset = dataset.cache()

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(node_ids), seed=seed)

        if repeat:
            return dataset.batch(batch_size).repeat()
        else:
            return dataset.batch(batch_size)


def _empirical_characteristic_function(samples, ts):
    """
    This function estimates the characteristic function for the wavelet spread of a single node.

    Args:
        samples (Tensor): a tensor of samples drawn from a wavelet distribution at different scales.
        ts (Tensor): a tensor containing the "time" points to sample the characteristic function at.
    Returns:
        embedding (Tensor): the node embedding for the GraphWave algorithm.
    """
    # (scales, ns) -> (1, scales, ns)
    samples = samples[tf.newaxis, :, :]
    # (nt,) -> (nt, 1, 1)
    ts = ts[:, tf.newaxis, tf.newaxis]

    # (1, scales, ns) * (nt, 1, 1) -> (nt, scales, ns) via broadcasting rules
    t_psi = samples * ts

    # (nt, scales, ns) -> (nt, scales)
    mean_cos_t_psi = tf.math.reduce_mean(tf.math.cos(t_psi), axis=2)

    # (nt, scales, ns) -> (nt, scales)
    mean_sin_t_psi = tf.math.reduce_mean(tf.math.sin(t_psi), axis=2)

    # [(nt, scales), (nt, scales)] -> (2 * nt * scales,)
    embedding = K.flatten(tf.concat([mean_cos_t_psi, mean_sin_t_psi], axis=0))

    return embedding


def _chebyshev(one_hot_encoded_col, laplacian, coeffs, max_eig):
    """
    This function calculates one column of the Chebyshev approximation of exp(-scale * laplacian) for
    all scales using the approach from: https://arxiv.org/abs/1105.1891. See equations (7)-(11) for more info.

    Args:
        one_hot_encoded_col (SparseTensor): a sparse tensor indicating which column (node) to calculate.
        laplacian (SparseTensor): the unnormalized graph laplacian
        coeffs: the Chebyshev coefficients for exp(-scale * x) for each scale in the shape (num_scales, deg)
    Returns:
        (num_scales, num_nodes) tensor of the wavelets for each scale for the specified node.
    """

    # Chebyshev polynomials are in range [-1, 1] by default so we shift the coordinates here
    # the laplacian in the new coordinates is y = (L / a) - I. But y is only accessed through matrix vector
    # multiplications so we model it as a linear operator
    def y(vector):
        return K.dot(laplacian, vector) / a - vector

    a = max_eig / 2

    # f is a one-hot vector to select a column from the Laplacian
    # this allows to compute the filtered laplacian (psi in the paper) one column at time
    # using only matrix vector products
    f = tf.reshape(
        tf.sparse.to_dense(one_hot_encoded_col), shape=(laplacian.shape[0], 1)
    )

    T_0 = f  # If = f
    T_1 = y(f)

    cheby_polys = [T_0, T_1]
    for i in range(coeffs.shape[1] - 2):
        cheby_poly = 2 * y(cheby_polys[-1]) - cheby_polys[-2]
        cheby_polys.append(cheby_poly)

    # note: difference to the paper. the 0th coefficient is not halved here because its
    # automatically halved by numpy
    cheby_polys = K.squeeze(tf.stack(cheby_polys, axis=0), axis=-1)
    return tf.matmul(coeffs, cheby_polys)

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
Example of heterogeneous graph node classification using HinSAGE.

Requires the preprocessed Yelp dataset .. see the script `yelp_preprocessing.py`
for more details. We assume that the preprocessing script has been run.

Example usage:
python yelp-example.py -l <location_of_preprocessed_data>

Additional command line arguments are available to tune the learned model, to see a
description of these arguments use the `--help` argument:
python yelp-example.py --help

"""
import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from tensorflow import keras
from tensorflow.keras import optimizers, layers, metrics
import tensorflow.keras.backend as K

from stellargraph.core import StellarGraph
from stellargraph.layer import HinSAGE
from stellargraph.mapper import HinSAGENodeGenerator

from sklearn import model_selection
from sklearn import metrics as sk_metrics


def weighted_binary_crossentropy(weights):
    """
    Weighted binary cross-entropy loss
    Args:
        weights: A list or numpy array of weights per class

    Returns:
        A Keras loss function
    """
    weights = np.asanyarray(weights, dtype="float32")

    def loss_fn(y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_true, y_pred) * weights, axis=-1)

    return loss_fn


def train(
    G,
    user_targets,
    layer_size,
    num_samples,
    batch_size,
    num_epochs,
    learning_rate,
    dropout,
):
    """
    Train a HinSAGE model on the specified graph G with given parameters.

    Args:
        G: A StellarGraph object ready for machine learning
        layer_size: A list of number of hidden nodes in each layer
        num_samples: Number of neighbours to sample at each layer
        batch_size: Size of batch for inference
        num_epochs: Number of epochs to train the model
        learning_rate: Initial Learning rate
        dropout: The dropout (0->1)
    """
    print(G.info())

    # Split "user" nodes into train/test
    # Split nodes into train/test using stratification.
    train_targets, test_targets = model_selection.train_test_split(
        user_targets, train_size=0.25, test_size=None
    )

    print("Train targets:\n", train_targets.iloc[:, 0].value_counts())
    print("Test targets:\n", test_targets.iloc[:, 0].value_counts())

    # The mapper feeds data from sampled subgraph to GraphSAGE model
    generator = HinSAGENodeGenerator(G, batch_size, num_samples, head_node_type="user")
    train_gen = generator.flow_from_dataframe(train_targets, shuffle=True)
    test_gen = generator.flow_from_dataframe(test_targets)

    # GraphSAGE model
    model = HinSAGE(layer_sizes=layer_size, generator=generator, dropout=dropout)
    x_inp, x_out = model.build()

    # Final estimator layer
    prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

    # The elite label is only true for a small fraction of the total users,
    # so weight the training loss to ensure that model learns to predict
    # the positive class.
    # class_count = train_targets.values.sum(axis=0)
    # weights = class_count.sum()/class_count
    weights = [0.01, 1.0]
    print("Weighting loss by: {}".format(weights))

    # Create Keras model for training
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=learning_rate),
        loss=weighted_binary_crossentropy(weights),
        metrics=[metrics.binary_accuracy],
    )

    # Train model
    history = model.fit(train_gen, epochs=num_epochs, verbose=2, shuffle=False)

    # Evaluate on test set and print metrics
    predictions = model.predict(test_gen)
    binary_predictions = predictions[:, 1] > 0.5
    print("\nTest Set Metrics (on {} nodes)".format(len(predictions)))

    # Calculate metrics using Scikit-Learn
    cm = sk_metrics.confusion_matrix(test_targets.iloc[:, 1], binary_predictions)
    print("Confusion matrix:")
    print(cm)

    accuracy = sk_metrics.accuracy_score(test_targets.iloc[:, 1], binary_predictions)
    precision = sk_metrics.precision_score(test_targets.iloc[:, 1], binary_predictions)
    recall = sk_metrics.recall_score(test_targets.iloc[:, 1], binary_predictions)
    f1 = sk_metrics.f1_score(test_targets.iloc[:, 1], binary_predictions)
    roc_auc = sk_metrics.roc_auc_score(test_targets.iloc[:, 1], predictions[:, 1])

    print(
        "accuracy = {:0.3}, precision = {:0.3}, recall = {:0.3}, f1 = {:0.3}".format(
            accuracy, precision, recall, f1
        )
    )
    print("ROC AUC = {:0.3}".format(roc_auc))

    # Save model
    save_str = "_n{}_l{}_d{}_r{}".format(
        "_".join([str(x) for x in num_samples]),
        "_".join([str(x) for x in layer_size]),
        dropout,
        learning_rate,
    )
    model.save("yelp_model" + save_str + ".h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Graph node classification using GraphSAGE"
    )
    parser.add_argument(
        "-l",
        "--location",
        type=str,
        default=None,
        help="The location of the pre-processes Yelp dataset.",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=200, help="Batch size for training"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="The number of epochs to train the model",
    )
    parser.add_argument(
        "-d",
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout for the GraphSAGE model, between 0.0 and 1.0",
    )
    parser.add_argument(
        "-r",
        "--learningrate",
        type=float,
        default=0.001,
        help="Learning rate for training model",
    )
    parser.add_argument(
        "-n",
        "--neighbour_samples",
        type=int,
        nargs="*",
        default=[10, 5],
        help="The number of nodes sampled at each layer",
    )
    parser.add_argument(
        "-s",
        "--layer_size",
        type=int,
        nargs="*",
        default=[80, 80],
        help="The number of hidden features at each layer",
    )

    args = parser.parse_args()

    # Load graph and data
    if args.location is not None:
        data_loc = os.path.expanduser(args.location)
    else:
        raise ValueError(
            "Please specify the directory containing the dataset using the '-l' flag"
        )

    # Read the data
    print("Reading user features and targets...")
    user_features = pd.read_pickle(os.path.join(data_loc, "user_features_filtered.pkl"))
    user_targets = pd.read_pickle(os.path.join(data_loc, "user_targets_filtered.pkl"))

    # Quick check of target sanity
    vc = user_targets.iloc[:, 0].value_counts()
    if vc.iloc[0] == vc.sum():
        raise ValueError(
            "Targets are all the same, there has been an error in data processing"
        )

    print("Reading business features...")
    business_features = pd.read_pickle(
        os.path.join(data_loc, "business_features_filtered.pkl")
    )

    # Load graph
    print("Loading the graph...")
    Gnx = nx.read_graphml(os.path.join(data_loc, "yelp_graph_filtered.graphml"))

    # Features should be supplied as a dictionary of {node_type: DataFrame} for all
    # node types in the graph
    features = {"user": user_features, "business": business_features}

    # Create stellar Graph object
    G = StellarGraph.from_networkx(
        Gnx, node_type_attr="ntype", edge_type_attr="etype", node_features=features
    )

    train(
        G,
        user_targets,
        args.layer_size,
        args.neighbour_samples,
        args.batch_size,
        args.epochs,
        args.learningrate,
        args.dropout,
    )

Graphs with time series and sequence data
=========================================

`StellarGraph <https://github.com/stellargraph/stellargraph>`_ provides an algorithm for doing time series and sequence prediction with graphs. This folder contains demos of all of them to explain how they work and how to use them as part of a TensorFlow Keras data science workflow.

A time series (or sequence) prediction task aims to predict future data points from existing observations. On a graph, this happens for each node. The edges represent connections between nodes that may improve the predictions by using information from neighbours.

Spatio-temporal data is a classic example of a graph with time series information. Spatial information can be encoded in the graph structure, where edges represent connectivity (such as roads) or distance between locations. Traffic prediction is an example of such data, where each node might be an intersection or traffic sensor that yields a time series of the number or speed of vehicles. The data from nearby nodes is likely to be helpful for predictions.

Find algorithms and demos for a graph
-------------------------------------

This table lists all representation learning demos, including the algorithms trained, how they are trained, the types of graph used, and the tasks demonstrated.

.. list-table::
   :header-rows: 1

   * - demo
     - algorithm(s)
     - task
   * - :doc:`Graph Convolution + LSTM <gcn-lstm-time-series>`
     - GraphConvolutionLSTM (T-GCN)
     - traffic prediction

See :doc:`the root README <../../README>` or each algorithm's documentation for the relevant citation(s). See :doc:`the demo index <../index>` for more tasks, and a summary of each algorithm.

Table of contents
-----------------

.. toctree::
    :titlesonly:
    :glob:

    ./*

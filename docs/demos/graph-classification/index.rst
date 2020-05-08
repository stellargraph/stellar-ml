Graph classification
=======================================

`StellarGraph <https://github.com/stellargraph/stellargraph>`_ provides algorithms for graph classification. This folder contains demos to explain how they work and how to use them as part of a TensorFlow Keras data science workflow.

A graph classification task predicts an attribute of each graph in a collection of graphs. For instance, labelling each graph with a categorical class (binary classification or multiclass classification), or predicting a continuous number (regression). It is supervised, where the model is trained using a subset of graphs that have ground-truth labels.

Find algorithms and demos for a collection of graphs
----------------------------------------------------

This table lists all graph classification demos, including the algorithms trained and the types of graphs used.

.. list-table::
   :header-rows: 1

   * - demo
     - algorithm(s)
     - node features
     - inductive
   * - :doc:`GCN Supervised Graph Classification <gcn-supervised-graph-classification>`
     - GCN, mean pooling
     - yes
     - yes
   * - :doc:`DGCNN <dgcnn-graph-classification>`
     - DeepGraphCNN
     - yes
     - yes


See :doc:`the demo index <../index>` for more tasks, and a summary of each algorithm.

Table of contents
-----------------

.. toctree::
    :titlesonly:
    :glob:

    ./*

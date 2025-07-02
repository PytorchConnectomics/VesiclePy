# Neuron Classification

This module performs a cluster analysis on morphological data of neurons and vesicles to identify distinct neuron subtypes. The analysis uses a morphology dataset of 20 neuron samples, characterized by a combination of numerical (e.g., 'TotalVol', 'NucVol', 'TotalLen') and categorical (e.g., 'Branch' type) features, as well as vesicle numbers by type (e.g., ‘CV’, ‘DCV’ and ‘DCH’). 

The primary methods employed include:
* The data used to perform the cluster analysis is included in the Jupyter Notebook.
* Log transformation for numerical features.
* Gower's distance to handle mixed data types.
* Hierarchical agglomerative clustering (complete linkage method).
* Cluster identification based on a distance threshold of 0.4.
* Visualization of results using a dendrogram and a 2D Multidimensional Scaling (MDS) plot.

This repository contains the Google Colab notebook used for the analysis.

The analysis is performed in a Python environment. The key dependencies are:
* Python 3.x
* pandas
* numpy
* scipy (for spatial distance, cluster hierarchy)
* gower
* matplotlib (for plotting)
* scikit-learn (for MDS)


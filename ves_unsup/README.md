# Unsupervised clustering of small vesicle appearances
This repository contains code for training a [PyroVED](https://github.com/ziatdinovmax/pyroVED) rotation and translation invariant variational autoencoder (VAE) to embed small vesicle appearances in a 2D latent space, where clustering can be done using [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html).

## Generate training data
`generate_sample.py` generates a `samples.npz` file with two entries: `patches` (`Nx11x11` array) and `ids` (`N` array). The `patches` array contains EM patches of the small vesicles resized to `11x11` patch size, and the `ids` array contains the corresponding IDs of the small vesicles (used later to identify which vesicle belongs to which cluster).

## Model training
`main.py` loads the `samples.npz` file and trains the PyroVED model, logging training loss to `wandb`. Model weights are saved to `model.pt`.

## Visualization and clustering
`plot.py` runs inference on all the patches, visualizing the predicted embeddings in a 2D latent space, and clustering them using UMAP.

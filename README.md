## Variational Autoencoder with Knot Classifier in the Latent Space

Experimental, independent re-implementation of a knot-type-aware variational autoencoder.

## Status
This repository contains a runnable research prototype of a variational autoencoder with a knot
classifier acting on the latent space. The code has been partially tested on a small dataset of
knotted coarse-grained polymers generated via molecular dynamics simulations (included in the
repository) and qualitatively reproduces the results reported in:

https://journals.aps.org/pre/abstract/10.1103/mdpw-lvcy

Results should not be interpreted as performance benchmarks.

## Attribution
This repository contains an independent re-implementation based on the method introduced in:

Braghetto, A., Kundu, S., Baiesi, M., and Orlandini, E.  
*Variational Autoencoders understand Knot Topology*  
https://journals.aps.org/pre/abstract/10.1103/mdpw-lvcy

The code was written from scratch based on the published paper and does not reuse the original
implementation.

## Description
The code implements a variational autoencoder with a transformer- and convolution-based encoder
and a decoder composed of transposed convolutional layers. The VAE learns a latent representation
of knotted polymer configurations while organizing the latent space according to knot type through
an MLP classifier acting directly on the latent embeddings and contributing to the loss function.

The code can also be run in evaluation mode to encode and decode a set of configurations after
training.

The implementation consists of a single runnable file that takes as input a configuration file
and a corresponding labels file.

## Run
The code can be executed in two modes.

- Training mode
python3 VAEC_knots.py train training_confs.dat training_labels.dat

This mode trains the variational autoencoder.

- Evaluation mode
python3 VAEC_knots.py eval training_confs.dat training_labels.dat

This mode loads a trained model and encodes/decodes the configurations contained in the input
file. It also produces scatter plots of the latent embeddings.

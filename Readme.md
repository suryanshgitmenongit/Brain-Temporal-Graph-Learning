# BrainTGL: Temporal Graph Learning for Brain Networks

This repository contains an implementation of the architecture of the BrainTGL model, a temporal graph learning framework for analyzing brain networks based on resting-state functional MRI (rs-fMRI) data. BrainTGL leverages graph convolutional networks (GCNs) and recurrent neural networks (RNNs) to capture both spatial and temporal dynamics in brain networks, enabling tasks such as brain disease classification and subtype identification. 

Please note that the code is only a partial implementation.
Fixes, additions will be pushed over time.

## Overview

BrainTGL operates on rs-fMRI data, which provides a time series of brain activity measurements. The model processes this data to construct a series of dynamic graphs representing the functional connectivity between brain regions over time. These dynamic graphs are then fed into a graph attention pooling module, which selects important graph structures while preserving temporal information. Subsequently, the pooled graph representations are passed through a dual temporal graph learning (DTGL) module, combining GCN and LSTM layers to capture spatial and temporal features simultaneously.

## Repository Structure

- `model.py`: Contains the source code for the BrainTGL model implementation.
- `README.md`: This file providing an overview of the repository.

## Usage

1. Clone the repository:
2. Install dependencies
3. Prepare your rs-fMRI dataset (not provided in this repository).
4. Train and evaluate the BrainTGL model using the provided scripts or integrate it into your own project.

## References
BrainTGL paper:
https://www.sciencedirect.com/science/article/abs/pii/S001048252201229X

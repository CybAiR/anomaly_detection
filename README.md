# Mars Surface Anomaly Detection
Program designed for real-time visual anomaly detection during ERC

## Overview
Input images are processed by neural network based on Wide ResNet-18 backbone.

Instead of recognising predefined classes of anomalies, the model learns what 
normal terrain should look like and identifies regions that differ significantly from the training distribution.

## Dataset
Images were collected at Kąkolewo Airport Campus of Poznań University of Technology, inside a geodome simulating Mars-like terrain.
The dataset contain normal terrain images and anomalies (consisting of both various objects and human images)


## Technologies

- Python
- PyTorch
- Anomalib
- OpenCV
- NumPy

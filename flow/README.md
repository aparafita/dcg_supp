# flow

This project implements basic Normalizing Flows in PyTorch 
and provides functionality for defining your own easily, 
following the conditioner-transformer architecture.
It is particularly intended for lower-dimensional flows and learning purposes.

Supports conditioning flows, meaning, learning probability distributions
conditioned by a given conditioning tensor; this is specially useful for modelling causal mechanisms.
# MATH304 Final Project – Movement Decoding from EEG (MATLAB)

This project implements a Support Vector Machine (SVM) in MATLAB to classify intended left vs. right hand movements based on brain signals (EEG). Developed as part of MATH 304: Numerical Analysis and Optimization at Duke Kunshan University.

## Overview

- **Input**: 204-feature EEG data from .mat files (`Data.mat`, `Sensors.mat`)
- **Method**: Barrier method for constrained optimization, using:
  - Newton's method for subproblems
  - Backtracking line search
- **Evaluation**: 6-fold cross-validation
- **Tools used**: Custom MATLAB scripts

Despite challenges with ill-conditioned Hessians, the project explores the use of second-order methods and potential improvements like L-BFGS.

## Files

| File              | Description                                                |
|-------------------|------------------------------------------------------------|
| `main.m`          | Entry point that loads data, runs training & evaluation     |
| `barrierMethod.m` | Implements the barrier method for constrained optimization  |
| `NewtonMethod.m`  | Computes Newton step for solving subproblems                |
| `loss.m`          | Computes SVM loss and gradients                             |
| `showWeights.m`   | Visualizes the learned SVM weights                          |
| `Data.mat`        | EEG feature data (204 features per trial)                   |
| `Sensors.mat`     | Metadata about EEG sensor layout                            |



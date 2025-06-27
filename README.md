# Bayesian Inference of Neuron Activation Parameters via the Leaky Integrate-and-Fire Model using BayesFlow

## Overview

This project applies amortized Bayesian inference for parameter estimation in the Leaky Integrate-and-Fire (LIF) model using the BayesFlow framework. We simulate voltage trajectories, define explicit proper priors, train a neural posterior estimator, and validate its ability to recover true underlying parameters through rigorous diagnostics.

---

## Goals

- Simulate voltage traces using the LIF model.
- Define explicit proper priors over model parameters.
- Train a BayesFlow posterior estimator \( q_\phi(\theta \mid y) \).
- Evaluate posterior recovery performance on test data.
- Perform diagnostic and calibration procedures to validate inference.

---

## Contents

- `simulator/`: LIF simulator and data generation scripts
- `models/`: BayesFlow summary and inference networks
- `training/`: Scripts for model training and validation
- `diagnostics/`: SBC, PPC, and typicality checks
- `notebooks/`: Jupyter notebooks for exploration and evaluation
- `report/`: Project report and appendix
- `README.md`: This file

---

## Statistical Model

### Priors

```math
\begin{align*}
\tau &\sim \text{LogNormal}(\mu_{\tau}, \sigma_{\tau}) \\
V_{\text{rest}} &\sim \mathcal{N}(\mu_r, \sigma_r) \\
C &\sim \text{LogNormal}(\mu_C, \sigma_C) \\
V_{\text{th}} &\sim \mathcal{N}(\mu_{\text{th}}, \sigma_{\text{th}})
\end{align*}

## Simulator

- **Input**: Sinusoidal or square current waveforms.
- **Integration**: Euler method for solving the LIF model ODEs.
- **Output**: Voltage trace \( V(t) \in \mathbb{R}^T \), where \( T \) is the number of time steps.

---

## BayesFlow Setup

### Approximator

- **SummaryNet**:  
  - LSTM (for raw voltage trace input)  
  - MLP (for summary statistics like ISI, spike counts)
- **InferenceNet**:  
  - Normalizing flow-based posterior estimator using Coupling Flows

### Architecture

- **SummaryNet**:  
  - 2-layer LSTM or MLP  
  - 64–128 hidden units  
  - ReLU activation
- **InferenceNet**:  
  - 4–6 affine coupling layers  
  - Batch normalization between layers

---

## Training

- **Samples**: 100,000 simulated parameter-trajectory pairs (offline generation)
- **Epochs**: 300
- **Batch Size**: 256
- **Optimizer**: Adam  
  - Learning Rate: \( 1 \times 10^{-4} \)  
  - Optional: cosine annealing or exponential decay schedule

---

## Diagnostics

- **Convergence**:  
  - Monitor training and validation loss curves

- **Simulation-Based Calibration (SBC)**:  
  - Rank histograms comparing true vs. inferred parameters

- **Posterior Predictive Checks (PPC)**:  
  - Sample parameters from posterior \( q_\phi(\theta \mid y_{\text{obs}}) \)  
  - Simulate voltage traces and compare with observed ones

- **Typicality Checks**:  
  - Project test data to summary space  
  - Assess alignment with training data distribution

---

## Inference & Validation

- Evaluate performance on held-out synthetic trajectories
- Analyze posterior \( q_\phi(\theta \mid y_{\text{test}}) \) for:
  - **Bias** in parameter estimates
  - **Coverage** of credible intervals
  - **Posterior contraction** (narrowing with more data)
- Optionally apply the model to real-world or biologically plausible traces

---

## Limitations

- The Leaky Integrate-and-Fire (LIF) model simplifies biological realism
- Posterior quality is sensitive to:
  - Prior specification
  - Coverage of training simulations
- Generalization to misspecified or out-of-distribution regimes may be limited

---

## Backend Options

BayesFlow supports three deep learning frameworks:

- **TensorFlow**: Default backend with full feature support
- **JAX**: Lightweight, functional alternative (recommended for speed)
- **PyTorch**: Available via custom interface (requires manual setup)

---

## Installation & Backend
- The dev version is used for this working example:
'''pip install git+https://github.com/bayesflow-org/bayesflow.git@dev'''
- The backend used for Keras3 is JAX:
'''os.environ["KERAS_BACKEND"] = "jax"'''

---

## Documentation

For more details, check out the [BayesFlow Docs](https://bayesflow.org/main/index.html).

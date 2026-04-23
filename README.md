# SEC Model

Code for the SEC model described in the manuscript.

## Overview

Spectral-Enhanced Cross-domain Network (SEC) is a daily runoff forecasting model designed for non-stationary hydrological sequences. The model combines spectral energy modulation with cross-domain multi-scale collaboration to improve the representation of weak high-frequency signals, multivariate interactions, and temporal dependencies at different scales. The architecture integrates a Spectral Energy Modulator, a Frequency Domain Feature Decoupling Module, a Variable Interaction Commonality Module, a Multi-Scale Temporal Graph Neural Network, and a Global Dependency Encoder.

## Files

- `sec_model.py`: SEC model implementation.
- `requirements.txt`: dependency list.
- `example_data/`: example station data for input format reference.

## Usage

```python
from sec_model import SECModel
```

## Requirements

- Python
- PyTorch

Install dependencies:

```bash
pip install -r requirements.txt
```

## Notes

- Module names follow the terminology used in the manuscript.
- Large datasets, experiment outputs, editor caches, and temporary files are excluded through `.gitignore`.

# Quantinuum Challenge

Our team's submission for the Quantinuum Challenge at iQuHACK 2025, focused on simulating Hamiltonian and Lindbladian quantum systems.

## Overview

This project implements advanced optimization techniques for quantum system simulations, with a particular focus on finding optimal Trotter decompositions that balance accuracy and circuit complexity. We developed a comprehensive framework that includes:

- Hamiltonian simulation optimization
- Lindbladian (open quantum system) simulation
- Automated parameter optimization for Trotter steps
- Circuit complexity analysis and optimization
- Error analysis and visualization tools

## Key Components

### 1. Hamiltonian Simulation (`notebooks/hamiltonian_simulation.ipynb`)
- Implementation of XXZ chain and transverse field Ising Hamiltonians
- First and second-order Trotter decomposition algorithms
- Energy simulation over time

### 2. Lindbladian Simulation (`notebooks/lindblad_simulation.ipynb`)
- Support for open quantum system simulation
- Implementation of dilation operators for dissipative dynamics
- Deterministic and stochastic simulation approaches
- XXZ Lindblad simulation with configurable parameters

### 3. Auto-Optimization Algorithm (`trotter_sim.py`)

Our main innovation is an automated optimization system that:
- Finds optimal number of Trotter steps considering both accuracy and circuit complexity based on depth and gate counts
- Uses a weighted scoring system to balance competing objectives
- Provides visualization of the optimization landscape


### 4. Circuit Analysis Tools

We implemented comprehensive circuit analysis utilities that provide:
- Gate count statistics
- Circuit depth analysis
- Two-qubit gate metrics
- Parameter rotation analysis
- Visualization of circuit properties

## Usage Example

```python
from meta_opt.trotter_sim import TrotterStep
from meta_opt.hamiltonian_sim import get_xxz_chain_hamiltonian

# Create Hamiltonian
hamiltonian = get_xxz_chain_hamiltonian(2, 1.0)

# Initialize optimizer
trotter_step = TrotterStep(hamiltonian, 2, 10.0, 30)

# Find optimal number of Trotter steps
optimal_steps = trotter_step.find_best_n_steps()
```

## Technical Details

Our implementation uses:
- pytket quantum SDK
- NumPy and SciPy for numerical calculations
- Matplotlib for visualization

## Future Work

Potential areas for expansion include:
- Support for more complex Hamiltonians
- Advanced error mitigation techniques
- Integration with hardware-specific noise models
- Parallel optimization strategies
- Machine learning-based parameter optimization

## Dependencies

- pytket >= 1.37.0
- pytket-qiskit >= 0.62.0
- pytket-quantinuum
- numpy
- scipy
- networkx
- matplotlib

## Team

This project was developed as part of the Quantinuum Challenge at iQuHACK 2025. Team members:
- Rohan Timmaraju
- Michael Cai
- Arun Moorthy
- Brian Zhao
- Ethan Kur

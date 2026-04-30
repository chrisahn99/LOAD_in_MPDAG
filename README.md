# LOAD in MPDAG

Local Ordering-based Ancestral Discovery (LOAD) in Maximal Partially Directed Acyclic Graphs (MPDAGs).

## Overview
This repository provides the core implementation of the LOAD algorithm and its extension for MPDAGs. It is designed for researchers in causal discovery to run local ancestral discovery experiments with varying degrees of background knowledge.

## Project Structure
- `src/`: Core Python modules.
  - `load.py`: Implementation of `load` and `load_in_mpdag`.
  - `background_knowledge.py`: Utilities for handling structural constraints.
  - `counting_ci_tests.py`: CI test wrappers with performance tracking.
  - `evaluation.py`: Metrics for validating discovered causal relations.
- `run_experiments.py`: Main script for conducting simulations.
- `run_experiments_v2.py`: Updated experiment pipeline with refined background knowledge sampling.
- `generate_data.R` / `evaluate.R`: R-based utilities for data synthesis and statistical comparison.

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Running Experiments
The public scripts provide the logic for synthetic data generation and algorithm execution. To run the standard experiment suite:
```bash
python run_experiments.py [OPTIONS]
```
*Note: You may need to create a local directory (e.g., `experiment_runs/`) to store outputs, as these are ignored by version control.*

## Implementation Details
The algorithm focuses on target-node-specific ancestral discovery. Key features include:
- Support for **Maximal Partially Directed Acyclic Graphs (MPDAGs)** as prior knowledge.
- Integration of **Conditional Independence (CI) test counting** to measure computational efficiency.
- Local sampling strategies for forbidden and required edges.

## Citation
If you use this code in your research, please cite:
[Insert Citation Here]

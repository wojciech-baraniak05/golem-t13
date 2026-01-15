# golem-t13
Multi Layer Perceptron and Decision Tree hybrid

## Project Structure

src/golem/ - Main package containing the pipeline and core functionality.  
src/golem/modeling/ - Training, testing, and evaluation utilities for models.  
src/golem/models/ - Model implementations including MLP, DecisionTree, and RandomForest classifiers.  
make_data.ipynb - generating 2, and loading 1 dataset - moons, circles, digits Pipeline class, also a little bit of visualization 

**Pipeline.py** - main class that is responsible for simplifying code used in one notebook, handles multiclass or nor problems, load data and so much more,  

## Setup - uv
To install uv on lin/mac:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
For windows:
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
## Initializing
Create a venv with uv
```bash
uv sync
```

## Overview
This project compares Multi Layer Perceptron, Decision tree, and the hybrid of these two.

In the Hybrid architecture, MLP returns embeddings (that are the results from the penultimate layer), then Decision Tree uses them to learn the proper classification. MLP architecture:

<img width="363" height="256" alt="image" src="https://github.com/user-attachments/assets/5d575d13-5e50-4535-81cc-d6046748fe25" />



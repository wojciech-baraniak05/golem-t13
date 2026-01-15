# golem-t13
Multi Layer Perceptron and Decision Tree hybrid

## Setup - uv
To install uv on lin/mac:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
For windows:
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
## Updating
Create a venv with uv
```bash
uv sync
```

## Overview
This project compares Multi Layer Perceptron, Decision tree, and the hybrid of these two.

In the Hybrid architecture, MLP returns embeddings (that are the results from the penultimate layer), then Decision Tree uses them to learn the proper classification. MLP architecture:

<img width="363" height="256" alt="image" src="https://github.com/user-attachments/assets/5d575d13-5e50-4535-81cc-d6046748fe25" />

## Datasets
Datasets used are make_moons, (...)


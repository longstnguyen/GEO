<p align="center"><img src="../docs/assets/results-header.svg" width="100%" alt="GEO results"></p>

<p align="center"><a href="../README.md">← Overview</a> · <a href="../docs/usage.md">Run experiments</a> · <a href="../docs/protocols.md">Protocols</a></p>

# Published results

These AUROC (%) values are transcribed from the GEO case-study table in the [IEEE Access survey](https://doi.org/10.1109/ACCESS.2026.3717452). Each configuration is evaluated on ten settings. The [CSV](paper-auroc.csv) provides the complete matrix.

**Bold** marks the best value and <ins>underlining</ins> marks the second-best distinct value in each metric column, including ties. **P**, **E**, and **O** denote propagation, energy regularization and one-class learning; ✓ enables a component. S/F/L denote structure, feature and label shifts. A dash records a non-converged configuration in the published table.

## Cora and Amazon-Photo

| Backbone | P | E | O | Cora S | Cora F | Cora L | Photo S | Photo F | Photo L |
| :-- | :--: | :--: | :--: | --: | --: | --: | --: | --: | --: |
| MSP | — | — | — | 71.54 | 85.20 | 91.15 | 78.67 | 97.45 | 42.38 |
| GCN | — | — | ✓ | 70.98 | 86.26 | 91.35 | 98.30 | 97.97 | 93.65 |
| GCN | — | ✓ | ✓ | 46.68 | 49.93 | 67.62 | 71.69 | 76.50 | 73.25 |
| GCN | ✓ | — | ✓ | 77.62 | 93.40 | - | 98.66 | 98.53 | 97.48 |
| GCN | ✓ | ✓ | — | **90.31** | 95.34 | 91.35 | 99.82 | **99.63** | 93.65 |
| GCN | ✓ | ✓ | ✓ | 90.24 | **95.51** | - | 98.89 | <ins>98.84</ins> | 96.94 |
| GEN | — | — | ✓ | 70.99 | 86.08 | 91.35 | 99.20 | 97.32 | 93.75 |
| GEN | — | ✓ | ✓ | 75.38 | 88.06 | 71.01 | 99.84 | 98.45 | 90.57 |
| GEN | ✓ | — | ✓ | 87.12 | 93.23 | 92.69 | 98.65 | 98.52 | 97.47 |
| GEN | ✓ | ✓ | — | <ins>90.25</ins> | 95.43 | 91.50 | 98.86 | 98.68 | - |
| GEN | ✓ | ✓ | ✓ | 90.18 | <ins>95.49</ins> | 91.48 | 98.91 | <ins>98.84</ins> | 97.04 |
| GAT | — | — | ✓ | 75.64 | 89.43 | 92.74 | <ins>99.98</ins> | 98.42 | **98.05** |
| GAT | — | ✓ | ✓ | 74.42 | 89.58 | 59.63 | **99.99** | 98.52 | 91.56 |
| GAT | ✓ | — | ✓ | 86.19 | 94.26 | **94.44** | 98.69 | 98.41 | <ins>97.61</ins> |
| GAT | ✓ | ✓ | — | 87.15 | 94.44 | 92.68 | 98.74 | 98.50 | 96.88 |
| GAT | ✓ | ✓ | ✓ | 87.22 | 94.82 | <ins>92.75</ins> | 98.74 | 98.49 | 96.96 |

## Coauthor-CS and ogbn-arxiv

| Backbone | P | E | O | CS S | CS F | CS L | arXiv |
| :-- | :--: | :--: | :--: | --: | --: | --: | --: |
| MSP | — | — | — | 94.63 | 96.59 | 93.96 | 63.91 |
| GCN | — | — | ✓ | 96.19 | 97.87 | 95.57 | 66.41 |
| GCN | — | ✓ | ✓ | 80.46 | 93.23 | 85.36 | 53.64 |
| GCN | ✓ | — | ✓ | 99.61 | 99.66 | 97.01 | 71.06 |
| GCN | ✓ | ✓ | — | **99.99** | **99.97** | **97.83** | 74.77 |
| GCN | ✓ | ✓ | ✓ | <ins>99.94</ins> | <ins>99.92</ins> | 97.10 | **86.36** |
| GEN | — | — | ✓ | 96.13 | 97.84 | 95.50 | 66.37 |
| GEN | — | ✓ | ✓ | 98.25 | 99.04 | 91.23 | 77.71 |
| GEN | ✓ | — | ✓ | 99.62 | 99.64 | 97.04 | 65.08 |
| GEN | ✓ | ✓ | — | <ins>99.94</ins> | <ins>99.92</ins> | <ins>97.12</ins> | 77.62 |
| GEN | ✓ | ✓ | ✓ | <ins>99.94</ins> | <ins>99.92</ins> | 97.08 | <ins>78.35</ins> |
| GAT | — | — | ✓ | 97.08 | 98.65 | 94.97 | 55.06 |
| GAT | — | ✓ | ✓ | 98.28 | 99.21 | 74.83 | 75.78 |
| GAT | ✓ | — | ✓ | 99.51 | 99.71 | 96.75 | 54.07 |
| GAT | ✓ | ✓ | — | 99.85 | 99.88 | 96.17 | 69.39 |
| GAT | ✓ | ✓ | ✓ | 99.86 | 99.88 | 96.10 | 76.30 |

## Run your own comparison

New experiments save configuration, logs, and machine-readable metrics under `outputs/`. The [running guide](../docs/usage.md#run-artifacts) describes these artifacts and the diagnostic score exports. Published values remain in [`paper-auroc.csv`](paper-auroc.csv).

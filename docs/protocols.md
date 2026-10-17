<p align="center"><img src="assets/guide.svg" width="100%" alt="GEO research notes"></p>

<p align="center"><a href="../README.md">← Overview</a> · <a href="usage.md">Running experiments</a> · <a href="method.md">Method</a> · <a href="protocols.md">Protocols</a></p>

# Dataset and evaluation protocols

The loaders define ID nodes, exposed OOD nodes used for regularization, and OOD test nodes used for evaluation. Cora uses its public Planetoid split; the other datasets draw ID train/validation/test partitions with `--train_prop` and `--valid_prop` (both default to 0.1).

## Synthetic shifts

Cora, Amazon-Photo and Coauthor-CS support three protocols. Features are row-normalized before generating OOD variants.

| Protocol | Construction | Exposure and test |
| :-- | :-- | :-- |
| Structure | Replace edges with a stochastic block model, preserving node features and labels | Two independently drawn graphs |
| Feature | Each node receives a random convex combination of two node-feature vectors | Two independently drawn feature matrices |
| Label | Partition nodes by class index | Disjoint ID, exposed and test classes |

Structural sampling uses within-block probability `1.5 × d` and cross-block probability `0.5 × d`, where `d` is the input graph's stored-edge density. Blocks follow the class count. Graph storage can include both directions of an undirected edge.

### Label cohorts

| Dataset | ID labels | Exposed OOD label | OOD test labels |
| :-- | :-- | :-- | :-- |
| Cora | 4–6 | 3 | 0–2 |
| Amazon-Photo | 5–7 | 4 | 0–3 |
| Coauthor-CS | 5–14 | 4 | 0–3 |

The graph and feature matrix are shared across label cohorts; node masks determine which instances contribute to each objective and metric. Cora's public split masks are intersected with its ID class mask.

## Temporal cohorts

Use `--dataset arxiv` to load `ogbn-arxiv`. The source experiment loader uses these year boundaries:

| Role | Publication years |
| :-- | :-- |
| ID | Through 2015 |
| Exposed OOD | 2016–2017 |
| OOD test cohort 1 | 2018 |
| OOD test cohort 2 | 2019 |
| OOD test cohort 3 | 2020 |

The feature matrix retains original node indices. Inductive edge masks include only nodes up to each cohort's upper year bound; `node_idx` identifies the nodes scored in that cohort. The result contains a separate metric triple for each year.

## Additional loaders

The source diagnostics also support Twitch language domains and ogbn-proteins species cohorts. Twitch uses DE as ID, EN for exposure, and ES/FR/RU for testing. Protein cohorts follow the sorted species identifiers in the dataset loader.

## Metrics and selection

- **AUROC:** area under the receiver operating characteristic curve.
- **AUPR:** average precision with ID nodes as the positive class.
- **FPR95:** false-positive rate at approximately 95% ID recall.
- **ID test score:** node-classification accuracy, or ROC-AUC for supported binary/multilabel datasets.

Reported metrics use a 0–100 scale. The detector's score increases with ID confidence. The epoch with minimum ID validation loss determines the detection metrics retained for each run; OOD test metrics do not select the epoch. Multiple initializations share the same constructed cohorts and data splits.

<p align="center"><a href="../README.md">← Overview</a> · <a href="usage.md">Running experiments</a> · <a href="method.md">Method</a> · <a href="protocols.md">Protocols</a></p>

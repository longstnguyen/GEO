<p align="center"><img src="assets/guide.svg" width="100%" alt="GEO research notes"></p>

<p align="center"><a href="../README.md">← Overview</a> · <a href="usage.md">Running experiments</a> · <a href="method.md">Method</a> · <a href="protocols.md">Protocols</a></p>

# Method and implementation

GEO connects supervised graph learning with energy separation and one-class compactness. The implementation keeps these computations in model classes, shares detector construction through a factory, and handles paths and logging in a separate command layer.

## Learning and scoring

For node logits $z$ and temperature $T$, energy is

$$E(z)=-T\log\sum_c\exp(z_c/T).$$

The detector returns **negative energy**, so higher values indicate ID confidence. Evaluation treats ID as the positive class. With score propagation enabled, each step combines the current score with the average of incoming-neighbor scores:

$$s^{(k+1)}=\alpha s^{(k)}+(1-\alpha)D^{-1}As^{(k)}.$$

Training uses supervised NLL plus enabled regularizers. Energy regularization penalizes ID energy above `m_in` and exposed-OOD energy below `m_out`, with squared hinge penalties. The one-class term is

$$L_{\mathrm{OCC}}=r^2+\frac{1}{\nu N}\sum_i\max(0,\lVert h_i-c\rVert^2-r^2).$$

The center is initialized from training-node representations and held fixed within a run. The radius starts at 0.1 and is optimized with the model; the center and radius are reset for every initialization. In the supplied encoder interface, the final encoder vectors serve both classification and one-class learning. Geometry shapes training; the inference score remains energy-based.

## Responsibility boundaries

| Layer | Modules | Responsibility |
| :-- | :-- | :-- |
| Command interface | `cli.py`, `parse.py` | Parse arguments, isolate output directories, record runtime metadata |
| Experiment pipelines | `main.py`, `discuss.py` | Load data, optimize, evaluate and select results |
| Detector factory | `factory.py` | Construct a supported detector from its registered name |
| Models and objectives | `backbone.py`, `baselines.py`, `gnnsafe.py`, `loss.py` | Encode graphs, calculate losses and return scores |
| Data and evaluation | `dataset.py`, `data_utils.py`, `logger.py` | Construct OOD cohorts, calculate metrics and summarize runs |

The factory maps names to detector classes. Each detector provides `reset_parameters`, `loss_compute`, and `detect`; this shared interface lets the experiment pipelines use confidence, energy, exposure and distance methods without duplicating construction logic.

## Extending an experiment

To add a detector, implement the shared detector interface and register the class in `DETECTORS` in [`factory.py`](../src/geo_ood/factory.py). Keep its score direction consistent with ID confidence. Add a small graph test for loss gradients and detection shape before running a dataset experiment.

To add a data protocol, return an ID graph, an exposure graph, and either one OOD test graph or a list of test cohorts. Every graph carries `node_idx`; the ID graph also supplies train/validation/test splits. Keep the cohort construction in [`dataset.py`](../src/geo_ood/dataset.py) so both experiment pipelines use the same definition.

## Repeatable evaluation

The seed controls NumPy, Python and PyTorch random draws. Repeated runs reset model parameters while sharing the generated data and splits. Detection selects metrics by minimum ID validation loss; classification selects by maximum ID validation score. Each run executes the requested number of epochs.

<p align="center"><a href="../README.md">← Overview</a> · <a href="usage.md">Running experiments</a> · <a href="method.md">Method</a> · <a href="protocols.md">Protocols</a></p>

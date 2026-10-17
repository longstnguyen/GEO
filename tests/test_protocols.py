import numpy as np
import pytest
import torch

from geo_ood.data_utils import get_measures, rand_splits
from geo_ood.dataset import create_feat_noise_dataset
from geo_ood.logger import Logger_detect
from test_model import graph


def test_split_disjoint_and_reproducible():
    ids = torch.arange(100)
    np.random.seed(12)
    a = rand_splits(ids, 0.1, 0.1)
    np.random.seed(12)
    b = rand_splits(ids, 0.1, 0.1)
    assert all(torch.equal(a[k], b[k]) for k in a)
    train, valid, test = (set(a[k].tolist()) for k in ["train", "valid", "test"])
    assert not (train & valid or train & test or valid & test)
    assert len(train | valid | test) == 100


def test_interpolation_preserves_shape_and_source():
    data = graph()
    original = data.x.clone()
    torch.manual_seed(4)
    a = create_feat_noise_dataset(data)
    torch.manual_seed(4)
    b = create_feat_noise_dataset(data)
    assert a.x.shape == data.x.shape and torch.equal(data.x, original)
    assert torch.equal(a.x, b.x) and not torch.equal(a.x, data.x)


def test_metrics_use_id_confidence_orientation():
    auroc, aupr, fpr, _ = get_measures(np.array([5.0, 6.0]), np.array([1.0, 2.0]))
    assert auroc == 1 and aupr == 1 and fpr == 0


def test_logger_selects_by_validation_loss():
    logger = Logger_detect(1)
    logger.add_result(0, [0.99, 0.99, 0.01, 0.9, 2.0])
    logger.add_result(0, [0.7, 0.6, 0.2, 0.8, 1.0])
    logger.add_result(0, [0.98, 0.97, 0.02, 0.95, 3.0])
    selected = logger.print_statistics()
    assert selected[0, 0].item() == pytest.approx(70)


def test_arxiv_cohorts_and_inductive_edges(monkeypatch):
    from types import SimpleNamespace
    import ogb.nodeproppred
    from geo_ood.dataset import load_arxiv_dataset

    years = np.array([2014, 2015, 2016, 2017, 2018, 2019, 2020]).reshape(-1, 1)
    edges = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])
    fixture = SimpleNamespace(
        graph={"node_year": years, "edge_index": edges, "node_feat": np.ones((7, 3))},
        labels=np.zeros((7, 1), dtype=int),
    )
    monkeypatch.setattr(ogb.nodeproppred, "NodePropPredDataset", lambda **kw: fixture)
    ind, exposure, tests = load_arxiv_dataset("unused")
    assert ind.node_idx.tolist() == [0, 1]
    assert exposure.node_idx.tolist() == [2, 3]
    assert [d.node_idx.tolist() for d in tests] == [[4], [5], [6]]
    for data, bound in zip([ind, exposure, *tests], [1, 3, 4, 5, 6]):
        assert data.edge_index.max().item() <= bound


def test_sbm_keeps_all_nodes_with_uneven_blocks(monkeypatch):
    import geo_ood.dataset as datasets
    from torch_geometric.data import Data

    seen = []

    def sample(block_sizes, edge_probs):
        seen.extend(block_sizes)
        return torch.tensor([[0, 10], [10, 0]])

    monkeypatch.setattr(datasets, "stochastic_blockmodel_graph", sample)
    data = Data(
        x=torch.ones(11, 3), y=torch.arange(11) % 4, edge_index=torch.tensor([[0, 1], [1, 0]])
    )
    result = datasets.create_sbm_dataset(data)
    assert sum(seen) == result.num_nodes == 11

import argparse

import pytest
import torch
from torch_geometric.data import Data

from geo_ood.gnnsafe import GNNSafe
from geo_ood.loss import loss_function
from geo_ood.parse import parser_add_main_args


def settings(backbone="gcn"):
    parser = argparse.ArgumentParser()
    parser_add_main_args(parser)
    return parser.parse_args(
        [
            "--backbone",
            backbone,
            "--hidden_channels",
            "8",
            "--use_occ",
            "--use_reg",
            "--use_prop",
            "--cpu",
        ]
    )


def graph():
    torch.manual_seed(8)
    edges = torch.stack([torch.arange(24), torch.arange(24).roll(-1)])
    data = Data(x=torch.rand(24, 6), edge_index=edges, y=(torch.arange(24) % 3).unsqueeze(1))
    data.splits = {
        "train": torch.arange(12),
        "valid": torch.arange(12, 18),
        "test": torch.arange(18, 24),
    }
    data.node_idx = torch.arange(24)
    return data


@pytest.mark.parametrize("backbone", ["gcn", "gat", "gen"])
def test_loss_scores_and_gradients(backbone):
    torch.manual_seed(123)
    args = settings(backbone)
    data = graph()
    model = GNNSafe(6, 3, args)
    model.init_center(data, torch.device("cpu"))
    loss = model.loss_compute(data, data, torch.nn.NLLLoss(), torch.device("cpu"), args)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.radius.grad.abs() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    scores = model.detect(data, data.node_idx, torch.device("cpu"), args)
    assert scores.shape == (24,) and torch.isfinite(scores).all()


def test_radius_gradient_and_run_reset():
    args = settings()
    model = GNNSafe(6, 3, args)
    loss, _, _ = loss_function(0.1, torch.zeros(8), torch.ones(4, 8), radius=model.radius)
    loss.backward()
    assert model.radius.grad.item() != 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.step()
    assert model.radius.item() != pytest.approx(0.1)
    model.center = torch.ones(8)
    model.reset_parameters()
    assert model.center is None
    assert model.radius.item() == pytest.approx(0.1)


def test_propagation_matches_source_incoming_average():
    model = GNNSafe(6, 3, settings())
    scores = torch.tensor([2.0, 20.0, 6.0])
    edges = torch.tensor([[0, 2], [1, 1]])
    result = model.propagation(scores, edges, prop_layers=1, alpha=0.5)
    assert torch.allclose(result, torch.tensor([1.0, 12.0, 3.0]))

import numpy as np
import pytest
import torch

from geo_ood.baselines import ODIN, Mahalanobis
from geo_ood.data_utils import evaluate_classify, eval_acc
from geo_ood.factory import build_detector
from test_model import graph, settings


def test_odin_returns_id_confidence(monkeypatch):
    args = settings()
    model = ODIN(6, 3, args)
    monkeypatch.setattr(model, "ODIN", lambda *a: np.array([[0.9, 0.1], [0.5, 0.5]]))
    score = model.detect(None, None, torch.device("cpu"), args)
    assert torch.allclose(score, torch.tensor([0.9, 0.5], dtype=score.dtype))


def test_mahalanobis_uses_selected_nodes_and_observed_classes():
    data = graph()
    data.y = (torch.arange(24) % 2 * 3 + 2).unsqueeze(1)
    selected = torch.tensor([7, 2, 18, 13, 20, 5])
    model = Mahalanobis(6, 6, settings())
    model.eval()
    with torch.no_grad():
        features = model.encoder.feature_list(data.x, data.edge_index)[1]
    means, precision = model.sample_estimator(
        2, [f.size(1) for f in features], data, selected, torch.device("cpu")
    )
    for layer, mean in zip(features, means):
        expected = torch.stack(
            [layer[selected][data.y[selected].flatten() == label].mean(0) for label in [2, 5]]
        )
        torch.testing.assert_close(mean, expected)
    assert all(torch.isfinite(p).all() for p in precision)


def test_classification_evaluation_uses_validation_split():
    args = settings()
    args.method = "msp"
    data = graph()
    model = build_detector(6, 3, args, torch.device("cpu"))
    result = evaluate_classify(model, data, eval_acc, torch.nn.NLLLoss(), args, "cpu")
    assert len(result) == 4
    assert all(np.isfinite(float(v)) for v in result)


def test_unknown_detector_has_actionable_error():
    args = settings()
    args.method = "missing"
    with pytest.raises(ValueError, match="choose from"):
        build_detector(6, 3, args, "cpu")

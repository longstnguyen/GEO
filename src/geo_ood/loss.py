import torch


def loss_function(nu, data_center, outputs, radius=0, mask=None):
    dist, scores = hypersphere_distance(data_center, outputs, radius, mask)
    loss = radius**2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    return loss, dist, scores


def hypersphere_distance(data_center, outputs, radius=0, mask=None):
    if mask is None:
        dist = torch.sum((outputs - data_center) ** 2, dim=1)
    else:
        dist = torch.sum((outputs[mask] - data_center) ** 2, dim=1)

    scores = dist - radius**2
    return dist, scores

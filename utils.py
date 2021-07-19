import torch
import copy
from collections import defaultdict


class Accumulator:
    def __init__(self):
        self.metrics = defaultdict(lambda: 0.)

    def add(self, key, value):
        self.metrics[key] += value

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def get_dict(self):
        return copy.deepcopy(dict(self.metrics))

    def items(self):
        return self.metrics.items()

    def __str__(self):
        return str(dict(self.metrics))

    def __truediv__(self, other):
        newone = Accumulator()
        for key, value in self.items():
            if isinstance(other, str):
                if other != key:
                    newone[key] = value / self[other]
                else:
                    newone[key] = value
            else:
                newone[key] = value / other
        return newone


def to_numpy(x):
    return x.detach().cpu().numpy()


def mean_field_logits(logits,
                      covmat=None,
                      mean_field_factor=1.,
                      likelihood='logistic'):
    """Adjust the model logits so its softmax approximates the posterior mean [1].
    Arguments:
    logits: A float tensor of shape (batch_size, num_classes).
    covmat: A float tensor of shape (batch_size, batch_size). If None then it
      assumes the covmat is an identity matrix.
    mean_field_factor: The scale factor for mean-field approximation, used to
      adjust the influence of posterior variance in posterior mean
      approximation. If covmat=None then it is used as the scaling parameter for
      temperature scaling.
    likelihood: Likelihood for integration in Gaussian-approximated latent
      posterior.
    Returns:
    True or False if `pred` has a constant boolean value, None otherwise.
    """
    if likelihood not in ('logistic', 'binary_logistic', 'poisson'):
        raise ValueError(
            f'Likelihood" must be one of (\'logistic\', \'binary_logistic\', \'poisson\'), got {likelihood}.'
        )

    if mean_field_factor < 0:
        return logits

    # Compute standard deviation.
    if covmat is None:
        variances = 1.
    else:
        variances = torch.diagonal(covmat)

    # Compute scaling coefficient for mean-field approximation.
    if likelihood == 'poisson':
        logits_scale = torch.exp(- variances * mean_field_factor / 2.)
    else:
        logits_scale = torch.sqrt(1. + variances * mean_field_factor)

    # Cast logits_scale to compatible dimension.
    if len(logits.shape) > 1:
        logits_scale = torch.unsqueeze(logits_scale, dim=-1)

    return logits / logits_scale

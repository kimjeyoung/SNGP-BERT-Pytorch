import torch


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

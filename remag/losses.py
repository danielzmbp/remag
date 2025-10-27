"""
Loss functions for contrastive learning in REMAG.

This module contains the loss functions used for training the Siamese network:
- BarlowTwinsLoss: Self-supervised contrastive learning
"""

import torch
import torch.nn as nn


class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins loss for self-supervised learning.

    The loss function computes the cross-correlation matrix between embeddings from two views
    and tries to make it as close as possible to the identity matrix. This encourages
    the network to produce similar embeddings for positive pairs while avoiding
    representational collapse by decorrelating different dimensions.

    Args:
        lambda_param: weight of the off-diagonal terms (decorrelation loss)
        eps: small value to avoid division by zero in normalization
    """

    def __init__(self, lambda_param=5e-3, eps=1e-6):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.eps = eps

    def forward(self, output1, output2, base_ids=None, return_stats=False):
        """
        Args:
            output1: a tensor of shape (batch_size, projection_dim)
            output2: a tensor of shape (batch_size, projection_dim)
            base_ids: unused in Barlow Twins but kept for compatibility
            return_stats: if True, return (loss, stats_dict) instead of just loss
        """
        batch_size, projection_dim = output1.shape

        # Normalize embeddings along the batch dimension (zero mean, unit std)
        output1_norm = (output1 - output1.mean(dim=0)) / (output1.std(dim=0) + self.eps)
        output2_norm = (output2 - output2.mean(dim=0)) / (output2.std(dim=0) + self.eps)

        # Compute cross-correlation matrix
        cross_corr = torch.matmul(output1_norm.T, output2_norm) / batch_size

        # Compute invariance loss (diagonal terms should be close to 1)
        invariance_loss = torch.pow(torch.diagonal(cross_corr) - 1.0, 2).sum()

        # Compute redundancy reduction loss (off-diagonal terms should be close to 0)
        off_diagonal_mask = ~torch.eye(projection_dim, dtype=torch.bool, device=output1.device)
        redundancy_loss = torch.pow(cross_corr[off_diagonal_mask], 2).sum()

        # Total loss
        loss = invariance_loss + self.lambda_param * redundancy_loss

        # Optionally compute statistics for debugging
        if return_stats:
            with torch.no_grad():
                diagonal = torch.diagonal(cross_corr)
                off_diagonal = cross_corr[off_diagonal_mask]

                stats = {
                    'mean_diagonal': diagonal.mean().item(),
                    'std_diagonal': diagonal.std().item(),
                    'mean_abs_off_diagonal': off_diagonal.abs().mean().item(),
                    'max_abs_off_diagonal': off_diagonal.abs().max().item(),
                    'invariance_loss': invariance_loss.item(),
                    'redundancy_loss': redundancy_loss.item(),
                }
                return loss, stats

        return loss

"""
Loss functions for contrastive learning in REMAG.

This module contains the loss functions used for training the Siamese network:
- BarlowTwinsLoss: Self-supervised contrastive learning
- SCGAuxiliaryLoss: Single copy core gene-aware penalty
- HybridContrastiveLoss: Combined loss for SCG-aware training
"""

import torch
import torch.nn as nn
from loguru import logger


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


class SCGAuxiliaryLoss(nn.Module):
    """
    Single Copy Core Gene (SCG) aware auxiliary loss.

    This loss component encourages the network to learn representations that respect
    SCG constraints:
    - Contigs sharing the same SCG should be pushed apart (duplication penalty)
    - Contigs with complementary SCG patterns can be slightly attracted (optional)

    The loss is designed to work alongside the main Barlow Twins loss to provide
    additional biological constraints during training.

    Args:
        duplication_weight: Weight for the duplication penalty term
        margin: Margin for the duplication penalty (default: 1.0)
        use_co_occurrence: Whether to use co-occurrence bonus (default: False)
        co_occurrence_weight: Weight for co-occurrence term if enabled
    """

    def __init__(self, duplication_weight=1.0, margin=1.0,
                 use_co_occurrence=False, co_occurrence_weight=0.1):
        super(SCGAuxiliaryLoss, self).__init__()
        self.duplication_weight = duplication_weight
        self.margin = margin
        self.use_co_occurrence = use_co_occurrence
        self.co_occurrence_weight = co_occurrence_weight
        self.eps = 1e-6

        logger.debug(f"Initialized SCGAuxiliaryLoss: duplication_weight={duplication_weight}, "
                    f"margin={margin}, use_co_occurrence={use_co_occurrence}")

    def forward(self, embeddings1, embeddings2, scg_features1, scg_features2, return_stats=False):
        """
        Compute SCG-aware auxiliary loss.

        IMPORTANT: embeddings1 and embeddings2 are positive pairs (fragments from same contigs).
        We only penalize NEGATIVE pairs (different contigs) that share SCGs.
        We compare within embeddings1 to avoid penalizing positive pairs.

        Args:
            embeddings1: Embeddings for batch 1 (batch_size, embedding_dim)
            embeddings2: Embeddings for batch 2 (batch_size, embedding_dim) - not used
            scg_features1: SCG features for batch 1 (batch_size, n_gene_families)
            scg_features2: SCG features for batch 2 (batch_size, n_gene_families) - not used
            return_stats: If True, return (loss, stats_dict)

        Returns:
            torch.Tensor: SCG auxiliary loss value
        """
        batch_size = embeddings1.shape[0]
        device = embeddings1.device

        # Check if SCG features are empty (no annotations available)
        if scg_features1.shape[1] == 0 or scg_features1.sum() == 0:
            # No SCG features - return zero loss
            if return_stats:
                return torch.tensor(0.0, device=device), {'duplication_loss': 0.0, 'n_duplicates': 0}
            return torch.tensor(0.0, device=device)

        # CRITICAL FIX: Use only embeddings1 to compare WITHIN batch
        # This avoids penalizing positive pairs (embeddings1[i], embeddings2[i])
        # Normalize embeddings for stable distance computation
        embeddings_norm = torch.nn.functional.normalize(embeddings1, p=2, dim=1)

        # Convert SCG features to binary (presence/absence)
        scg_binary = (scg_features1 > 0).float()

        # Compute SCG overlap WITHIN batch: which pairs share at least one SCG family
        # scg_overlap[i,j] = number of shared SCG families between contigs i and j
        scg_overlap = torch.matmul(scg_binary, scg_binary.T)  # (batch_size, batch_size)

        # Duplication mask: pairs that share at least one SCG (should be pushed apart)
        # EXCLUDE DIAGONAL: Don't penalize a contig for being similar to itself!
        duplication_mask = (scg_overlap > 0).float()
        diagonal_mask = torch.eye(batch_size, device=device)
        duplication_mask = duplication_mask * (1.0 - diagonal_mask)  # Zero out diagonal

        # Compute pairwise distances between normalized embeddings
        # Using L2 distance: ||emb_i - emb_j||
        embeddings_expanded1 = embeddings_norm.unsqueeze(1)  # (batch_size, 1, emb_dim)
        embeddings_expanded2 = embeddings_norm.unsqueeze(0)  # (1, batch_size, emb_dim)
        distances = torch.norm(embeddings_expanded1 - embeddings_expanded2, dim=2)  # (batch_size, batch_size)

        # Duplication penalty: push apart DIFFERENT contigs that share SCGs
        # Use hinge loss: max(0, margin - distance)
        # We want distance >= margin for pairs with shared SCGs
        duplication_loss = torch.relu(self.margin - distances) * duplication_mask
        duplication_loss = duplication_loss.sum() / (duplication_mask.sum() + self.eps)

        total_loss = self.duplication_weight * duplication_loss

        # Optional: Co-occurrence bonus (currently disabled by default)
        co_occurrence_loss = torch.tensor(0.0, device=device)
        if self.use_co_occurrence:
            # Compute complementarity: pairs with different SCGs might belong together
            # This is experimental and needs careful tuning
            co_occurrence_loss = self._compute_co_occurrence_loss(
                scg_binary, distances, diagonal_mask
            )
            total_loss += self.co_occurrence_weight * co_occurrence_loss

        # Statistics for monitoring
        if return_stats:
            with torch.no_grad():
                n_duplicates = duplication_mask.sum().item()
                avg_distance_duplicates = (distances * duplication_mask).sum() / (duplication_mask.sum() + self.eps)

                stats = {
                    'duplication_loss': duplication_loss.item(),
                    'n_duplicates': int(n_duplicates),
                    'avg_distance_duplicates': avg_distance_duplicates.item(),
                }

                if self.use_co_occurrence:
                    stats['co_occurrence_loss'] = co_occurrence_loss.item()

                return total_loss, stats

        return total_loss

    def _compute_co_occurrence_loss(self, scg_binary, distances, diagonal_mask):
        """
        Compute co-occurrence loss (experimental feature).

        The idea is to slightly encourage contigs with complementary SCG profiles
        to stay close, but this needs to be balanced carefully to avoid forcing
        unrelated contigs together.

        Args:
            scg_binary: Binary SCG matrix for batch (batch_size, n_gene_families)
            distances: Pairwise distances between embeddings (batch_size, batch_size)
            diagonal_mask: Mask for diagonal elements (batch_size, batch_size)

        Returns:
            torch.Tensor: Co-occurrence loss
        """
        # Count total SCGs per contig
        n_scg = scg_binary.sum(dim=1, keepdim=True)  # (batch_size, 1)

        # Compute overlap within batch
        overlap = torch.matmul(scg_binary, scg_binary.T)

        # Complementarity score: pairs with SCGs but low overlap
        # High score = good complementarity (different SCGs present)
        complementarity = ((n_scg + n_scg.T) - 2 * overlap) / (n_scg + n_scg.T + self.eps)

        # Only consider pairs with at least some SCGs, excluding diagonal
        has_scgs = ((n_scg > 0) & (n_scg.T > 0)).float()
        has_scgs = has_scgs * (1.0 - diagonal_mask)  # Exclude diagonal

        # Encourage complementary pairs to be closer (minimize distance)
        co_occurrence_loss = (distances * complementarity * has_scgs).sum() / (has_scgs.sum() + self.eps)

        return co_occurrence_loss


class HybridContrastiveLoss(nn.Module):
    """
    Hybrid loss combining Barlow Twins and SCG auxiliary losses.

    This loss function combines:
    1. Barlow Twins loss for k-mer and coverage-based similarity
    2. SCG auxiliary loss for biological constraints

    The final loss is:
        L = alpha * L_barlow_twins + beta * L_scg_auxiliary

    Args:
        alpha: Weight for Barlow Twins loss (default: 1.0)
        beta: Weight for SCG auxiliary loss (default: 0.1)
        barlow_twins_lambda: Lambda parameter for Barlow Twins (default: 5e-3)
        scg_margin: Margin for SCG duplication penalty (default: 1.0)
        use_scg_co_occurrence: Whether to use SCG co-occurrence bonus (default: False)
    """

    def __init__(self, alpha=1.0, beta=0.1, barlow_twins_lambda=5e-3,
                 scg_margin=1.0, use_scg_co_occurrence=False):
        super(HybridContrastiveLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.barlow_twins_loss = BarlowTwinsLoss(lambda_param=barlow_twins_lambda)
        self.scg_auxiliary_loss = SCGAuxiliaryLoss(
            duplication_weight=1.0,
            margin=scg_margin,
            use_co_occurrence=use_scg_co_occurrence,
            co_occurrence_weight=0.1
        )

        logger.info(f"Initialized HybridContrastiveLoss: alpha={alpha}, beta={beta}, "
                   f"barlow_lambda={barlow_twins_lambda}, scg_margin={scg_margin}")

    def forward(self, output1, output2, embeddings1, embeddings2,
                scg_features1, scg_features2, base_ids=None, return_stats=False):
        """
        Compute hybrid loss.

        Args:
            output1: Projections from model for batch 1 (for Barlow Twins)
            output2: Projections from model for batch 2 (for Barlow Twins)
            embeddings1: Embeddings for batch 1 (for SCG loss)
            embeddings2: Embeddings for batch 2 (for SCG loss)
            scg_features1: SCG features for batch 1
            scg_features2: SCG features for batch 2
            base_ids: Base contig IDs (unused, for compatibility)
            return_stats: If True, return (loss, stats_dict)

        Returns:
            torch.Tensor or tuple: Loss value, or (loss, stats) if return_stats=True
        """
        # Compute Barlow Twins loss on projections
        if return_stats:
            barlow_loss, barlow_stats = self.barlow_twins_loss(
                output1, output2, base_ids, return_stats=True
            )
        else:
            barlow_loss = self.barlow_twins_loss(output1, output2, base_ids)
            barlow_stats = None

        # Compute SCG auxiliary loss on embeddings
        if return_stats:
            scg_loss, scg_stats = self.scg_auxiliary_loss(
                embeddings1, embeddings2, scg_features1, scg_features2, return_stats=True
            )
        else:
            scg_loss = self.scg_auxiliary_loss(
                embeddings1, embeddings2, scg_features1, scg_features2
            )
            scg_stats = None

        # Combine losses
        total_loss = self.alpha * barlow_loss + self.beta * scg_loss

        if return_stats:
            # Flatten barlow stats to top level for backward compatibility
            stats = {
                'total_loss': total_loss.item(),
                'barlow_loss': barlow_loss.item(),
                'scg_loss': scg_loss.item(),
            }

            # Add barlow stats at top level (for existing logging code)
            if barlow_stats is not None:
                stats.update(barlow_stats)

            # Add SCG stats with prefix
            if scg_stats is not None:
                for key, value in scg_stats.items():
                    stats[f'scg_{key}'] = value

            return total_loss, stats

        return total_loss

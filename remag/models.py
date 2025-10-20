"""
Neural network models for REMAG
"""

import itertools
import numpy as np
import os
import random
import re

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from loguru import logger
from .utils import get_torch_device


class EarlyStoppingManager:
    """Manages early stopping logic during training."""
    
    def __init__(self, patience=20):
        self.patience = patience
        self.best_loss = float("inf")
        self.best_model_state = None
        self.epochs_no_improve = 0
    
    def check_improvement(self, current_loss, model_state):
        """Check if current loss is an improvement and update state."""
        if current_loss < self.best_loss:
            import copy

            self.best_loss = current_loss
            self.best_model_state = copy.deepcopy(model_state)
            self.epochs_no_improve = 0
            return True
        else:
            self.epochs_no_improve += 1
            return False
    
    def should_stop(self):
        """Check if training should stop based on patience."""
        return self.epochs_no_improve >= self.patience
    
    def get_best_state(self):
        """Get the best model state and loss."""
        return self.best_model_state, self.best_loss


class LearningRateScheduler:
    """Handles learning rate scheduling setup."""
    
    @staticmethod
    def create_warmup_cosine_scheduler(optimizer, args):
        """Create a warmup + cosine annealing scheduler."""
        base_learning_rate = getattr(args, 'base_learning_rate', 1e-3)
        scaled_lr = (args.batch_size / 256) * base_learning_rate * 0.2
        warmup_epochs = 10
        warmup_start_lr = scaled_lr * 0.1
        
        # Update optimizer's initial learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_start_lr
        
        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                target_multiplier = scaled_lr / warmup_start_lr
                return 1.0 + (target_multiplier - 1.0) * epoch / warmup_epochs
            else:
                cosine_epoch = epoch - warmup_epochs
                cosine_total = args.epochs - warmup_epochs
                if cosine_total <= 0:
                    return scaled_lr / warmup_start_lr

                min_lr_factor = 0.01
                max_multiplier = scaled_lr / warmup_start_lr
                min_multiplier = (scaled_lr * min_lr_factor) / warmup_start_lr
                cosine_factor = 0.5 * (1 + np.cos(np.pi * cosine_epoch / cosine_total))
                return min_multiplier + (max_multiplier - min_multiplier) * cosine_factor
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)


class TrainingManager:
    """Manages the complete training process."""
    
    def __init__(self, args):
        self.args = args
        self.early_stopping = EarlyStoppingManager(patience=20)
        self.device = get_torch_device()
    
    def setup_training(self, model, features_df):
        """Set up training components (dataset, dataloader, optimizer, scheduler)."""
        dataset = SequenceDataset(features_df, max_positive_pairs=self.args.max_positive_pairs)
        has_enough_data = len(dataset) > self.args.batch_size * 10

        dataloader_kwargs = {
            "batch_size": self.args.batch_size,
            "shuffle": True,
            "drop_last": not has_enough_data,
        }
        if self.device.type == "cuda":
            dataloader_kwargs["num_workers"] = self.args.cores if self.args.cores > 0 else 4
            dataloader_kwargs["pin_memory"] = True

        dataloader = DataLoader(dataset, **dataloader_kwargs)
        
        optimizer = optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0.05, betas=(0.9, 0.95)
        )
        
        scheduler = LearningRateScheduler.create_warmup_cosine_scheduler(optimizer, self.args)
        criterion = BarlowTwinsLoss(lambda_param=5e-3)

        return dataloader, optimizer, scheduler, criterion
    
    def train_epoch(self, model, dataloader, optimizer, criterion):
        """Train for one epoch and return average loss."""
        model.train()
        running_loss = 0.0
        matrix_stats = None

        for batch_idx, (features1, features2, base_ids) in enumerate(dataloader):
            features1, features2, base_ids = (
                features1.to(self.device),
                features2.to(self.device),
                base_ids.to(self.device),
            )

            optimizer.zero_grad()
            output1, output2 = model(features1, features2)

            # Get matrix statistics from last batch for debug logging
            is_last_batch = (batch_idx == len(dataloader) - 1)
            if is_last_batch:
                loss, matrix_stats = criterion(output1, output2, base_ids, return_stats=True)
            else:
                loss = criterion(output1, output2, base_ids)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        return running_loss / len(dataloader), matrix_stats


def get_model_path(args):
    return os.path.join(args.output, "siamese_model.pt")


class AdaptiveDropout(nn.Module):
    """Adaptive dropout that adjusts dropout rate based on input statistics."""

    def __init__(self, base_rate=0.1, max_rate=0.3, adaptation_factor=0.1):
        super(AdaptiveDropout, self).__init__()
        self.base_rate = base_rate
        self.max_rate = max_rate
        self.adaptation_factor = adaptation_factor

    def forward(self, x):
        if not self.training:
            return x

        # Compute input variance as a proxy for feature reliability
        input_var = torch.var(x, dim=-1, keepdim=True)
        normalized_var = torch.sigmoid(input_var * self.adaptation_factor)

        # Higher variance = higher dropout (less reliable features)
        adaptive_rate = self.base_rate + (self.max_rate - self.base_rate) * normalized_var

        # Apply dropout with adaptive rates
        dropout_mask = torch.bernoulli(1 - adaptive_rate)
        return x * dropout_mask / (1 - adaptive_rate + 1e-8)


class EnhancedFusionLayer(nn.Module):
    """Enhanced fusion layer with bidirectional attention and gated fusion.

    Uses ReLU activation in fusion/projection layers for stronger non-linearity
    and better gradient flow in complex fusion operations.
    """

    # Class constants for better maintainability
    DEFAULT_NUM_HEADS = 4
    DEFAULT_DROPOUT = 0.1
    GATE_INPUT_MULTIPLIER = 2  # For concatenated features in gates
    MULTI_SCALE_CONCAT_MULTIPLIER = 6  # 3 scales * 2 (kmer+coverage) features each
    COMPRESSOR_HIDDEN_MULTIPLIER = 4
    COMPRESSOR_OUTPUT_MULTIPLIER = 2
    INTERACTION_HIDDEN_MULTIPLIER = 3
    INTERACTION_INTERMEDIATE_MULTIPLIER = 2
    FINAL_FUSION_INPUT_MULTIPLIER = 4  # gated_kmer + gated_coverage + interaction + alignment
    FINAL_FUSION_HIDDEN_MULTIPLIER = 2
    RESIDUAL_WEIGHT_INIT = 0.5
    ADAPTIVE_DROPOUT_MAX_MULTIPLIER = 2.0

    def __init__(self, kmer_dim, coverage_dim, embedding_dim,
                 num_heads=DEFAULT_NUM_HEADS, dropout=DEFAULT_DROPOUT):
        super(EnhancedFusionLayer, self).__init__()

        self.embedding_dim = embedding_dim

        # Project to common dimension
        self.kmer_proj = nn.Linear(kmer_dim, embedding_dim)
        self.coverage_proj = nn.Linear(coverage_dim, embedding_dim)

        # Bidirectional cross-attention
        self.kmer_to_coverage_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.coverage_to_kmer_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )


        # Gated fusion mechanism
        self.kmer_gate = nn.Sequential(
            nn.Linear(embedding_dim * self.GATE_INPUT_MULTIPLIER, embedding_dim),
            nn.Sigmoid()
        )
        self.coverage_gate = nn.Sequential(
            nn.Linear(embedding_dim * self.GATE_INPUT_MULTIPLIER, embedding_dim),
            nn.Sigmoid()
        )

        # Multi-scale feature fusion (all project back to embedding_dim for consistent concatenation)
        self.multi_scale_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(embedding_dim // 2, embedding_dim)  # Project back to common dim
            ),
            nn.Linear(embedding_dim, embedding_dim),  # Identity scale
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 2),
                nn.ReLU(),
                nn.Linear(embedding_dim * 2, embedding_dim)  # Project back to common dim
            )
        ])

        # Batch normalization and residual connections
        self.batch_norm1 = nn.BatchNorm1d(embedding_dim)
        self.batch_norm2 = nn.BatchNorm1d(embedding_dim)
        self.residual_weight = nn.Parameter(torch.tensor(self.RESIDUAL_WEIGHT_INIT))

        # Multi-scale compressor to reduce dimensions for interaction module
        self.multi_scale_compressor = nn.Sequential(
            nn.Linear(embedding_dim * self.MULTI_SCALE_CONCAT_MULTIPLIER, embedding_dim * self.COMPRESSOR_HIDDEN_MULTIPLIER),
            nn.BatchNorm1d(embedding_dim * self.COMPRESSOR_HIDDEN_MULTIPLIER),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * self.COMPRESSOR_HIDDEN_MULTIPLIER, embedding_dim * self.COMPRESSOR_OUTPUT_MULTIPLIER)
        )

        # Feature interaction module
        self.interaction_module = nn.Sequential(
            nn.Linear(embedding_dim * self.COMPRESSOR_OUTPUT_MULTIPLIER, embedding_dim * self.INTERACTION_HIDDEN_MULTIPLIER),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * self.INTERACTION_HIDDEN_MULTIPLIER, embedding_dim * self.INTERACTION_INTERMEDIATE_MULTIPLIER),
            nn.ReLU(),
            nn.Linear(embedding_dim * self.INTERACTION_INTERMEDIATE_MULTIPLIER, embedding_dim)
        )

        # Adaptive dropout
        self.adaptive_dropout = AdaptiveDropout(
            base_rate=dropout,
            max_rate=dropout * self.ADAPTIVE_DROPOUT_MAX_MULTIPLIER
        )

        # Final fusion MLP with improved architecture
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim * self.FINAL_FUSION_INPUT_MULTIPLIER, embedding_dim * self.FINAL_FUSION_HIDDEN_MULTIPLIER),
            nn.BatchNorm1d(embedding_dim * self.FINAL_FUSION_HIDDEN_MULTIPLIER),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * self.FINAL_FUSION_HIDDEN_MULTIPLIER, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )

    def forward(self, kmer_features, coverage_features):
        # Input validation
        if kmer_features.dim() != 2 or coverage_features.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got kmer: {kmer_features.dim()}D, "
                           f"coverage: {coverage_features.dim()}D")

        if kmer_features.size(0) != coverage_features.size(0):
            raise ValueError(f"Batch size mismatch: kmer {kmer_features.size(0)} "
                           f"vs coverage {coverage_features.size(0)}")

        if kmer_features.size(1) != self.kmer_proj.in_features:
            raise ValueError(f"K-mer feature dimension mismatch: expected "
                           f"{self.kmer_proj.in_features}, got {kmer_features.size(1)}")

        if coverage_features.size(1) != self.coverage_proj.in_features:
            raise ValueError(f"Coverage feature dimension mismatch: expected "
                           f"{self.coverage_proj.in_features}, got {coverage_features.size(1)}")

        batch_size = kmer_features.size(0)

        # Project to common dimension
        kmer_proj = self.kmer_proj(kmer_features)  # [B, embed_dim]
        coverage_proj = self.coverage_proj(coverage_features)  # [B, embed_dim]

        # Add sequence dimension for attention
        kmer_seq = kmer_proj.unsqueeze(1)  # [B, 1, embed_dim]
        coverage_seq = coverage_proj.unsqueeze(1)  # [B, 1, embed_dim]

        # Bidirectional cross-attention
        # K-mer attending to coverage
        kmer_attended, kmer_attn_weights = self.kmer_to_coverage_attn(
            kmer_seq, coverage_seq, coverage_seq
        )
        kmer_attended = kmer_attended.squeeze(1)  # [B, embed_dim]

        # Coverage attending to k-mer
        coverage_attended, coverage_attn_weights = self.coverage_to_kmer_attn(
            coverage_seq, kmer_seq, kmer_seq
        )
        coverage_attended = coverage_attended.squeeze(1)  # [B, embed_dim]

        # Gated fusion mechanism
        combined_for_gate = torch.cat([kmer_proj, coverage_proj], dim=1)
        kmer_gate_weight = self.kmer_gate(combined_for_gate)
        coverage_gate_weight = self.coverage_gate(combined_for_gate)

        # Apply gates
        gated_kmer = kmer_gate_weight * kmer_proj + (1 - kmer_gate_weight) * kmer_attended
        gated_coverage = coverage_gate_weight * coverage_proj + (1 - coverage_gate_weight) * coverage_attended

        # Apply batch normalization with improved residual connections
        # Use learnable weighting with better gradient flow
        residual_weight_clamped = torch.clamp(self.residual_weight, min=0.1, max=0.9)

        gated_kmer = self.batch_norm1(
            (1 - residual_weight_clamped) * gated_kmer + residual_weight_clamped * kmer_proj
        )
        gated_coverage = self.batch_norm2(
            (1 - residual_weight_clamped) * gated_coverage + residual_weight_clamped * coverage_proj
        )

        # Multi-scale feature processing
        scale_features = []
        for scale_layer in self.multi_scale_fusion:
            scale_kmer = scale_layer(gated_kmer)
            scale_coverage = scale_layer(gated_coverage)
            scale_features.append(torch.cat([scale_kmer, scale_coverage], dim=1))

        # Concatenate multi-scale features (now all have consistent dimensions)
        # Each scale produces embedding_dim features for both kmer and coverage -> embedding_dim * 2 per scale
        # 3 scales * embedding_dim * 2 = embedding_dim * 6 total
        multi_scale_concat = torch.cat(scale_features, dim=1)  # [B, embedding_dim * 6]

        # Feature interaction module expects embedding_dim * 2, so we need to compress first
        multi_scale_compressed = self.multi_scale_compressor(multi_scale_concat)
        interaction_features = self.interaction_module(multi_scale_compressed)

        # Apply adaptive dropout
        interaction_features = self.adaptive_dropout(interaction_features)

        # Final fusion with all components
        final_features = torch.cat([
            gated_kmer,
            gated_coverage,
            interaction_features,
            kmer_attended + coverage_attended  # Cross-modal alignment signal
        ], dim=1)

        # Final MLP
        output = self.fusion_mlp(final_features)

        return output


class SiameseNetwork(nn.Module):
    def __init__(self, n_kmer_features, n_coverage_features, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        
        self.n_kmer_features = n_kmer_features
        self.n_coverage_features = n_coverage_features
        self.has_coverage = n_coverage_features > 0
        
        # Separate encoders for k-mer and coverage features
        # Using LeakyReLU in encoders to prevent dead neurons during feature extraction
        self.kmer_encoder = nn.Sequential(
            nn.Linear(n_kmer_features, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        
        if self.has_coverage:
            # Adaptively size the coverage encoder based on number of samples
            # Assume ~2 features per sample (mean + std), so n_samples ≈ n_coverage_features / 2
            n_samples_estimate = max(1, n_coverage_features // 2)
            
            # Scale hidden dimensions based on number of samples
            # More samples = more complex co-abundance patterns = larger encoder
            if n_samples_estimate <= 2:
                coverage_hidden1 = 32
                coverage_hidden2 = 16
            elif n_samples_estimate <= 5:
                coverage_hidden1 = 64
                coverage_hidden2 = 32
            elif n_samples_estimate <= 10:
                coverage_hidden1 = 128
                coverage_hidden2 = 64
            else:
                coverage_hidden1 = 256
                coverage_hidden2 = 128
                
            logger.debug(f"Coverage encoder sized for ~{n_samples_estimate} samples: "
                        f"{n_coverage_features} -> {coverage_hidden1} -> {coverage_hidden2}")
            
            self.coverage_encoder = nn.Sequential(
                nn.Linear(n_coverage_features, coverage_hidden1),
                nn.BatchNorm1d(coverage_hidden1),
                nn.LeakyReLU(),
                nn.Dropout(0.05),
                nn.Linear(coverage_hidden1, coverage_hidden2),
                nn.BatchNorm1d(coverage_hidden2),
                nn.LeakyReLU(),
            )
            
            # Store final coverage dimension for fusion layer
            self.coverage_final_dim = coverage_hidden2
            
            # Enhanced fusion layer with bidirectional attention and gated fusion
            self.fusion_layer = EnhancedFusionLayer(
                kmer_dim=128,
                coverage_dim=self.coverage_final_dim,
                embedding_dim=embedding_dim,
                num_heads=4,
                dropout=0.1
            )
        else:
            # No coverage features - use a simple projection
            self.kmer_projection = nn.Sequential(
                nn.Linear(128, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
            )
        
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

    def _encode_features(self, x):
        """Internal method to encode features using appropriate architecture"""
        # Split input into k-mer and coverage features
        kmer_features = x[:, :self.n_kmer_features]
        
        # Encode k-mer features
        kmer_encoded = self.kmer_encoder(kmer_features)
        
        if self.has_coverage:
            # Normal path with coverage features
            coverage_features = x[:, self.n_kmer_features:]
            coverage_encoded = self.coverage_encoder(coverage_features)
            
            # Advanced fusion with cross-attention
            representation = self.fusion_layer(kmer_encoded, coverage_encoded)
        else:
            # No coverage features - use simple projection
            representation = self.kmer_projection(kmer_encoded)
        
        return representation

    def forward_one(self, x):
        # Used for training, returns projection
        representation = self._encode_features(x)
        projection = self.projection_head(representation)
        return projection

    def get_embedding(self, x):
        # Used for inference, returns representation
        return self._encode_features(x)

    def forward(self, x1, x2):
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)
        return output1, output2



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


class SequenceDataset(Dataset):
    def __init__(self, features_df, max_positive_pairs=500000):
        """Initialize contrastive learning dataset with positive pairs from same contigs."""
        self.features_df = features_df
        self.fragment_headers = self.features_df.index.tolist()

        # Group fragment indices by base contig name
        self.contig_to_fragment_indices = self._group_indices_by_base_contig()

        self.base_name_to_id = {
            name: i for i, name in enumerate(self.contig_to_fragment_indices.keys())
        }
        self.index_to_base_id = {}
        for base_name, fragment_indices in self.contig_to_fragment_indices.items():
            base_id = self.base_name_to_id[base_name]
            for frag_idx in fragment_indices:
                self.index_to_base_id[frag_idx] = base_id

        original_count = len(self.contig_to_fragment_indices)
        self.contig_to_fragment_indices = {
            base_name: indices
            for base_name, indices in self.contig_to_fragment_indices.items()
            if len(indices) > 1
        }
        
        logger.debug(f"Filtered to {len(self.contig_to_fragment_indices)} contigs with multiple fragments (removed {original_count - len(self.contig_to_fragment_indices)})")

        if not self.contig_to_fragment_indices:
            raise ValueError(
                "No base contigs found with multiple fragments. Cannot generate positive pairs."
            )

        # Generate and select positive pairs
        all_potential_pairs = self._generate_all_positive_pairs()
        self.training_pairs = self._select_positive_pairs(
            all_potential_pairs, max_positive_pairs
        )

        if len(self.training_pairs) == 0:
            raise ValueError("No positive pairs selected. Training cannot proceed.")

        random.shuffle(self.training_pairs)
        logger.info(
            f"Dataset initialized with {len(self.training_pairs)} positive pairs"
        )

    def _group_indices_by_base_contig(self):
        """Group fragment indices by their original contig's base name."""
        groups = {}
        for i, fragment_header in enumerate(self.fragment_headers):
            # Match patterns: .original, .h1.N, .h2.N, or .N (where N is a number)
            match = re.match(r"(.+)\.(?:h[12]\.(\d+)|(\d+)|original)$", fragment_header)
            if match:
                base_name = match.group(1)
            else:
                base_name = fragment_header
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(i)
        return groups

    def _generate_all_positive_pairs(self):
        all_pairs = []
        for base_name, fragment_indices in self.contig_to_fragment_indices.items():
            for i, j in itertools.combinations(range(len(fragment_indices)), 2):
                idx1 = fragment_indices[i]
                idx2 = fragment_indices[j]
                all_pairs.append((idx1, idx2))
        return all_pairs

    def _select_positive_pairs(self, all_potential_pairs, max_cap):
        if len(all_potential_pairs) > max_cap:
            return random.sample(all_potential_pairs, max_cap)
        return all_potential_pairs

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, idx):
        idx1, idx2 = self.training_pairs[idx]
        tensor1 = torch.tensor(self.features_df.iloc[idx1].values, dtype=torch.float32)
        tensor2 = torch.tensor(self.features_df.iloc[idx2].values, dtype=torch.float32)
        base_id = torch.tensor(self.index_to_base_id[idx1], dtype=torch.long)
        return tensor1, tensor2, base_id


def train_siamese_network(features_df, args):
    """Train the Siamese network for contrastive learning."""
    model_path = get_model_path(args)

    # Feature dimensions: k-mer features are always 136, coverage is 2 per sample
    n_kmer_features = 136
    total_features = features_df.shape[1]
    n_coverage_features = total_features - n_kmer_features
    
    logger.info(f"Using dual-encoder architecture: {n_kmer_features} k-mer + {n_coverage_features} coverage features")

    # Load existing model if available
    if os.path.exists(model_path):
        logger.info(f"Loading existing model from {model_path}")
        device = get_torch_device()
        model = SiameseNetwork(
            n_kmer_features=n_kmer_features, 
            n_coverage_features=n_coverage_features,
            embedding_dim=args.embedding_dim
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        return model

    device = get_torch_device()

    dataset = SequenceDataset(features_df, max_positive_pairs=args.max_positive_pairs)
    has_enough_data = len(dataset) > args.batch_size * 10

    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": not has_enough_data,
    }
    if device.type == "cuda":
        dataloader_kwargs["num_workers"] = args.cores if args.cores > 0 else 4
        dataloader_kwargs["pin_memory"] = True

    dataloader = DataLoader(dataset, **dataloader_kwargs)

    if len(dataloader) == 0:
        logger.warning(
            f"DataLoader is empty (Dataset size: {len(dataset)} < Batch size: {args.batch_size}). "
            f"Creating untrained model. This typically happens with very small datasets. "
            f"Consider reducing batch size with --batch-size {max(1, len(dataset)//2)}."
        )
        # Create untrained model to avoid crashes
        model = SiameseNetwork(
            n_kmer_features=n_kmer_features,
            n_coverage_features=n_coverage_features,
            embedding_dim=args.embedding_dim
        ).to(device)
        torch.save(model.state_dict(), model_path)
        return model

    # Initialize model, loss, optimizer
    model = SiameseNetwork(
        n_kmer_features=n_kmer_features,
        n_coverage_features=n_coverage_features,
        embedding_dim=args.embedding_dim
    ).to(device)
    logger.info(f"Starting training for {args.epochs} epochs...")

    # Initialize training manager and set up training
    trainer = TrainingManager(args)
    dataloader, optimizer, scheduler, criterion = trainer.setup_training(model, features_df)

    # Training loop
    epoch_progress = tqdm(range(args.epochs), desc="Training Progress")
    
    for epoch in epoch_progress:
        avg_loss, matrix_stats = trainer.train_epoch(model, dataloader, optimizer, criterion)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        logger.debug(f"Epoch {epoch+1}/{args.epochs} - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

        # Log cross-correlation matrix statistics for debugging
        if matrix_stats is not None:
            logger.debug(
                f"Cross-correlation matrix stats - "
                f"Diagonal: {matrix_stats['mean_diagonal']:.4f} ± {matrix_stats['std_diagonal']:.4f}, "
                f"Off-diagonal: {matrix_stats['mean_abs_off_diagonal']:.4f} (max: {matrix_stats['max_abs_off_diagonal']:.4f}), "
                f"Invariance loss: {matrix_stats['invariance_loss']:.2f}, "
                f"Redundancy loss: {matrix_stats['redundancy_loss']:.2f}"
            )

        # Early stopping check
        trainer.early_stopping.check_improvement(avg_loss, model.state_dict())
        if trainer.early_stopping.should_stop():
            logger.info(f"Early stopping after {epoch+1} epochs (patience: {trainer.early_stopping.patience})")
            break

        # Update progress bar
        epoch_progress.set_postfix({
            "Loss": f"{avg_loss:.4f}",
            "LR": f"{current_lr:.2e}",
            "Best": f"{trainer.early_stopping.best_loss:.4f}"
        })

        # Print to screen every 20 epochs or on the last epoch
        if (epoch + 1) % 20 == 0 or epoch == args.epochs - 1:
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

    # Load best model
    best_state, best_loss = trainer.early_stopping.get_best_state()
    if best_state:
        model.load_state_dict(best_state)
        logger.info(f"Loaded best model with loss: {best_loss:.4f}")

    # Save model only if keeping intermediate files
    if getattr(args, "keep_intermediate", False):
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
    
    return model


def generate_embeddings(model, features_df, args):
    """Generate embeddings for all features using the trained model."""
    embeddings_path = os.path.join(args.output, "embeddings.csv")

    # Check if embeddings file already exists
    if os.path.exists(embeddings_path):
        logger.info(f"Loading existing embeddings from {embeddings_path}")
        return pd.read_csv(embeddings_path, index_col=0)

    device = get_torch_device()
    logger.debug(f"Using device: {device}")

    model.eval()
    embeddings = {}

    # Filter to only original contigs (ending with .original) for efficiency
    original_contig_mask = features_df.index.str.endswith(".original")
    original_features_df = features_df[original_contig_mask].copy()

    logger.debug(
        f"Generating embeddings for {len(original_features_df)} original contigs (filtered from {len(features_df)} total fragments)..."
    )

    with torch.no_grad():
        batch_size = args.batch_size
        for i in range(0, len(original_features_df), batch_size):
            batch_df = original_features_df.iloc[i:i+batch_size]
            batch_features = torch.tensor(batch_df.values, dtype=torch.float32).to(device)
            batch_embeddings = model.get_embedding(batch_features)
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            
            for j, header in enumerate(batch_df.index):
                clean_header = header.replace(".original", "")
                embeddings[clean_header] = batch_embeddings[j].cpu().numpy()

    embeddings_df = pd.DataFrame.from_dict(embeddings, orient="index")

    # Always save embeddings for downstream analysis and visualization
    embeddings_df.to_csv(embeddings_path)
    logger.info(f"Embeddings saved to {embeddings_path}")

    return embeddings_df


def generate_embeddings_for_fragments(model, features_df, fragment_names, args):
    """Generate embeddings for specific fragments (e.g., h1/h2 fragments for chimera detection)."""
    device = get_torch_device()
    logger.debug(f"Using device: {device}")

    model.eval()
    model.to(device)
    embeddings = {}

    # Filter to only requested fragments
    fragment_mask = features_df.index.isin(fragment_names)
    fragment_features_df = features_df[fragment_mask].copy()

    logger.debug(
        f"Generating embeddings for {len(fragment_features_df)} fragments (filtered from {len(features_df)} total fragments)..."
    )

    if fragment_features_df.empty:
        logger.warning("No matching fragments found for embedding generation")
        return pd.DataFrame()

    with torch.no_grad():
        batch_size = args.batch_size
        for i in range(0, len(fragment_features_df), batch_size):
            batch_df = fragment_features_df.iloc[i:i+batch_size]
            batch_features = torch.tensor(batch_df.values, dtype=torch.float32).to(device)
            batch_embeddings = model.get_embedding(batch_features)
            
            for j, header in enumerate(batch_df.index):
                embeddings[header] = batch_embeddings[j].cpu().numpy()

    embeddings_df = pd.DataFrame.from_dict(embeddings, orient="index")
    logger.info(f"Generated embeddings for {len(embeddings_df)} fragments")
    return embeddings_df

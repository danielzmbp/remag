"""
Neural network models for REMAG
"""

import itertools
import numpy as np
import os
import random
import re
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from loguru import logger
from .utils import get_torch_device


def get_model_path(args):
    """Get the path for the Siamese model file."""
    return os.path.join(args.output, "siamese_model.pt")


class FusionLayer(nn.Module):
    def __init__(self, kmer_dim, coverage_dim, embedding_dim, hidden_dim=None):
        super(FusionLayer, self).__init__()
        
        # Use embedding_dim as the common dimension for simplicity
        self.kmer_proj = nn.Linear(kmer_dim, embedding_dim)
        self.coverage_proj = nn.Linear(coverage_dim, embedding_dim)
        
        # Single cross-attention layer (simplified from two separate ones)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        
        # Simplified fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
    def forward(self, kmer_features, coverage_features):
        # Project to common dimension
        kmer_proj = self.kmer_proj(kmer_features)
        coverage_proj = self.coverage_proj(coverage_features)
        
        kmer_seq = kmer_proj.unsqueeze(1)
        coverage_seq = coverage_proj.unsqueeze(1)
        
        # Cross-attention: let k-mer features attend to coverage
        attended_features, _ = self.cross_attention(
            kmer_seq, coverage_seq, coverage_seq
        )
        attended_features = attended_features.squeeze(1)
        
        # Concatenate original k-mer projection with attended features
        combined = torch.cat([kmer_proj, attended_features], dim=1)
        
        # Apply fusion MLP
        output = self.fusion_mlp(combined)
        
        return output



class SiameseNetwork(nn.Module):
    def __init__(self, n_kmer_features, n_coverage_features, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        
        self.n_kmer_features = n_kmer_features
        self.n_coverage_features = n_coverage_features
        self.has_coverage = n_coverage_features > 0
        
        # Separate encoders for k-mer and coverage features
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
            
            # Advanced fusion layer with cross-attention and MLP
            self.fusion_layer = FusionLayer(
                kmer_dim=128, 
                coverage_dim=self.coverage_final_dim, 
                embedding_dim=embedding_dim
            )
        else:
            # No coverage features - use a simple projection
            self.kmer_projection = nn.Sequential(
                nn.Linear(128, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
        
        # Simplified projection head - 512 dims is sufficient for this use case
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
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
    
    def forward(self, output1, output2, base_ids=None):
        """
        Args:
            output1: a tensor of shape (batch_size, projection_dim)
            output2: a tensor of shape (batch_size, projection_dim)  
            base_ids: unused in Barlow Twins but kept for compatibility
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
        """Generate pairs of fragment indices from the same base contig."""
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
    criterion = BarlowTwinsLoss(lambda_param=5e-3)

    # Barlow Twins uses different learning rate scaling
    base_learning_rate = getattr(args, 'base_learning_rate', 1e-3)
    # Barlow Twins typically uses linear scaling rule with batch size
    scaled_lr = (args.batch_size / 256) * base_learning_rate * 0.2  # Lower LR for Barlow Twins
    warmup_epochs = 10  # Longer warmup for Barlow Twins
    warmup_start_lr = scaled_lr * 0.1

    optimizer = optim.AdamW(
        model.parameters(), lr=warmup_start_lr, weight_decay=0.05, betas=(0.9, 0.95)
    )

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

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    logger.info(f"Starting training for {args.epochs} epochs...")

    # Early stopping parameters 
    patience = 20
    best_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    # Training loop
    epoch_progress = tqdm(range(args.epochs), desc="Training Progress")
    
    for epoch in epoch_progress:
        model.train()
        running_loss = 0.0

        for features1, features2, base_ids in dataloader:
            features1, features2, base_ids = (
                features1.to(device),
                features2.to(device),
                base_ids.to(device),
            )

            optimizer.zero_grad()
            output1, output2 = model(features1, features2)
            loss = criterion(output1, output2, base_ids)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        avg_loss = running_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]["lr"]

        # Always log to file
        logger.debug(
            f"Epoch {epoch+1}/{args.epochs} - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}"
        )
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs (patience: {patience})")
                break
        
        # Update epoch progress bar
        epoch_progress.set_postfix({
            "Loss": f"{avg_loss:.4f}",
            "LR": f"{current_lr:.2e}",
            "Best": f"{best_loss:.4f}"
        })
        
        # Print to screen every 5 epochs or on the last epoch
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
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
        import pandas as pd
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

    import pandas as pd
    embeddings_df = pd.DataFrame.from_dict(embeddings, orient="index")
    
    # Save embeddings only if keeping intermediate files
    if getattr(args, "keep_intermediate", False):
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
        import pandas as pd
        return pd.DataFrame()

    with torch.no_grad():
        batch_size = args.batch_size
        for i in range(0, len(fragment_features_df), batch_size):
            batch_df = fragment_features_df.iloc[i:i+batch_size]
            batch_features = torch.tensor(batch_df.values, dtype=torch.float32).to(device)
            batch_embeddings = model.get_embedding(batch_features)
            
            for j, header in enumerate(batch_df.index):
                embeddings[header] = batch_embeddings[j].cpu().numpy()

    import pandas as pd
    embeddings_df = pd.DataFrame.from_dict(embeddings, orient="index")
    logger.info(f"Generated embeddings for {len(embeddings_df)} fragments")
    return embeddings_df

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


class SiameseNetwork(nn.Module):
    def __init__(self, input_size=None, embedding_dim=128, n_kmer_features=None, n_coverage_features=None, n_lm_features=None):
        super(SiameseNetwork, self).__init__()
        
        if n_lm_features is not None and n_coverage_features is not None:
            # Language model + coverage features architecture
            self.dual_encoder = True
            self.use_language_model = True
            self.n_lm_features = n_lm_features
            self.n_coverage_features = n_coverage_features
            
            # Language model feature encoder (768 -> 256)
            self.lm_encoder = nn.Sequential(
                nn.Linear(n_lm_features, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
            )
            
            # Coverage encoder (same as before)
            self.coverage_encoder = nn.Sequential(
                nn.Linear(n_coverage_features, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.Dropout(0.05),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(),
            )
            
            # Fusion layer to combine encoded features
            fusion_input_size = 256 + 16  # lm_encoder output + coverage_encoder output
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_input_size, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Dropout(0.05),
                nn.Linear(256, embedding_dim),
            )
            
        elif n_kmer_features is not None and n_coverage_features is not None:
            # Original k-mer + coverage features architecture
            self.dual_encoder = True
            self.use_language_model = False
            self.n_kmer_features = n_kmer_features
            self.n_coverage_features = n_coverage_features
            
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
            
            self.coverage_encoder = nn.Sequential(
                nn.Linear(n_coverage_features, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.Dropout(0.05),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(),
            )
            
            # Fusion layer to combine encoded features
            fusion_input_size = 128 + 16  # kmer_encoder output + coverage_encoder output
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_input_size, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Dropout(0.05),
                nn.Linear(256, embedding_dim),
            )
            
        else:
            self.dual_encoder = False
            self.use_language_model = False
            if input_size is None:
                raise ValueError("Either provide input_size or both n_kmer_features and n_coverage_features or n_lm_features and n_coverage_features")
            
            # The base network generates the representations for downstream tasks
            self.base_network = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Linear(256, embedding_dim),
            )
        
        self.projection_head = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def _encode_features(self, x):
        """Internal method to encode features using appropriate architecture"""
        if self.dual_encoder:
            if self.use_language_model:
                # Split input into language model and coverage features
                lm_features = x[:, :self.n_lm_features]
                coverage_features = x[:, self.n_lm_features:]
                
                # Encode each feature type separately
                lm_encoded = self.lm_encoder(lm_features)
                coverage_encoded = self.coverage_encoder(coverage_features)
                
                # Fuse the encoded features
                fused_features = torch.cat([lm_encoded, coverage_encoded], dim=1)
                representation = self.fusion_layer(fused_features)
                return representation
            else:
                # Split input into k-mer and coverage features
                kmer_features = x[:, :self.n_kmer_features]
                coverage_features = x[:, self.n_kmer_features:]
                
                # Encode each feature type separately
                kmer_encoded = self.kmer_encoder(kmer_features)
                coverage_encoded = self.coverage_encoder(coverage_features)
                
                # Fuse the encoded features
                fused_features = torch.cat([kmer_encoded, coverage_encoded], dim=1)
                representation = self.fusion_layer(fused_features)
                return representation
        else:
            return self.base_network(x)

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


class InfoNCELoss(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This version uses a standard implementation that leverages all samples in the batch as negatives.
    It correctly handles masking of false negatives without corrupting the true positive pairs.

    Args:
        temperature: a float value for temperature scaling
        eps: a small value to avoid division by zero in normalization
    """

    def __init__(self, temperature=0.07, eps=1e-6):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.eps = eps
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def forward(self, output1, output2, base_ids):
        """
        Args:
            output1: a tensor of shape (batch_size, embedding_dim)
            output2: a tensor of shape (batch_size, embedding_dim)
            base_ids: a tensor of shape (batch_size) for identifying false negatives
        """
        batch_size = output1.shape[0]
        device = output1.device

        # Concatenate the two sets of features to create a 2N x D tensor
        features = torch.cat([output1, output2], dim=0)

        # Normalize features for cosine similarity. Add eps for numerical stability.
        features = nn.functional.normalize(features, p=2, dim=1, eps=self.eps)

        # Create a 2N x 2N similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # --- Create Labels for Positive Pairs ---
        # The positive for feature i (from output1) is feature i + batch_size (from output2)
        # and vice-versa.
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # We need to mask out "false negatives" (other fragments from the same contig)
        # without masking out the "true positives".

        # 1. Create a mask that identifies all pairs from the same contig
        all_base_ids = torch.cat([base_ids, base_ids], dim=0)
        mask_same_contig = all_base_ids.unsqueeze(1) == all_base_ids.unsqueeze(0)

        # 2. Create a mask that identifies the true positive pairs
        mask_positives = torch.zeros_like(mask_same_contig)
        mask_positives[torch.arange(2 * batch_size), labels] = 1

        # 3. Create the final mask for elements to be IGNORED in the loss.
        # These are pairs from the same contig that are NOT the true positive pair.
        # This also implicitly handles the self-similarity (diagonal) case because a
        # sample is never its own positive pair in this setup.
        mask_to_ignore = mask_same_contig & ~mask_positives.bool()

        # 4. Apply the mask by setting the logits of ignored pairs to a very low value.
        similarity_matrix[mask_to_ignore] = -1e9

        # --- Calculate Loss ---
        loss_vec = self.criterion(similarity_matrix, labels)

        # Calculate the mean loss (all pairs have equal weight)
        loss = loss_vec.mean()

        return loss


class SequenceDataset(Dataset):
    def __init__(self, features_df, max_positive_pairs=500000):
        """Initialize contrastive learning dataset with positive pairs from same contigs."""
        self.features_df = features_df
        self.fragment_headers = self.features_df.index.tolist()

        # Group fragment indices by base contig name
        self.contig_to_fragment_indices = self._group_indices_by_base_contig()

        # Create base name to ID mapping
        self.base_name_to_id = {
            name: i for i, name in enumerate(self.contig_to_fragment_indices.keys())
        }
        self.index_to_base_id = {}
        for base_name, fragment_indices in self.contig_to_fragment_indices.items():
            base_id = self.base_name_to_id[base_name]
            for frag_idx in fragment_indices:
                self.index_to_base_id[frag_idx] = base_id

        logger.debug(f"Found {len(self.contig_to_fragment_indices)} base contigs before filtering")
        for base_name, indices in list(self.contig_to_fragment_indices.items())[:5]:
            logger.debug(f"  {base_name}: {len(indices)} fragments")
        original_count = len(self.contig_to_fragment_indices)
        self.contig_to_fragment_indices = {
            base_name: indices
            for base_name, indices in self.contig_to_fragment_indices.items()
            if len(indices) > 1
        }
        
        logger.debug(f"After filtering: {len(self.contig_to_fragment_indices)} contigs with multiple fragments (removed {original_count - len(self.contig_to_fragment_indices)})")

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
    model_path = os.path.join(args.output, "siamese_model.pt")

    # Determine feature dimensions based on input
    total_features = features_df.shape[1]
    
    # Check if we have language model features (768 dimensions)
    if hasattr(args, 'use_language_model') and args.use_language_model:
        n_lm_features = 768  # Language model embedding dimension
        n_coverage_features = total_features - n_lm_features
        logger.info(f"Using dual-encoder architecture: {n_lm_features} language model + {n_coverage_features} coverage features")
        
        # Load existing model if available
        if os.path.exists(model_path):
            logger.info(f"Loading existing model from {model_path}")
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
            model = SiameseNetwork(
                n_lm_features=n_lm_features, 
                n_coverage_features=n_coverage_features,
                embedding_dim=args.embedding_dim
            ).to(device)
            model.load_state_dict(torch.load(model_path, weights_only=False))
            return model
    else:
        # Original k-mer features
        n_kmer_features = 136
        n_coverage_features = total_features - n_kmer_features
        logger.info(f"Using dual-encoder architecture: {n_kmer_features} k-mer + {n_coverage_features} coverage features")
        
        # Load existing model if available
        if os.path.exists(model_path):
            logger.info(f"Loading existing model from {model_path}")
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
            model = SiameseNetwork(
                n_kmer_features=n_kmer_features, 
                n_coverage_features=n_coverage_features,
                embedding_dim=args.embedding_dim
            ).to(device)
            model.load_state_dict(torch.load(model_path, weights_only=False))
            return model

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Create dataset and dataloader
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
        logger.error(
            f"DataLoader is empty! Dataset size: {len(dataset)}, Batch size: {args.batch_size}"
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
    if hasattr(args, 'use_language_model') and args.use_language_model:
        model = SiameseNetwork(
            n_lm_features=n_lm_features,
            n_coverage_features=n_coverage_features,
            embedding_dim=args.embedding_dim
        ).to(device)
    else:
        model = SiameseNetwork(
            n_kmer_features=n_kmer_features,
            n_coverage_features=n_coverage_features,
            embedding_dim=args.embedding_dim
        ).to(device)
    criterion = InfoNCELoss(temperature=args.nce_temperature)

    base_learning_rate = 1e-3
    scaled_lr = (args.batch_size / 256) * base_learning_rate
    warmup_epochs = 5
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
    patience = 10
    best_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for features1, features2, base_ids in progress_bar:
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

            # Update progress bar
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Update learning rate
        scheduler.step()

        # Calculate average loss for this epoch
        avg_loss = running_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
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

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with loss: {best_loss:.4f}")

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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
    )
    logger.info(f"Using device: {device}")
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
            
            for j, header in enumerate(batch_df.index):
                embeddings[header] = batch_embeddings[j].cpu().numpy()

    import pandas as pd
    embeddings_df = pd.DataFrame.from_dict(embeddings, orient="index")
    embeddings_df.to_csv(embeddings_path)
    logger.info(f"Embeddings saved to {embeddings_path}")
    return embeddings_df


def generate_embeddings_for_fragments(model, features_df, fragment_names, args):
    """Generate embeddings for specific fragments (e.g., h1/h2 fragments for chimera detection)."""
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
    )
    logger.info(f"Using device: {device}")
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

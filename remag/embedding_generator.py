"""
DNA sequence embedding generation using DNABERT-2 model
"""
from tqdm import tqdm
import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModel
from loguru import logger
import warnings

# Disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add model directory to Python path
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)


class DNAEmbeddingGenerator:
    """
    Generate sequence embeddings using DNABERT-2 model from llm/ directory
    """
    
    def __init__(self, model_path: str, device: str = None, max_length: int = 512):
        """
        Initialize the embedding generator
        
        Args:
            model_path: Path to the DNABERT-2 model directory
            device: Device to run model on ('cpu', 'cuda', or None for auto)
            max_length: Maximum sequence length for tokenization
        """
        self.model_path = model_path
        self.max_length = max_length
        
        # Set device with MPS support for Apple Silicon
        if device is None or device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device('mps')
                logger.info("Using Apple Silicon MPS acceleration")
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        
        # MPS-specific optimizations
        if self.device.type == 'mps':
            # Enable memory efficient attention for MPS
            try:
                torch.backends.mps.enable_memory_efficient_attention = True
                logger.debug("Enabled MPS memory efficient attention")
            except AttributeError:
                pass
        
    def _load_model(self):
        """Load the DNABERT-2 model and tokenizer"""
        try:
            logger.info(f"Loading DNABERT-2 model from {self.model_path}")
            
            # Try to load using the custom bert_layers module first
            try:
                from bert_layers import BertModel
                from configuration_bert import BertConfig
                
                # Load configuration
                config = BertConfig.from_pretrained(self.model_path)
                
                # Load model using custom BertModel
                self.model = BertModel.from_pretrained(self.model_path, config=config)
                self.model.to(self.device)
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                
                logger.info(f"Model loaded using custom BERT layers. Hidden size: {config.hidden_size}")
                
            except ImportError:
                # Fallback to standard HuggingFace loading
                logger.info("Custom BERT layers not found, using standard HuggingFace loading")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                
                # Load model
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                ).to(self.device)
                
                logger.info(f"Model loaded successfully. Hidden size: {self.model.config.hidden_size}")
            
            # Set to evaluation mode
            self.model.eval()
            logger.info(f"Model is on device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _prepare_sequence(self, sequence: str) -> str:
        """
        Prepare DNA sequence for tokenization
        
        Args:
            sequence: Raw DNA sequence
            
        Returns:
            Prepared sequence string
        """
        # Convert to uppercase and ensure only valid nucleotides
        clean_seq = ''.join(c for c in sequence.upper() if c in 'ATGC')
        
        # Truncate if too long
        if len(clean_seq) > self.max_length - 2:  # Account for special tokens
            clean_seq = clean_seq[:self.max_length - 2]
            
        return clean_seq
    
    def _get_sequence_embedding(self, sequence: str) -> np.ndarray:
        """
        Generate embedding for a single sequence
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            # Prepare sequence
            clean_seq = self._prepare_sequence(sequence)
            
            if len(clean_seq) == 0:
                logger.warning("Empty sequence after cleaning, returning zero embedding")
                return np.zeros(self.model.config.hidden_size, dtype=np.float32)
            
            # Tokenize
            inputs = self.tokenizer(
                clean_seq,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs, output_all_encoded_layers=False)
                
                # Handle different output formats
                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    # Custom BertModel returns (hidden_states, pooled_output)
                    sequence_output, pooled_output = outputs
                    if pooled_output is not None:
                        # Use pooled output if available
                        embedding = pooled_output.cpu().numpy().squeeze()
                    else:
                        # Fall back to mean pooling
                        embeddings = sequence_output
                        attention_mask = inputs['attention_mask']
                        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                        mean_embeddings = sum_embeddings / sum_mask
                        embedding = mean_embeddings.cpu().numpy().squeeze()
                else:
                    # Standard transformer output
                    if hasattr(outputs, 'last_hidden_state'):
                        embeddings = outputs.last_hidden_state
                    else:
                        embeddings = outputs[0]
                    
                    # Mean pooling over sequence length (excluding padding)
                    attention_mask = inputs['attention_mask']
                    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    mean_embeddings = sum_embeddings / sum_mask
                    embedding = mean_embeddings.cpu().numpy().squeeze()
                
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Error generating embedding for sequence: {e}")
            return np.zeros(self.model.config.hidden_size, dtype=np.float32)
    
    def generate_embeddings(self, sequences_to_process: List[Tuple[str, str]], 
                          batch_size: int = 8) -> pd.DataFrame:
        """
        Generate embeddings for a list of sequences
        
        Args:
            sequences_to_process: List of (header, sequence) tuples
            batch_size: Number of sequences to process in each batch
            
        Returns:
            DataFrame with headers as index and embedding features as columns
        """
        logger.info(f"Generating embeddings for {len(sequences_to_process)} sequences on {self.device}")
        
        embeddings_dict = OrderedDict()
        num_batches = (len(sequences_to_process) + batch_size - 1) // batch_size
        
        # Process sequences in batches with progress bar
        with tqdm(total=len(sequences_to_process), desc="Generating embeddings", unit="seq") as pbar:
            for i in range(0, len(sequences_to_process), batch_size):
                batch = sequences_to_process[i:i + batch_size]
                
                # Prepare batch sequences
                batch_sequences = [self._prepare_sequence(seq) for _, seq in batch]
                batch_headers = [header for header, _ in batch]
                
                # Filter out empty sequences
                valid_indices = [j for j, seq in enumerate(batch_sequences) if len(seq) > 0]
                
                if not valid_indices:
                    # All sequences in batch are empty
                    for header in batch_headers:
                        embeddings_dict[header] = np.zeros(self.model.config.hidden_size, dtype=np.float32)
                    pbar.update(len(batch))
                    continue
                
                valid_sequences = [batch_sequences[j] for j in valid_indices]
                valid_headers = [batch_headers[j] for j in valid_indices]
                
                try:
                    # Tokenize batch
                    inputs = self.tokenizer(
                        valid_sequences,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_length
                    ).to(self.device)
                    
                    # Generate embeddings
                    with torch.no_grad():
                        outputs = self.model(**inputs, output_all_encoded_layers=False)
                        
                        # Handle different output formats
                        if isinstance(outputs, tuple) and len(outputs) >= 2:
                            # Custom BertModel returns (hidden_states, pooled_output)
                            sequence_output, pooled_output = outputs
                            if pooled_output is not None:
                                # Use pooled output if available
                                batch_embeddings = pooled_output.cpu().numpy()
                            else:
                                # Fall back to mean pooling
                                embeddings = sequence_output
                                attention_mask = inputs['attention_mask']
                                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                                sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                                mean_embeddings = sum_embeddings / sum_mask
                                batch_embeddings = mean_embeddings.cpu().numpy()
                        else:
                            # Standard transformer output
                            if hasattr(outputs, 'last_hidden_state'):
                                embeddings = outputs.last_hidden_state
                            else:
                                embeddings = outputs[0]
                            
                            # Mean pooling
                            attention_mask = inputs['attention_mask']
                            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                            mean_embeddings = sum_embeddings / sum_mask
                            batch_embeddings = mean_embeddings.cpu().numpy()
                    
                    # Store embeddings
                    for j, header in enumerate(valid_headers):
                        embeddings_dict[header] = batch_embeddings[j].astype(np.float32)
                    
                    # Handle empty sequences in this batch
                    for j, header in enumerate(batch_headers):
                        if j not in valid_indices:
                            embeddings_dict[header] = np.zeros(self.model.config.hidden_size, dtype=np.float32)
                            
                except Exception as e:
                    logger.warning(f"Error processing batch {i//batch_size + 1}: {e}")
                    # Fallback to individual processing
                    for header, seq in batch:
                        embeddings_dict[header] = self._get_sequence_embedding(seq)
                
                # Update progress bar
                pbar.update(len(batch))
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(embeddings_dict, orient="index", dtype=np.float32)
        df.columns = [f"emb_{i}" for i in range(df.shape[1])]
        
        logger.info(f"Generated embeddings: {df.shape[0]} sequences, {df.shape[1]} dimensions")
        
        return df


def calculate_dna_embeddings(
    sequences_to_process: List[Tuple[str, str]],
    model_path: str,
    device: str = "auto",
    batch_size: int = 8,
    max_length: int = 512
) -> pd.DataFrame:
    """
    Convenience function to generate DNA embeddings
    
    Args:
        sequences_to_process: List of (header, sequence) tuples
        model_path: Path to DNABERT-2 model directory
        device: Device to use ('cpu', 'cuda', 'mps', or 'auto' for automatic selection)
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        
    Returns:
        DataFrame with embeddings
    """
    generator = DNAEmbeddingGenerator(
        model_path=model_path,
        device=device,
        max_length=max_length
    )
    
    return generator.generate_embeddings(sequences_to_process, batch_size=batch_size)

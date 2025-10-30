"""
Standalone Character Tokenizer for DNA sequences.
No transformers dependency required - only uses Python standard library.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union


class StandaloneCharacterTokenizer:
    """
    Lightweight character tokenizer for DNA sequences.
    Compatible with the transformers-based CharacterTokenizer but without dependencies.
    """

    # Special token IDs (must match transformers version)
    CLS_TOKEN_ID = 0
    SEP_TOKEN_ID = 1
    BOS_TOKEN_ID = 2
    MASK_TOKEN_ID = 3
    PAD_TOKEN_ID = 4
    RESERVED_TOKEN_ID = 5
    UNK_TOKEN_ID = 6

    def __init__(
        self,
        characters: Sequence[str],
        model_max_length: int,
        padding_side: str = 'left'
    ):
        """
        Initialize character tokenizer.

        Args:
            characters: List of characters (e.g., ['A', 'C', 'G', 'T', 'N'])
            model_max_length: Maximum sequence length
            padding_side: 'left' or 'right' padding
        """
        self.characters = list(characters)
        self.model_max_length = model_max_length
        self.padding_side = padding_side

        # Build vocabulary
        self._vocab_str_to_int = {
            "[CLS]": self.CLS_TOKEN_ID,
            "[SEP]": self.SEP_TOKEN_ID,
            "[BOS]": self.BOS_TOKEN_ID,
            "[MASK]": self.MASK_TOKEN_ID,
            "[PAD]": self.PAD_TOKEN_ID,
            "[RESERVED]": self.RESERVED_TOKEN_ID,
            "[UNK]": self.UNK_TOKEN_ID,
        }

        # Add regular characters starting at ID 7
        for i, ch in enumerate(characters):
            self._vocab_str_to_int[ch] = i + 7

        # Reverse mapping
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self._vocab_str_to_int)

    @property
    def pad_token_id(self) -> int:
        """Return padding token ID."""
        return self.PAD_TOKEN_ID

    @property
    def cls_token_id(self) -> int:
        """Return CLS token ID."""
        return self.CLS_TOKEN_ID

    @property
    def sep_token_id(self) -> int:
        """Return SEP token ID."""
        return self.SEP_TOKEN_ID

    @property
    def unk_token_id(self) -> int:
        """Return UNK token ID."""
        return self.UNK_TOKEN_ID

    def get_vocab(self) -> Dict[str, int]:
        """Return vocabulary dictionary."""
        return self._vocab_str_to_int.copy()

    def _tokenize(self, text: str) -> List[str]:
        """Split text into character tokens."""
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token (character) to its ID."""
        return self._vocab_str_to_int.get(token, self.UNK_TOKEN_ID)

    def _convert_id_to_token(self, token_id: int) -> str:
        """Convert an ID to its token (character)."""
        return self._vocab_int_to_str.get(token_id, "[UNK]")

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Convert token(s) to ID(s).

        Handles both single string and list of strings for compatibility with transformers.
        This is needed for predictor.py line 129 which calls convert_tokens_to_ids with a single char.

        Args:
            tokens: Single token (str) or list of tokens (List[str])

        Returns:
            Single ID (int) if input was str, or list of IDs (List[int]) if input was list
        """
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert list of IDs to list of tokens."""
        return [self._convert_id_to_token(id) for id in ids]

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], Dict[str, List[int]]]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add [CLS] and [SEP] tokens
            padding: Whether to pad ('max_length' or True for model_max_length)
            truncation: Whether to truncate to max_length
            max_length: Maximum length (uses model_max_length if None)
            return_tensors: 'pt' for PyTorch tensors, None for lists

        Returns:
            Dict with 'input_ids' and 'attention_mask' or just list of IDs
        """
        # Tokenize
        tokens = self._tokenize(text)

        # Convert to IDs
        ids = self.convert_tokens_to_ids(tokens)

        # Add special tokens
        if add_special_tokens:
            ids = [self.CLS_TOKEN_ID] + ids + [self.SEP_TOKEN_ID]

        # Truncate (preserve CLS at start and SEP at end if special tokens were added)
        target_length = max_length if max_length is not None else self.model_max_length
        if truncation and len(ids) > target_length:
            if add_special_tokens:
                # Keep CLS at start and SEP at end
                ids = ids[:target_length - 1] + [self.SEP_TOKEN_ID]
            else:
                ids = ids[:target_length]

        # Padding
        attention_mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)  # All 0s for single sequences
        if padding:
            if len(ids) < target_length:
                pad_length = target_length - len(ids)
                if self.padding_side == 'left':
                    ids = [self.PAD_TOKEN_ID] * pad_length + ids
                    attention_mask = [0] * pad_length + attention_mask
                    token_type_ids = [0] * pad_length + token_type_ids
                else:
                    ids = ids + [self.PAD_TOKEN_ID] * pad_length
                    attention_mask = attention_mask + [0] * pad_length
                    token_type_ids = token_type_ids + [0] * pad_length

        # Return format
        if return_tensors == 'pt':
            import torch
            return {
                'input_ids': torch.tensor([ids]),
                'attention_mask': torch.tensor([attention_mask]),
                'token_type_ids': torch.tensor([token_type_ids])
            }
        else:
            # Always return dict format (matching transformers behavior)
            return {
                'input_ids': ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to remove special tokens

        Returns:
            Decoded text string
        """
        tokens = self.convert_ids_to_tokens(token_ids)

        if skip_special_tokens:
            special_tokens = {"[CLS]", "[SEP]", "[BOS]", "[MASK]", "[PAD]", "[RESERVED]", "[UNK]"}
            tokens = [t for t in tokens if t not in special_tokens]

        return "".join(tokens)

    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ):
        """
        Tokenize text (supports single string or list of strings).
        """
        if isinstance(text, str):
            return self.encode(
                text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
            )
        else:
            # Batch encoding
            results = [
                self.encode(
                    t,
                    add_special_tokens=add_special_tokens,
                    padding=False,  # We'll pad the batch after
                    truncation=truncation,
                    max_length=max_length,
                    return_tensors=None,
                )
                for t in text
            ]

            if padding:
                # Pad batch to same length
                target_length = max_length if max_length is not None else self.model_max_length

                # When padding="max_length", always pad to target_length
                # When padding=True, pad to longest sequence in batch
                if padding == "max_length":
                    max_len = target_length
                else:
                    max_len = max(len(r) if isinstance(r, list) else len(r['input_ids']) for r in results)
                    max_len = min(max_len, target_length) if truncation else max_len

                input_ids = []
                attention_masks = []
                token_type_ids_list = []

                for r in results:
                    ids = r if isinstance(r, list) else r['input_ids']
                    if len(ids) < max_len:
                        pad_length = max_len - len(ids)
                        if self.padding_side == 'left':
                            ids = [self.PAD_TOKEN_ID] * pad_length + ids
                            mask = [0] * pad_length + [1] * (max_len - pad_length)
                            token_types = [0] * max_len
                        else:
                            ids = ids + [self.PAD_TOKEN_ID] * pad_length
                            mask = [1] * (max_len - pad_length) + [0] * pad_length
                            token_types = [0] * max_len
                    else:
                        mask = [1] * len(ids)
                        token_types = [0] * len(ids)

                    input_ids.append(ids)
                    attention_masks.append(mask)
                    token_type_ids_list.append(token_types)

                if return_tensors == 'pt':
                    import torch
                    return {
                        'input_ids': torch.tensor(input_ids),
                        'attention_mask': torch.tensor(attention_masks),
                        'token_type_ids': torch.tensor(token_type_ids_list)
                    }
                else:
                    return {
                        'input_ids': input_ids,
                        'attention_mask': attention_masks,
                        'token_type_ids': token_type_ids_list
                    }
            else:
                return results

    def get_config(self) -> Dict:
        """Get tokenizer configuration for saving."""
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
            "padding_side": self.padding_side,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "StandaloneCharacterTokenizer":
        """Create tokenizer from configuration dictionary."""
        characters = [chr(i) for i in config["char_ords"]]
        model_max_length = config["model_max_length"]
        padding_side = config.get("padding_side", "left")
        return cls(
            characters=characters,
            model_max_length=model_max_length,
            padding_side=padding_side,
        )

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """Save tokenizer configuration to directory."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        cfg_file = save_dir / "tokenizer_config.json"
        cfg = self.get_config()

        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs) -> "StandaloneCharacterTokenizer":
        """Load tokenizer from saved configuration."""
        cfg_file = Path(save_directory) / "tokenizer_config.json"

        with open(cfg_file) as f:
            cfg = json.load(f)

        return cls.from_config(cfg)

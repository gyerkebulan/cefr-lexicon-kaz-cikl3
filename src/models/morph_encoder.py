"""
Morphology encoder: Char-CNN and feature MLP for CEFR classification.
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("cefr")


# Character vocabulary for Kazakh Cyrillic
KAZAKH_CHARS = (
    'абвгдежзийклмнопрстуфхцчшщъыьэюя'  # Russian base
    'әғқңөұүһі'  # Kazakh-specific
)
PAD_CHAR = '<PAD>'
UNK_CHAR = '<UNK>'


class CharVocab:
    """Character vocabulary for char-level encoding."""
    
    def __init__(self, chars: str = KAZAKH_CHARS):
        self.chars = chars.lower()
        self.char_to_idx = {PAD_CHAR: 0, UNK_CHAR: 1}
        for i, c in enumerate(self.chars):
            self.char_to_idx[c] = i + 2
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
    
    def encode(self, word: str, max_len: int = 30) -> List[int]:
        """Encode word to character indices."""
        word = word.lower()[:max_len]
        indices = [self.char_to_idx.get(c, 1) for c in word]
        # Pad
        indices = indices + [0] * (max_len - len(indices))
        return indices
    
    def encode_batch(self, words: List[str], max_len: int = 30) -> torch.Tensor:
        """Encode batch of words."""
        return torch.tensor([self.encode(w, max_len) for w in words], dtype=torch.long)


class CharCNNEncoder(nn.Module):
    """
    Character-level CNN encoder for words/lemmas.
    """
    
    def __init__(
        self,
        vocab_size: int = 50,
        embed_dim: int = 64,
        num_filters: int = 128,
        kernel_sizes: List[int] = [2, 3, 4],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        
        # Character embedding
        self.char_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Convolutional layers (one per kernel size)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k//2)
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension
        self.output_dim = num_filters * len(kernel_sizes)
        
        logger.info(f"CharCNN: vocab={vocab_size}, embed={embed_dim}, "
                    f"filters={num_filters}, kernels={kernel_sizes}, output={self.output_dim}")
    
    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            char_ids: [batch, max_char_len]
            
        Returns:
            char_emb: [batch, output_dim]
        """
        # Embed characters
        x = self.char_embedding(char_ids)  # [batch, seq, embed]
        x = x.transpose(1, 2)  # [batch, embed, seq] for conv1d
        
        # Apply each conv and max-pool
        conv_outputs = []
        for conv in self.convs:
            h = F.relu(conv(x))  # [batch, filters, seq]
            h = F.max_pool1d(h, h.size(2)).squeeze(2)  # [batch, filters]
            conv_outputs.append(h)
        
        # Concatenate all conv outputs
        combined = torch.cat(conv_outputs, dim=1)  # [batch, num_filters * len(kernels)]
        return self.dropout(combined)


class MorphFeatureEncoder(nn.Module):
    """
    MLP encoder for morphological feature vector.
    """
    
    def __init__(
        self,
        input_dim: int = 22,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.1,
        num_layers: int = 2,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        in_dim = input_dim
        for _ in range(max(1, num_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        logger.info(
            "MorphFeatureEncoder: input=%d, hidden=%d x%d, output=%d, layer_norm=%s",
            input_dim,
            hidden_dim,
            max(1, num_layers),
            output_dim,
            use_layer_norm,
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: [batch, input_dim]
            
        Returns:
            morph_emb: [batch, output_dim]
        """
        return self.mlp(features)


class CombinedMorphEncoder(nn.Module):
    """
    Combined morphology encoder: Char-CNN + Feature MLP.
    """
    
    def __init__(
        self,
        char_vocab_size: int = 50,
        char_embed_dim: int = 64,
        char_num_filters: int = 128,
        char_kernel_sizes: List[int] = [2, 3, 4],
        morph_input_dim: int = 22,
        morph_hidden_dim: int = 64,
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.char_encoder = CharCNNEncoder(
            vocab_size=char_vocab_size,
            embed_dim=char_embed_dim,
            num_filters=char_num_filters,
            kernel_sizes=char_kernel_sizes,
            dropout=dropout
        )
        
        self.morph_encoder = MorphFeatureEncoder(
            input_dim=morph_input_dim,
            hidden_dim=morph_hidden_dim,
            output_dim=morph_hidden_dim,
            dropout=dropout
        )
        
        # Combine char + morph
        combined_dim = self.char_encoder.output_dim + morph_hidden_dim
        self.projection = nn.Linear(combined_dim, output_dim)
        
        self.output_dim = output_dim
    
    def forward(
        self,
        char_ids: torch.Tensor,
        morph_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            char_ids: [batch, max_char_len]
            morph_features: [batch, morph_input_dim]
            
        Returns:
            morph_emb: [batch, output_dim]
        """
        char_emb = self.char_encoder(char_ids)
        morph_emb = self.morph_encoder(morph_features)
        
        combined = torch.cat([char_emb, morph_emb], dim=1)
        return self.projection(combined)

"""
Morphology-only CEFR classifiers for ablation study.

MorphOnlyCEFR: CharCNN + MorphMLP (no transformer context)
CharCNNOnlyCEFR: CharCNN only (no MorphMLP, no transformer)
"""

import logging
from typing import List

import torch
import torch.nn as nn

from .morph_encoder import CharCNNEncoder, CombinedMorphEncoder

logger = logging.getLogger("cefr")


class MorphOnlyCEFR(nn.Module):
    """
    Morphology-only CEFR classifier (CharCNN + MorphMLP).

    Ablation condition: tests the combined morphological branch
    without any transformer context encoder.
    """

    def __init__(
        self,
        char_vocab_size: int = 50,
        char_embed_dim: int = 64,
        char_num_filters: int = 128,
        char_kernel_sizes: List[int] = [2, 3, 4],
        morph_input_dim: int = 22,
        morph_hidden_dim: int = 64,
        morph_output_dim: int = 128,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = CombinedMorphEncoder(
            char_vocab_size=char_vocab_size,
            char_embed_dim=char_embed_dim,
            char_num_filters=char_num_filters,
            char_kernel_sizes=char_kernel_sizes,
            morph_input_dim=morph_input_dim,
            morph_hidden_dim=morph_hidden_dim,
            output_dim=morph_output_dim,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(morph_output_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(morph_output_dim, num_classes)

        logger.info(
            "MorphOnlyCEFR: morph_output=%d, classes=%d",
            morph_output_dim,
            num_classes,
        )

    def forward(
        self,
        char_ids: torch.Tensor,
        morph_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            char_ids: [batch, max_char_len]
            morph_features: [batch, morph_input_dim]

        Returns:
            logits: [batch, num_classes]
        """
        emb = self.encoder(char_ids, morph_features)
        emb = self.norm(emb)
        emb = self.dropout(emb)
        return self.classifier(emb)


class CharCNNOnlyCEFR(nn.Module):
    """
    CharCNN-only CEFR classifier.

    Ablation condition: tests character-level patterns alone,
    without engineered morphological features or transformer context.
    """

    def __init__(
        self,
        char_vocab_size: int = 50,
        char_embed_dim: int = 64,
        char_num_filters: int = 128,
        char_kernel_sizes: List[int] = [2, 3, 4],
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = CharCNNEncoder(
            vocab_size=char_vocab_size,
            embed_dim=char_embed_dim,
            num_filters=char_num_filters,
            kernel_sizes=char_kernel_sizes,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(self.encoder.output_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.output_dim, num_classes)

        logger.info(
            "CharCNNOnlyCEFR: cnn_output=%d, classes=%d",
            self.encoder.output_dim,
            num_classes,
        )

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_ids: [batch, max_char_len]

        Returns:
            logits: [batch, num_classes]
        """
        emb = self.encoder(char_ids)
        emb = self.norm(emb)
        emb = self.dropout(emb)
        return self.classifier(emb)

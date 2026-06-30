"""
Dual-encoder model: combines context (mBERT) and morphology (Char-CNN + features).
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .context_encoder import ContextEncoder
from .morph_encoder import CombinedMorphEncoder, CharVocab
from .fusion import get_fusion_module
from .coral_loss import CoralOrdinalClassifier

logger = logging.getLogger("cefr")


class DualEncoderCEFR(nn.Module):
    """
    Dual-encoder CEFR classifier.
    
    Combines:
    1. Context encoder (mBERT) for sentence context
    2. Morph encoder (Char-CNN + feature MLP) for word-level morphology
    3. Fusion module (concat or gated)
    """
    
    def __init__(
        self,
        # Context encoder params
        transformer_model: str = "bert-base-multilingual-cased",
        freeze_transformer: bool = False,
        freeze_except_top_n: int = 0,
        pooling: str = "mean",
        # Morph encoder params
        char_vocab_size: int = 50,
        char_embed_dim: int = 64,
        char_num_filters: int = 128,
        char_kernel_sizes: List[int] = [2, 3, 4],
        morph_input_dim: int = 22,
        morph_hidden_dim: int = 64,
        morph_output_dim: int = 128,
        # Fusion params
        fusion_type: str = "gated",
        fusion_output_dim: int = 256,
        # Classification
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Context encoder
        self.context_encoder = ContextEncoder(
            model_name=transformer_model,
            pooling=pooling,
            freeze_transformer=freeze_transformer,
            freeze_except_top_n=freeze_except_top_n
        )
        
        # Morph encoder
        self.morph_encoder = CombinedMorphEncoder(
            char_vocab_size=char_vocab_size,
            char_embed_dim=char_embed_dim,
            char_num_filters=char_num_filters,
            char_kernel_sizes=char_kernel_sizes,
            morph_input_dim=morph_input_dim,
            morph_hidden_dim=morph_hidden_dim,
            output_dim=morph_output_dim,
            dropout=dropout
        )
        
        # Fusion
        self.fusion = get_fusion_module(
            fusion_type=fusion_type,
            context_dim=self.context_encoder.output_dim,
            morph_dim=self.morph_encoder.output_dim,
            output_dim=fusion_output_dim
        )
        
        # Classifier
        self.fusion_norm = nn.LayerNorm(self.fusion.output_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.fusion.output_dim, num_classes)

        # Ordinal regression head (CORAL)
        self.ordinal_head = CoralOrdinalClassifier(self.fusion.output_dim, num_classes)

        logger.info(
            f"DualEncoder: context_dim={self.context_encoder.output_dim}, "
            f"morph_dim={self.morph_encoder.output_dim}, "
            f"fusion={fusion_type}, output={self.fusion.output_dim}"
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_mask: torch.Tensor,
        char_ids: torch.Tensor,
        morph_features: torch.Tensor,
        return_ordinal: bool = False
    ):
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len] - tokenized sentence
            attention_mask: [batch, seq_len]
            target_mask: [batch, seq_len] - target word positions
            char_ids: [batch, max_char_len] - character indices
            morph_features: [batch, morph_input_dim] - morphological features
            return_ordinal: if True, also return ordinal logits

        Returns:
            logits: [batch, num_classes]
            ordinal_logits: [batch, num_classes-1] (only if return_ordinal=True)
        """
        # Encode context
        context_emb = self.context_encoder(input_ids, attention_mask, target_mask)

        # Encode morphology
        morph_emb = self.morph_encoder(char_ids, morph_features)

        # Fuse
        fused = self.fusion(context_emb, morph_emb)

        # Classify
        fused = self.fusion_norm(fused)
        fused = self.dropout(fused)
        logits = self.classifier(fused)

        if return_ordinal:
            ordinal_logits = self.ordinal_head(fused)
            return logits, ordinal_logits

        return logits


class DualEncoderDataset(torch.utils.data.Dataset):
    """
    Dataset for dual encoder training.
    """
    
    def __init__(
        self,
        sentences: List[str],
        targets: List[str],
        lemmas: List[str],
        features: torch.Tensor,
        labels: List[str],
        tokenizer,
        char_vocab: CharVocab,
        max_char_len: int = 30,
        label_to_idx: Dict[str, int] = None
    ):
        self.sentences = sentences
        self.targets = targets
        self.lemmas = lemmas
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.char_vocab = char_vocab
        self.max_char_len = max_char_len
        
        # Label mapping
        self.label_to_idx = label_to_idx or {
            "A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4
        }
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        target = self.targets[idx]
        lemma = self.lemmas[idx]
        
        # Tokenize sentence
        encoded = self.tokenizer([sentence], [target])
        
        # Character encoding
        char_ids = self.char_vocab.encode(lemma, self.max_char_len)
        
        # Features
        morph_features = self.features[idx]
        
        # Label
        label = self.label_to_idx.get(self.labels[idx], 2)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'target_mask': encoded['target_mask'].squeeze(0),
            'char_ids': torch.tensor(char_ids, dtype=torch.long),
            'morph_features': morph_features,
            'label': torch.tensor(label, dtype=torch.long)
        }


def collate_dual_encoder(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length sequences."""
    # Pad sequences to max length in batch
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    input_ids = []
    attention_mask = []
    target_mask = []
    char_ids = []
    morph_features = []
    labels = []
    
    for item in batch:
        seq_len = item['input_ids'].size(0)
        pad_len = max_len - seq_len
        
        if pad_len > 0:
            input_ids.append(
                torch.cat([item['input_ids'], torch.zeros(pad_len, dtype=torch.long)])
            )
            attention_mask.append(
                torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)])
            )
            target_mask.append(
                torch.cat([item['target_mask'], torch.zeros(pad_len, dtype=torch.long)])
            )
        else:
            input_ids.append(item['input_ids'])
            attention_mask.append(item['attention_mask'])
            target_mask.append(item['target_mask'])
        
        char_ids.append(item['char_ids'])
        morph_features.append(item['morph_features'])
        labels.append(item['label'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'target_mask': torch.stack(target_mask),
        'char_ids': torch.stack(char_ids),
        'morph_features': torch.stack(morph_features),
        'labels': torch.stack(labels)
    }

"""
Context encoder using mBERT/XLM-R for CEFR classification.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger("cefr")
OFFLINE_FALLBACK_MODEL = "bert-base-multilingual-cased"


def _local_files_only() -> bool:
    return os.environ.get("CEFR_ALLOW_HF_DOWNLOAD", "").strip().lower() not in {"1", "true", "yes"}


def _load_model_with_offline_fallback(model_name: str):
    local_only = _local_files_only()
    try:
        return AutoModel.from_pretrained(model_name, local_files_only=local_only), model_name
    except Exception as exc:
        if model_name == OFFLINE_FALLBACK_MODEL:
            raise
        logger.warning(
            f"Could not load {model_name} with local_files_only={local_only} ({exc}). "
            f"Falling back to {OFFLINE_FALLBACK_MODEL}."
        )
        return (
            AutoModel.from_pretrained(OFFLINE_FALLBACK_MODEL, local_files_only=local_only),
            OFFLINE_FALLBACK_MODEL,
        )


def _load_tokenizer_with_offline_fallback(model_name: str):
    local_only = _local_files_only()
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=local_only), model_name
    except Exception as exc:
        if model_name == OFFLINE_FALLBACK_MODEL:
            raise
        logger.warning(
            f"Could not load tokenizer for {model_name} with local_files_only={local_only} ({exc}). "
            f"Falling back to {OFFLINE_FALLBACK_MODEL}."
        )
        return (
            AutoTokenizer.from_pretrained(OFFLINE_FALLBACK_MODEL, local_files_only=local_only),
            OFFLINE_FALLBACK_MODEL,
        )


class ContextEncoder(nn.Module):
    """
    Transformer-based context encoder.
    
    Encodes a sentence and extracts representation for the target word.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        pooling: str = "mean",  # "cls", "mean", "first", "attention"
        freeze_transformer: bool = False,
        freeze_except_top_n: int = 0,
        target_token: str = "[TGT]"
    ):
        super().__init__()

        self.model_name = model_name
        self.pooling = pooling
        self.target_token = target_token

        # Load transformer
        self.transformer, self.model_name = _load_model_with_offline_fallback(model_name)
        self.hidden_size = self.transformer.config.hidden_size

        # Attention pooling layer
        if self.pooling == "attention":
            self.attention_weights = nn.Linear(self.hidden_size, 1)

        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
            logger.info("Transformer weights frozen")
        elif freeze_except_top_n > 0:
            self._freeze_except_top_n(freeze_except_top_n)

        logger.info(f"Loaded {self.model_name}, hidden_size={self.hidden_size}")

    def _freeze_except_top_n(self, n: int):
        """Freeze all transformer layers except the top N."""
        # Freeze embeddings
        for param in self.transformer.embeddings.parameters():
            param.requires_grad = False

        # Get encoder layers (works for both BERT and XLM-R)
        encoder = self.transformer.encoder
        num_layers = len(encoder.layer)
        freeze_up_to = num_layers - n

        for i, layer in enumerate(encoder.layer):
            if i < freeze_up_to:
                for param in layer.parameters():
                    param.requires_grad = False

        trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.transformer.parameters())
        logger.info(
            f"Froze transformer except top {n} layers: "
            f"{trainable:,}/{total:,} params trainable ({100*trainable/total:.1f}%)"
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            target_mask: [batch, seq_len] - 1 for target tokens, 0 otherwise
            
        Returns:
            context_emb: [batch, hidden_size]
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = outputs.last_hidden_state  # [batch, seq, hidden]
        
        if self.pooling == "cls":
            # Use [CLS] token
            return hidden_states[:, 0, :]
        
        elif self.pooling == "mean" and target_mask is not None:
            # Mean pool over target tokens
            target_mask = target_mask.unsqueeze(-1).float()  # [batch, seq, 1]
            masked = hidden_states * target_mask
            sum_emb = masked.sum(dim=1)  # [batch, hidden]
            count = target_mask.sum(dim=1).clamp(min=1)  # [batch, 1]
            return sum_emb / count
            
        elif self.pooling == "attention" and target_mask is not None:
            # Attention pooling over target tokens
            # Compute raw attention scores
            scores = self.attention_weights(hidden_states)  # [batch, seq, 1]
            
            # Mask out non-target tokens effectively (-inf before softmax)
            scores = scores.masked_fill(target_mask.unsqueeze(-1) == 0, float('-inf'))
            
            # Apply softmax
            attn_weights = torch.softmax(scores, dim=1)  # [batch, seq, 1]
            
            # Apply weights to hidden states and sum
            weighted_emb = hidden_states * attn_weights
            return weighted_emb.sum(dim=1)  # [batch, hidden]
        
        else:
            # Fallback: mean pool over all tokens
            mask = attention_mask.unsqueeze(-1).float()
            masked = hidden_states * mask
            return masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    
    @property
    def output_dim(self) -> int:
        return self.hidden_size


class ContextTokenizer:
    """
    Tokenizer wrapper that handles target word marking.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        max_length: int = 128,
        target_token: str = "[TGT]"
    ):
        self.tokenizer, self.model_name = _load_tokenizer_with_offline_fallback(model_name)
        self.max_length = max_length
        self.target_token = target_token
        
        # Add target token if not in vocab
        # Use get_vocab() for compatibility with both BERT and XLM-R tokenizers
        vocab = self.tokenizer.get_vocab()
        if target_token not in vocab:
            self.tokenizer.add_special_tokens({'additional_special_tokens': [target_token]})
            logger.info(f"Added {target_token} to tokenizer")
    
    def __call__(
        self,
        sentences: List[str],
        targets: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize sentences with target markers.
        
        Args:
            sentences: List of sentences
            targets: List of target words (to be marked)
            
        Returns:
            Dict with input_ids, attention_mask, target_mask
        """
        # Mark targets in sentences
        marked_sentences = []
        for sent, target in zip(sentences, targets):
            # Simple replacement (case-insensitive)
            marked = sent.replace(target, f"{self.target_token} {target} {self.target_token}")
            marked_sentences.append(marked)
        
        # Tokenize
        encoded = self.tokenizer(
            marked_sentences,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Create target mask
        target_token_id = self.tokenizer.convert_tokens_to_ids(self.target_token)
        
        # Find positions between [TGT] markers
        input_ids = encoded['input_ids']
        batch_size, seq_len = input_ids.shape
        target_mask = torch.zeros_like(input_ids)
        
        for i in range(batch_size):
            in_target = False
            for j in range(seq_len):
                if input_ids[i, j] == target_token_id:
                    in_target = not in_target
                elif in_target:
                    target_mask[i, j] = 1
        
        return {
            'input_ids': input_ids,
            'attention_mask': encoded['attention_mask'],
            'target_mask': target_mask
        }
    
    def resize_embeddings(self, model: nn.Module) -> None:
        """Resize model embeddings if tokens were added."""
        if hasattr(model, 'transformer'):
            model.transformer.resize_token_embeddings(len(self.tokenizer))


class ContextOnlyCEFR(nn.Module):
    """
    Context-only CEFR classifier (mBERT baseline).
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        num_classes: int = 5,
        dropout: float = 0.1,
        freeze_transformer: bool = False
    ):
        super().__init__()
        
        self.encoder = ContextEncoder(
            model_name=model_name,
            pooling="mean",
            freeze_transformer=freeze_transformer
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.output_dim, num_classes)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Returns:
            logits: [batch, num_classes]
        """
        context_emb = self.encoder(input_ids, attention_mask, target_mask)
        context_emb = self.dropout(context_emb)
        return self.classifier(context_emb)

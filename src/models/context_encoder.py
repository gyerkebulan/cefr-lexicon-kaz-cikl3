"""
Context encoder using mBERT/XLM-R for CEFR classification.
"""

import logging
import os
import unicodedata
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger("cefr")
OFFLINE_FALLBACK_MODEL = "bert-base-multilingual-cased"
TARGET_SPAN_FALLBACK_STATUSES = {"missing_fallback", "truncated_fallback"}


def count_target_span_failures(statuses: List[str]) -> int:
    return sum(status in TARGET_SPAN_FALLBACK_STATUSES for status in statuses)


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
        if target_mask is not None:
            invalid_rows = target_mask.sum(dim=1) <= 0
            if invalid_rows.any():
                indices = invalid_rows.nonzero(as_tuple=False).flatten().tolist()
                raise ValueError(f"Empty target mask for batch rows {indices}")

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
            pooled = sum_emb / count
            
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
            pooled = weighted_emb.sum(dim=1)  # [batch, hidden]
        
        else:
            # Fallback: mean pool over all tokens
            mask = attention_mask.unsqueeze(-1).float()
            masked = hidden_states * mask
            pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        if not torch.isfinite(pooled).all():
            raise FloatingPointError("Context pooling produced NaN or infinity")
        return pooled
    
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
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("Offset-aware target detection requires a fast tokenizer")
        self.max_length = max_length
        self.target_token = target_token
        # ponytail: in-process counter; return statuses through batches if DataLoader workers are added.
        self.target_span_failure_count = 0

    def reset_stats(self) -> None:
        self.target_span_failure_count = 0
    
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
        if len(sentences) != len(targets):
            raise ValueError("sentences and targets must have equal length")
        texts, spans_by_row, statuses = [], [], []
        for sentence, target in zip(sentences, targets):
            text = unicodedata.normalize("NFC", str(sentence))
            normalized_target = unicodedata.normalize("NFC", str(target)).strip()
            if not normalized_target:
                raise ValueError("Target must not be empty")
            spans = self._find_spans(text, normalized_target)
            if spans:
                status = "repeated" if len(spans) > 1 else "found"
            else:
                text, spans, status = normalized_target, [(0, len(normalized_target))], "missing_fallback"
            texts.append(text)
            spans_by_row.append(spans)
            statuses.append(status)

        encoded = self.tokenizer(
            texts, max_length=self.max_length, padding="max_length", truncation=True,
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        target_mask = torch.zeros_like(encoded["input_ids"])
        for row, spans in enumerate(spans_by_row):
            for column, (start, end) in enumerate(offsets[row].tolist()):
                if end > start and any(start < span_end and end > span_start for span_start, span_end in spans):
                    target_mask[row, column] = 1
            if target_mask[row].sum() == 0:
                fallback = self.tokenizer(
                    str(targets[row]), max_length=self.max_length, padding="max_length",
                    truncation=True, return_offsets_mapping=True, return_tensors="pt",
                )
                fallback_offsets = fallback.pop("offset_mapping")[0]
                encoded["input_ids"][row] = fallback["input_ids"][0]
                encoded["attention_mask"][row] = fallback["attention_mask"][0]
                target_mask[row] = (
                    (fallback_offsets[:, 1] > fallback_offsets[:, 0])
                    & fallback["attention_mask"][0].bool()
                ).long()
                statuses[row] = "truncated_fallback"
            if target_mask[row].sum() == 0:
                raise ValueError(f"Target tokenization produced an empty mask for row {row}")

        self.target_span_failure_count += count_target_span_failures(statuses)
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'target_mask': target_mask,
            'target_status': statuses,
            'target_span_count': torch.tensor([len(spans) for spans in spans_by_row]),
        }

    @staticmethod
    def _find_spans(sentence: str, target: str) -> List[Tuple[int, int]]:
        folded, origins = [], []
        for index, character in enumerate(sentence):
            expansion = character.casefold()
            folded.extend(expansion)
            origins.extend([index] * len(expansion))
        needle = target.casefold()
        haystack = "".join(folded)
        spans, start = [], 0
        while True:
            found = haystack.find(needle, start)
            if found < 0:
                break
            end = found + len(needle)
            before = haystack[found - 1] if found else ""
            after = haystack[end] if end < len(haystack) else ""
            if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                spans.append((origins[found], origins[end - 1] + 1))
            start = found + max(len(needle), 1)
        return spans
    
    def resize_embeddings(self, model: nn.Module) -> None:
        """Kept for call-site compatibility; offset marking adds no tokens."""


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

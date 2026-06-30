#!/usr/bin/env python3
"""
Train context-only (XLM-R / mBERT) model for CEFR classification.

Ablation condition: tests transformer contextual embeddings alone,
without any morphological features or CharCNN.

Usage:
    python scripts/train_context.py
    python scripts/train_context.py --transformer xlm-roberta-base
    python scripts/train_context.py --transformer bert-base-multilingual-cased
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.utils import (
    load_config, load_paths, setup_logging, set_seed, get_device,
    CEFR_LABELS, CEFR_TO_INT,
)
from src.data_prep import load_splits, print_label_distribution
from src.metrics import (
    compute_metrics, save_metrics, save_predictions,
    update_run_index, print_metrics_summary,
)
from src.models.context_encoder import ContextOnlyCEFR, ContextTokenizer

# Reuse helpers from train_dual
from scripts.train_dual import (
    build_lemma_context_map,
    attach_context_columns,
    compute_class_weights,
    focal_loss,
)


# ── Dataset ──────────────────────────────────────────────────────────────────

class ContextDataset(Dataset):
    """Dataset for context-only encoder."""

    def __init__(self, sentences, targets, labels, tokenizer):
        self.sentences = sentences
        self.targets = targets
        self.labels = [CEFR_TO_INT[l] for l in labels]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoded = self.tokenizer([self.sentences[idx]], [self.targets[idx]])
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "target_mask": encoded["target_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids, attention_mask, target_mask, labels = [], [], [], []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len
        if pad_len > 0:
            input_ids.append(torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
            attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
            target_mask.append(torch.cat([item["target_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        else:
            input_ids.append(item["input_ids"])
            attention_mask.append(item["attention_mask"])
            target_mask.append(item["target_mask"])
        labels.append(item["label"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "target_mask": torch.stack(target_mask),
        "labels": torch.stack(labels),
    }


# ── Train / Eval ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device,
                class_weights=None, focal_gamma=2.0):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_mask = batch["target_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask, target_mask)
        loss = focal_loss(logits, labels, gamma=focal_gamma, class_weights=class_weights)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_mask = batch["target_mask"].to(device)
        labels = batch["labels"]

        logits = model(input_ids, attention_mask, target_mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().tolist()

        all_preds.extend([CEFR_LABELS[p] for p in preds])
        all_labels.extend([CEFR_LABELS[l] for l in labels.tolist()])
        all_probs.extend(probs)

    metrics = compute_metrics(all_labels, all_preds)
    return metrics, all_preds, np.array(all_probs)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train context-only ablation model")
    parser.add_argument("--transformer", type=str, default=None,
                        help="Transformer model name (default: from config, typically xlm-roberta-base)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--freeze-top-n", type=int, default=None,
                        help="Unfreeze only top N transformer layers")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--variant", default="context_only_xlmr", help="Model variant name")
    args = parser.parse_args()

    logger = setup_logging()
    paths = load_paths()
    config = load_config("training")
    seed = args.seed or int(config.get("seed", 42))
    set_seed(seed)
    device = get_device()

    transformer_model = args.transformer or config.get("transformer_model", "xlm-roberta-base")
    freeze_top_n = args.freeze_top_n if args.freeze_top_n is not None else config.get("freeze_transformer_except_top_n", 2)
    dropout = config.get("dropout", 0.3)
    batch_size = config.get("batch_size", 16)
    epochs = args.epochs or config.get("epochs", 20)
    patience = config.get("early_stopping_patience", 7)
    transformer_lr = config.get("learning_rate", 2e-5)
    head_lr = transformer_lr * 25  # 5e-4

    model_name = f"ablation_{args.variant}_s{seed}"
    logger.info(f"Training {model_name} (transformer={transformer_model}) on {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    processed_dir = args.data_dir or Path(paths["data"]["processed_dir"])
    train_df, dev_df, test_df = load_splits(processed_dir)

    if "sentence" not in train_df.columns:
        logger.warning("No 'sentence' column. Building context cache from external corpus.")
        corpus_dir = Path(paths["data"]["raw_dir"]) / "monolingual"
        context_cache = PROJECT_ROOT / "artifacts" / "context" / "lemma_sentences.json"
        train_df, dev_df, test_df = attach_context_columns(
            train_df, dev_df, test_df,
            corpus_dir=corpus_dir, cache_path=context_cache, logger=logger,
        )

    print_label_distribution(train_df, "Train")

    # ── Tokenizer & datasets ──────────────────────────────────────────────
    tokenizer = ContextTokenizer(
        model_name=transformer_model,
        max_length=config.get("max_seq_length", 128),
    )

    def make_dataset(df):
        return ContextDataset(
            df["sentence"].tolist() if "sentence" in df.columns else df["lemma"].tolist(),
            df["target"].tolist() if "target" in df.columns else df["lemma"].tolist(),
            df["cefr"].tolist(),
            tokenizer,
        )

    train_ds = make_dataset(train_df)
    dev_ds = make_dataset(dev_df)
    test_ds = make_dataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    class_weights = compute_class_weights(train_df, device)

    # ── Model ─────────────────────────────────────────────────────────────
    model = ContextOnlyCEFR(
        model_name=transformer_model,
        num_classes=5,
        dropout=dropout,
        freeze_transformer=False,
    )
    # Freeze all but top N layers (matching dual encoder protocol)
    model.encoder._freeze_except_top_n(freeze_top_n)
    tokenizer.resize_embeddings(model.encoder)
    model = model.to(device)

    # Discriminative LR: transformer lower, head higher
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": transformer_lr},
        {"params": model.classifier.parameters(), "lr": head_lr},
    ], weight_decay=config.get("weight_decay", 0.01))

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    logger.info(f"Discriminative LR: transformer={transformer_lr}, head={head_lr}")

    # ── Training loop ─────────────────────────────────────────────────────
    model_dir = Path(paths["results"]["models"]) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    best_dev_f1 = 0
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            class_weights=class_weights, focal_gamma=args.focal_gamma,
        )
        dev_metrics, _, _ = evaluate(model, dev_loader, device)

        logger.info(
            f"Epoch {epoch+1}/{epochs}: loss={train_loss:.4f}, "
            f"dev_acc={dev_metrics['accuracy']:.4f}, dev_f1={dev_metrics['macro_f1']:.4f}"
            f"{' *' if dev_metrics['macro_f1'] > best_dev_f1 else ''}"
        )

        if dev_metrics["macro_f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["macro_f1"]
            patience_counter = 0
            torch.save(model.state_dict(), model_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # ── Evaluate best ─────────────────────────────────────────────────────
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    dev_metrics, dev_preds, dev_probs = evaluate(model, dev_loader, device)
    test_metrics, test_preds, test_probs = evaluate(model, test_loader, device)

    print_metrics_summary(
        model_name=model_name,
        train_size=len(train_df), dev_size=len(dev_df), test_size=len(test_df),
        dev_metrics=dev_metrics, test_metrics=test_metrics,
        output_dir=Path(paths["results"]["dir"]),
    )

    # ── Save outputs ──────────────────────────────────────────────────────
    results_dir = Path(paths["results"]["dir"])
    save_metrics(
        test_metrics, f"{model_name}_test", results_dir / "metrics",
        extra_info={
            "seed": seed, "transformer_model": transformer_model,
            "focal_gamma": args.focal_gamma, "freeze_top_n": freeze_top_n,
        },
    )

    for split_name, df, preds, probs in [
        ("dev", dev_df, dev_preds, dev_probs),
        ("test", test_df, test_preds, test_probs),
    ]:
        df_out = df.copy()
        df_out["pred_cefr"] = preds
        df_out["gold_cefr"] = df_out["cefr"]
        for i, label in enumerate(CEFR_LABELS):
            df_out[f"prob_{label}"] = probs[:, i]
        save_predictions(df_out, model_name, results_dir / "predictions", split=split_name)

    update_run_index(
        model_name=model_name, config="training.yaml",
        dev_metrics=dev_metrics, test_metrics=test_metrics,
        index_path=Path(paths["results"]["run_index"]),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

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
import time
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
from src.data_prep import load_train_dev, print_label_distribution
from src.metrics import (
    compute_metrics, save_metrics, save_predictions,
    update_run_index, print_metrics_summary,
)
from src.models.context_encoder import ContextOnlyCEFR, ContextTokenizer
from src.research_protocol import (
    build_run_manifest, load_protocol, require_gpu_budget, verify_dataset_manifest,
    write_json,
)

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

    probabilities = np.array(all_probs)
    metrics = compute_metrics(all_labels, all_preds, probabilities=probabilities)
    return metrics, all_preds, probabilities


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
    parser.add_argument("--research-config", type=Path, default=None)
    parser.add_argument("--proposal-id", type=str, default=None)
    args = parser.parse_args()
    started_at = time.perf_counter()

    logger = setup_logging()
    paths = load_paths()
    config = load_config("training")
    seed = args.seed or int(config.get("seed", 42))
    set_seed(seed)
    device = get_device()
    protocol = protocol_hash = None
    if args.research_config:
        protocol, protocol_hash = load_protocol(args.research_config)
        require_gpu_budget(
            PROJECT_ROOT, protocol=protocol, protocol_hash=protocol_hash
        )
        if not args.proposal_id:
            raise ValueError("--proposal-id is required with --research-config")

    transformer_model = args.transformer or config.get("transformer_model", "xlm-roberta-base")
    freeze_top_n = args.freeze_top_n if args.freeze_top_n is not None else config.get("freeze_transformer_except_top_n", 2)
    dropout = config.get("dropout", 0.3)
    batch_size = config.get("batch_size", 16)
    epochs = args.epochs or config.get("epochs", 20)
    patience = config.get("early_stopping_patience", 7)
    if protocol:
        screen = protocol["evaluation"]["screening"]
        if epochs > screen["max_epochs"]:
            raise ValueError("Candidate epochs exceed the locked screening ceiling")
        patience = min(patience, screen["patience"])
    transformer_lr = config.get("learning_rate", 2e-5)
    head_lr = transformer_lr * 25  # 5e-4

    model_name = f"ablation_{args.variant}_s{seed}"
    logger.info(f"Training {model_name} (transformer={transformer_model}) on {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    processed_dir = (
        args.data_dir
        or (PROJECT_ROOT / protocol["dataset"]["output_dir"] if protocol else None)
        or Path(paths["data"]["processed_dir"])
    )
    train_df, dev_df = load_train_dev(processed_dir)
    if protocol:
        verify_dataset_manifest(
            processed_dir, protocol_hash=protocol_hash, splits=("train", "dev")
        )

    if "sentence" not in train_df.columns:
        logger.warning("No 'sentence' column. Building context cache from external corpus.")
        corpus_dir = Path(paths["data"]["raw_dir"]) / "monolingual"
        context_cache = PROJECT_ROOT / "artifacts" / "context" / "lemma_sentences.json"
        train_df, dev_df = attach_context_columns(
            train_df, dev_df,
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

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
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

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

    training_started = time.perf_counter()
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
    training_seconds = time.perf_counter() - training_started
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    inference_started = time.perf_counter()
    tokenizer.reset_stats()
    dev_metrics, dev_preds, dev_probs = evaluate(model, dev_loader, device)
    dev_metrics["training_seconds"] = training_seconds
    dev_metrics["inference_latency_ms_per_example"] = (
        1000 * (time.perf_counter() - inference_started) / len(dev_df)
    )
    dev_metrics.update({
        "invalid_prediction_count": 0,
        "target_span_failure_count": tokenizer.target_span_failure_count,
        "nan_count": 0,
    })

    print_metrics_summary(
        model_name=model_name,
        train_size=len(train_df), dev_size=len(dev_df),
        dev_metrics=dev_metrics,
        output_dir=Path(paths["results"]["dir"]),
    )

    # ── Save outputs ──────────────────────────────────────────────────────
    results_dir = Path(paths["results"]["dir"])
    metrics_path = save_metrics(
        {**dev_metrics, "split": "dev"}, f"{model_name}_dev", results_dir / "metrics",
        extra_info={
            "seed": seed, "transformer_model": transformer_model,
            "focal_gamma": args.focal_gamma, "freeze_top_n": freeze_top_n,
        },
    )

    dev_output = dev_df.copy()
    dev_output["pred_cefr"] = dev_preds
    dev_output["gold_cefr"] = dev_output["cefr"]
    for i, label in enumerate(CEFR_LABELS):
        dev_output[f"prob_{label}"] = dev_probs[:, i]
    predictions_path = save_predictions(
        dev_output, model_name, results_dir / "predictions", split="dev"
    )

    update_run_index(
        model_name=model_name, config="training.yaml",
        dev_metrics=dev_metrics,
        index_path=Path(paths["results"]["run_index"]),
    )
    if protocol:
        checkpoint_path = model_dir / "best_model.pt"
        manifest = build_run_manifest(
            root=PROJECT_ROOT,
            series_id=protocol["series"]["id"],
            proposal_id=args.proposal_id,
            protocol_hash=protocol_hash,
            seed=seed,
            configuration={
                "variant": args.variant, "transformer_model": transformer_model,
                "freeze_top_n": freeze_top_n, "epochs": epochs,
                "focal_gamma": args.focal_gamma,
            },
            command=[sys.executable, *sys.argv],
            metrics=dev_metrics,
            runtime_seconds=time.perf_counter() - started_at,
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            peak_vram_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else 0,
            artifacts=[checkpoint_path, metrics_path, predictions_path],
            outcome="diagnostic_complete",
        )
        run_dir = (
            results_dir / "autoresearch" / protocol["series"]["id"]
            / args.proposal_id / f"seed-{seed}"
        )
        write_json(run_dir / "run-manifest.json", manifest, exclusive=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

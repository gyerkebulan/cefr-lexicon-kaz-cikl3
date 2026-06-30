#!/usr/bin/env python3
"""
Train dual-encoder model for CEFR classification.

Usage:
    python scripts/train_dual.py --variant dual_gated
    python scripts/train_dual.py --variant dual_concat
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from src.utils import (
    load_config, load_paths, setup_logging, set_seed, get_device,
    CEFR_LABELS, CEFR_TO_INT
)
from src.data_prep import load_splits, print_label_distribution
from src.features import SuffixInventory, augment_with_features, get_feature_columns
from src.metrics import (
    compute_metrics, save_metrics, save_predictions,
    update_run_index, print_metrics_summary
)
from src.models.dual_encoder import DualEncoderCEFR, collate_dual_encoder
from src.models.morph_encoder import CharVocab
from src.models.context_encoder import ContextTokenizer
from src.models.coral_loss import coral_loss, coral_predict, CombinedOrdinalLoss

MORPH_FEATURE_PREFIXES = (
    "morph_",
    "n_analyses",
    "n_morphemes",
    "derivational_",
    "has_",
    "apertium_",
    "suffix_",
    "case_",
    "poss_",
    "verb_",
    "hfst_",
    "recognized_",
    "not_recognized_",
)
MORPH_FEATURE_EXACT = {
    "morphemes_per_char",
    "deriv_ratio",
    "morph_ngram_deriv_infl",
    "morph_ngram_plural_case",
    "morph_ngram_possessive_case",
    "morph_ngram_deriv_case",
}
KAZAKH_TOKEN_RE = re.compile(r"[а-яәғқңөұүһі]+", re.IGNORECASE)


class DualDataset(torch.utils.data.Dataset):
    """Dataset for dual encoder."""

    def __init__(self, df, tokenizer, char_vocab, feature_cols, max_char_len=30):
        self.sentences = df['sentence'].tolist() if 'sentence' in df.columns else df['lemma'].tolist()
        self.targets = df['target'].tolist() if 'target' in df.columns else df['lemma'].tolist()
        self.lemmas = df['lemma'].tolist()
        self.labels = [CEFR_TO_INT[l] for l in df['cefr'].tolist()]

        # Extract features
        available_cols = [c for c in feature_cols if c in df.columns]
        self.features = torch.tensor(df[available_cols].values, dtype=torch.float32)

        self.tokenizer = tokenizer
        self.char_vocab = char_vocab
        self.max_char_len = max_char_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # Tokenize sentence
        encoded = self.tokenizer([self.sentences[idx]], [self.targets[idx]])

        # Character encoding
        char_ids = self.char_vocab.encode(self.lemmas[idx], self.max_char_len)

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'target_mask': encoded['target_mask'].squeeze(0),
            'char_ids': torch.tensor(char_ids, dtype=torch.long),
            'morph_features': self.features[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def compute_class_weights(train_df, device):
    """Compute inverse-frequency class weights."""
    counts = train_df['cefr'].value_counts()
    class_counts = torch.tensor(
        [counts.get(label, 1) for label in CEFR_LABELS],
        dtype=torch.float32
    )
    weights = (1.0 / class_counts) * class_counts.sum() / len(CEFR_LABELS)
    return weights.to(device)


class ClassBalancedBatchSampler(Sampler):
    """Sample near-equal class counts in each batch via replacement."""

    def __init__(self, labels, batch_size: int, num_classes: int, seed: int = 42):
        if batch_size < num_classes:
            raise ValueError(
                f"batch_size ({batch_size}) must be >= num_classes ({num_classes}) for balanced batches"
            )
        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.samples_per_class = self.batch_size // self.num_classes
        self.remainder = self.batch_size % self.num_classes
        self.num_batches = math.ceil(len(self.labels) / self.batch_size)
        self._epoch = 0

        self.class_indices = {
            class_id: np.where(self.labels == class_id)[0]
            for class_id in range(self.num_classes)
        }
        missing = [class_id for class_id, idxs in self.class_indices.items() if len(idxs) == 0]
        if missing:
            raise ValueError(f"Cannot build balanced sampler; missing classes in train split: {missing}")

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        class_ids = np.arange(self.num_classes)
        for _ in range(self.num_batches):
            batch = []
            extra = set(rng.choice(class_ids, size=self.remainder, replace=False).tolist()) if self.remainder else set()
            for class_id in class_ids:
                n_samples = self.samples_per_class + (1 if class_id in extra else 0)
                sampled = rng.choice(self.class_indices[int(class_id)], size=n_samples, replace=True)
                batch.extend(sampled.tolist())
            rng.shuffle(batch)
            yield batch


def ordinal_soft_labels(label, num_classes=5, sigma=0.5):
    """Generate Gaussian soft targets centered on the true ordinal label."""
    targets = torch.zeros(num_classes)
    for i in range(num_classes):
        targets[i] = math.exp(-((i - label) ** 2) / (2 * sigma ** 2))
    return targets / targets.sum()


def soft_cross_entropy(logits, soft_targets):
    """Cross-entropy loss with soft (non-one-hot) targets."""
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def focal_loss(logits, labels, gamma=2.0, class_weights=None):
    """Multi-class focal loss with optional per-class weighting."""
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()

    labels = labels.view(-1, 1)
    target_log_probs = log_probs.gather(1, labels).squeeze(1)
    target_probs = probs.gather(1, labels).squeeze(1)

    focal_factor = (1.0 - target_probs).clamp(min=1e-6).pow(gamma)
    loss = -focal_factor * target_log_probs

    if class_weights is not None:
        class_alpha = class_weights[labels.squeeze(1)]
        loss = loss * class_alpha

    return loss.mean()


def earth_mover_distance_loss(logits, labels, num_classes=5, class_weights=None):
    """Ordinal EMD loss using CDF distance between predicted and target distributions."""
    probs = F.softmax(logits, dim=1)
    targets = F.one_hot(labels, num_classes=num_classes).float()
    cdf_pred = torch.cumsum(probs, dim=1)
    cdf_true = torch.cumsum(targets, dim=1)
    # Normalized L1 distance over cumulative distributions.
    per_sample = torch.abs(cdf_pred - cdf_true).sum(dim=1) / max(num_classes - 1, 1)
    if class_weights is not None:
        per_sample = per_sample * class_weights[labels]
    return per_sample.mean()


def symmetric_kl_loss(logits_a, logits_b):
    """Symmetric KL divergence between two logits distributions."""
    log_probs_a = F.log_softmax(logits_a, dim=1)
    log_probs_b = F.log_softmax(logits_b, dim=1)
    probs_a = log_probs_a.exp()
    probs_b = log_probs_b.exp()

    kl_ab = F.kl_div(log_probs_a, probs_b, reduction="batchmean")
    kl_ba = F.kl_div(log_probs_b, probs_a, reduction="batchmean")
    return 0.5 * (kl_ab + kl_ba)


def is_morph_feature(column: str) -> bool:
    return column in MORPH_FEATURE_EXACT or column.startswith(MORPH_FEATURE_PREFIXES)


def select_feature_subset(
    train_df: pd.DataFrame,
    feature_cols,
    *,
    top_k: int,
    min_morph_features: int,
    seed: int,
):
    available_cols = [c for c in feature_cols if c in train_df.columns]
    if not available_cols:
        return [], [], {}

    X = train_df[available_cols].fillna(0.0)
    variances = X.var(axis=0)
    non_constant_cols = variances[variances > 1e-10].index.tolist()
    dropped_cols = [c for c in available_cols if c not in non_constant_cols]
    if not non_constant_cols:
        return available_cols, dropped_cols, {}

    if top_k <= 0 or top_k >= len(non_constant_cols):
        selected_cols = non_constant_cols
        return selected_cols, dropped_cols, {}

    label_col = "cefr" if "cefr" in train_df.columns else "cefr_level"
    y = np.array([CEFR_TO_INT[label] for label in train_df[label_col]], dtype=np.int64)
    forest = ExtraTreesClassifier(
        n_estimators=400,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )
    forest.fit(X[non_constant_cols], y)
    importance = pd.Series(forest.feature_importances_, index=non_constant_cols).sort_values(ascending=False)

    selected_cols = list(importance.head(top_k).index)
    morph_ranked = [col for col in importance.index if is_morph_feature(col)]
    current_morph = sum(is_morph_feature(col) for col in selected_cols)
    for col in morph_ranked:
        if current_morph >= min_morph_features:
            break
        if col in selected_cols:
            continue
        selected_cols.append(col)
        current_morph += 1

    return selected_cols, dropped_cols, importance.to_dict()


def compute_feature_stats(train_df: pd.DataFrame, feature_cols):
    feature_stats = {}
    for col in feature_cols:
        values = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0)
        mean = float(values.mean())
        std = float(values.std())
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        feature_stats[col] = {"mean": mean, "std": std}
    return feature_stats


def apply_feature_stats(df: pd.DataFrame, feature_stats):
    df = df.copy()
    for col, stats in feature_stats.items():
        values = pd.to_numeric(df[col], errors="coerce").fillna(0.0) if col in df.columns else 0.0
        df[col] = ((values - stats["mean"]) / stats["std"]).astype(np.float32)
    return df


def build_lemma_context_map(lemmas, corpus_dir: Path, cache_path: Path, logger):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lemma_list = sorted({str(lemma).lower() for lemma in lemmas if str(lemma).strip()})
    context_map = {}

    if cache_path.exists():
        context_map = json.loads(cache_path.read_text(encoding="utf-8"))
        context_map = {str(k).lower(): str(v) for k, v in context_map.items()}

    remaining = set(lemma_list) - set(context_map)
    if not remaining:
        logger.info(f"Loaded cached context sentences for {len(context_map)} lemmas from {cache_path}")
        return context_map

    sent_files = sorted(corpus_dir.rglob("*-sentences.txt"))
    logger.info(
        f"Building lemma context cache from {len(sent_files)} sentence files; "
        f"{len(remaining)} lemmas still need a sentence"
    )

    for sent_file in sent_files:
        logger.info(f"Scanning contexts from {sent_file.name}")
        with sent_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                sentence = parts[1] if len(parts) == 2 else parts[0]
                tokens = set(KAZAKH_TOKEN_RE.findall(sentence.lower()))
                matches = tokens & remaining
                if not matches:
                    continue
                for lemma in matches:
                    context_map[lemma] = sentence
                remaining -= matches
                if not remaining:
                    break
        if not remaining:
            break

    cache_path.write_text(json.dumps(context_map, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        f"Context sentence coverage: {len(context_map)}/{len(lemma_list)} "
        f"({100.0 * len(context_map) / max(len(lemma_list), 1):.1f}%)"
    )
    return context_map


def attach_context_columns(train_df, dev_df, test_df, corpus_dir: Path, cache_path: Path, logger):
    all_lemmas = pd.concat(
        [train_df["lemma"], dev_df["lemma"], test_df["lemma"]], ignore_index=True
    ).astype(str)
    context_map = build_lemma_context_map(all_lemmas.tolist(), corpus_dir, cache_path, logger)

    enriched = []
    for split_name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        df = df.copy()
        df["target"] = df["lemma"].astype(str)
        df["sentence"] = df["lemma"].astype(str).map(
            lambda lemma: context_map.get(lemma.lower(), str(lemma))
        )
        found = (df["sentence"] != df["lemma"].astype(str)).sum()
        logger.info(
            f"Attached corpus context for {split_name}: {found}/{len(df)} "
            f"({100.0 * found / max(len(df), 1):.1f}%)"
        )
        enriched.append(df)
    return tuple(enriched)


def train_epoch(model, loader, optimizer, scheduler, device,
                class_weights=None, label_smoothing_sigma=0.0,
                use_ordinal=False, ordinal_weight=0.5,
                loss_type="ce", focal_gamma=2.0, emd_weight=0.0, r_drop_alpha=0.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_classes = 5

    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target_mask = batch['target_mask'].to(device)
        char_ids = batch['char_ids'].to(device)
        morph_features = batch['morph_features'].to(device)
        labels = batch['labels'].to(device)

        if use_ordinal:
            logits, ordinal_logits = model(
                input_ids, attention_mask, target_mask, char_ids, morph_features,
                return_ordinal=True
            )
            ce_loss = F.cross_entropy(logits, labels, weight=class_weights)
            ord_loss = coral_loss(ordinal_logits, labels, num_classes=num_classes)
            loss = ce_loss + ordinal_weight * ord_loss
        else:
            def compute_cls_loss(current_logits):
                if loss_type == "focal":
                    base_loss = focal_loss(
                        current_logits,
                        labels,
                        gamma=focal_gamma,
                        class_weights=class_weights,
                    )
                elif loss_type == "emd":
                    base_loss = earth_mover_distance_loss(
                        current_logits,
                        labels,
                        num_classes=num_classes,
                        class_weights=class_weights,
                    )
                elif label_smoothing_sigma > 0:
                    # Ordinal-aware Gaussian label smoothing
                    soft_targets = torch.stack([
                        ordinal_soft_labels(l.item(), num_classes, label_smoothing_sigma)
                        for l in labels
                    ]).to(device)
                    # Incorporate class weights into soft targets
                    if class_weights is not None:
                        per_sample_weight = class_weights[labels]
                        sample_loss = -(F.log_softmax(current_logits, dim=1) * soft_targets).sum(dim=1)
                        base_loss = (sample_loss * per_sample_weight).mean()
                    else:
                        base_loss = soft_cross_entropy(current_logits, soft_targets)
                else:
                    base_loss = F.cross_entropy(current_logits, labels, weight=class_weights)

                if emd_weight > 0 and loss_type != "emd":
                    base_loss = base_loss + emd_weight * earth_mover_distance_loss(
                        current_logits,
                        labels,
                        num_classes=num_classes,
                        class_weights=class_weights,
                    )
                return base_loss

            if r_drop_alpha > 0:
                logits_1 = model(input_ids, attention_mask, target_mask, char_ids, morph_features)
                logits_2 = model(input_ids, attention_mask, target_mask, char_ids, morph_features)
                cls_loss = 0.5 * (compute_cls_loss(logits_1) + compute_cls_loss(logits_2))
                reg_loss = symmetric_kl_loss(logits_1, logits_2)
                loss = cls_loss + r_drop_alpha * reg_loss
            else:
                logits = model(input_ids, attention_mask, target_mask, char_ids, morph_features)
                loss = compute_cls_loss(logits)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, use_ordinal=False):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target_mask = batch['target_mask'].to(device)
        char_ids = batch['char_ids'].to(device)
        morph_features = batch['morph_features'].to(device)
        labels = batch['labels']

        if use_ordinal:
            logits, ordinal_logits = model(
                input_ids, attention_mask, target_mask, char_ids, morph_features,
                return_ordinal=True
            )
            preds_tensor, probs_tensor = coral_predict(ordinal_logits)
            preds = preds_tensor.cpu().tolist()
            probs = probs_tensor.cpu().numpy()
        else:
            logits = model(input_ids, attention_mask, target_mask, char_ids, morph_features)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().tolist()

        all_preds.extend([CEFR_LABELS[p] for p in preds])
        all_labels.extend([CEFR_LABELS[l] for l in labels.tolist()])
        all_probs.extend(probs)

    metrics = compute_metrics(all_labels, all_preds)
    return metrics, all_preds, np.array(all_probs)


def main():
    parser = argparse.ArgumentParser(description="Train dual encoder model")
    parser.add_argument("--variant", default="dual_gated", choices=["dual_gated", "dual_concat", "dual_cross_attn", "dual_residual_gated"])
    parser.add_argument("--pooling", default="mean", choices=["mean", "cls", "first", "attention"], help="Context encoder pooling strategy")
    parser.add_argument("--ordinal", action="store_true", help="Use CORAL ordinal regression loss")
    parser.add_argument("--ordinal-weight", type=float, default=0.5,
                        help="Weight for ordinal loss component (default: 0.5)")
    parser.add_argument("--freeze-bert", action="store_true",
                        help="Freeze BERT transformer layers")
    parser.add_argument("--unfreeze-layers", type=int, default=None,
                        help="Override number of top transformer layers to unfreeze")
    parser.add_argument("--tag", type=str, default="v3",
                        help="Version tag for model name (default: v3)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Optional split directory override containing train/dev/test CSVs")
    parser.add_argument("--checkpoint-path", type=Path, default=None,
                        help="Optional checkpoint to load before training/evaluation")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training and evaluate a loaded checkpoint")
    parser.add_argument("--feature-selection", default="importance", choices=["importance", "all"],
                        help="How to choose MorphMLP features")
    parser.add_argument("--feature-top-k", type=int, default=72,
                        help="Top-k train-selected features to retain before adding morphology quota")
    parser.add_argument("--min-morph-features", type=int, default=20,
                        help="Minimum number of morphology-centric features kept in the selected subset")
    parser.add_argument("--include-static-embeddings", action="store_true",
                        help="Include precomputed emb_* columns in the MorphMLP input")
    parser.add_argument("--transformer-model", type=str, default=None,
                        help="Optional transformer model name or local path override")
    parser.add_argument("--loss-type", default="ce", choices=["ce", "focal", "emd"],
                        help="Primary classification loss when not using --ordinal")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Gamma value for focal loss (used when --loss-type focal)")
    parser.add_argument("--emd-weight", type=float, default=0.0,
                        help="Optional auxiliary EMD weight added on top of CE/focal losses")
    parser.add_argument("--r-drop-alpha", type=float, default=0.0,
                        help="R-Drop symmetric KL coefficient (0 disables R-Drop)")
    parser.add_argument("--class-balanced-batches", action="store_true",
                        help="Use a class-balanced batch sampler for train DataLoader")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed override")
    args = parser.parse_args()

    # Setup
    logger = setup_logging()
    paths = load_paths()
    config = load_config("training")
    effective_seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    set_seed(effective_seed)
    device = get_device()

    # Determine fusion type from variant
    if args.variant == "dual_gated":
        fusion_type = "gated"
    elif args.variant == "dual_residual_gated":
        fusion_type = "residual_gated"
    elif args.variant == "dual_cross_attn":
        fusion_type = "cross_attention"
    else:
        fusion_type = "concat"
        
    model_name = args.variant + f"_{args.tag}"
    if args.seed is not None:
        model_name += f"_s{effective_seed}"
    if args.ordinal:
        model_name += "_ordinal"
    if args.freeze_bert:
        model_name += "_frozen"
    if args.loss_type == "focal":
        model_name += f"_focalg{str(args.focal_gamma).replace('.', 'p')}"
    elif args.loss_type == "emd":
        model_name += "_emd"
    if args.emd_weight > 0 and args.loss_type != "emd":
        model_name += f"_emdw{str(args.emd_weight).replace('.', 'p')}"
    if args.r_drop_alpha > 0:
        model_name += f"_rdrop{str(args.r_drop_alpha).replace('.', 'p')}"
    if args.class_balanced_batches:
        model_name += "_cbatch"
    use_ordinal = args.ordinal

    # Config values
    dropout = config.get("dropout", 0.3)
    freeze_top_n = args.unfreeze_layers or config.get("freeze_transformer_except_top_n", 2)
    use_class_weights = config.get("use_class_weights", True)
    label_smoothing_sigma = config.get("label_smoothing_sigma", 0.5)
    transformer_model = args.transformer_model or config.get("transformer_model", "xlm-roberta-base")
    if args.loss_type in {"focal", "emd"} and not use_ordinal and label_smoothing_sigma > 0:
        logger.info(f"Disabling label smoothing because {args.loss_type} loss is enabled")
        label_smoothing_sigma = 0.0
    if args.r_drop_alpha > 0 and use_ordinal:
        logger.info("Disabling R-Drop because ordinal mode is enabled")
        args.r_drop_alpha = 0.0
    if args.class_balanced_batches and use_class_weights:
        logger.info("Disabling class weights because class-balanced batches are enabled")
        use_class_weights = False

    logger.info(
        f"Training {model_name} (fusion={fusion_type}, ordinal={use_ordinal}, "
        f"model={transformer_model}, freeze_top_n={freeze_top_n}, "
        f"dropout={dropout}, class_weights={use_class_weights}, loss={args.loss_type}, "
        f"emd_weight={args.emd_weight}, r_drop_alpha={args.r_drop_alpha}, "
        f"class_balanced_batches={args.class_balanced_batches}, "
        f"label_smooth_sigma={label_smoothing_sigma}, seed={effective_seed}) on {device}"
    )

    # Load data
    processed_dir = args.data_dir if args.data_dir is not None else Path(paths["data"]["processed_dir"])
    logger.info(f"Loading splits from {processed_dir}")
    train_df, dev_df, test_df = load_splits(processed_dir)
    print_label_distribution(train_df, "Train")

    # Check if we have context data
    has_context = 'sentence' in train_df.columns
    if not has_context:
        logger.warning("No 'sentence' column. Building context cache from the external corpus.")
        corpus_dir = Path(paths["data"]["raw_dir"]) / "monolingual"
        context_cache = PROJECT_ROOT / "artifacts" / "context" / "lemma_sentences.json"
        train_df, dev_df, test_df = attach_context_columns(
            train_df, dev_df, test_df, corpus_dir=corpus_dir, cache_path=context_cache, logger=logger
        )

    # Extract any missing features
    suffix_inv = SuffixInventory(config.get("lang", "kaz"))
    train_df = augment_with_features(train_df, suffix_inventory=suffix_inv)
    dev_df = augment_with_features(dev_df, suffix_inventory=suffix_inv)
    test_df = augment_with_features(test_df, suffix_inventory=suffix_inv)

    # Handcrafted features drive the MorphMLP; static emb_* vectors are optional.
    feature_cols = get_feature_columns()
    emb_cols = [c for c in train_df.columns if c.startswith('emb_')]
    if args.include_static_embeddings:
        feature_cols = feature_cols + emb_cols

    # Fill NaN for features that may be missing in some rows
    for df in [train_df, dev_df, test_df]:
        for c in feature_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0)

    model_dir = Path(paths["results"]["models"]) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    feature_spec_path = model_dir / "feature_spec.json"
    loaded_feature_spec = None

    if args.checkpoint_path is not None:
        checkpoint_spec = args.checkpoint_path.parent / "feature_spec.json"
        if checkpoint_spec.exists():
            loaded_feature_spec = json.loads(checkpoint_spec.read_text(encoding="utf-8"))

    if loaded_feature_spec is not None:
        selected_feature_cols = loaded_feature_spec["selected_feature_cols"]
        feature_stats = loaded_feature_spec["feature_stats"]
        logger.info(f"Loaded feature spec from {args.checkpoint_path.parent / 'feature_spec.json'}")
    else:
        if args.feature_selection == "all":
            available_feature_cols = [c for c in feature_cols if c in train_df.columns]
            dropped_feature_cols = [
                c for c in available_feature_cols
                if float(pd.to_numeric(train_df[c], errors="coerce").fillna(0.0).var()) <= 1e-10
            ]
            selected_feature_cols = [c for c in available_feature_cols if c not in dropped_feature_cols]
            feature_importance = {}
        else:
            selected_feature_cols, dropped_feature_cols, feature_importance = select_feature_subset(
                train_df,
                feature_cols,
                top_k=args.feature_top_k,
                min_morph_features=args.min_morph_features,
                seed=effective_seed,
            )
        feature_stats = compute_feature_stats(train_df, selected_feature_cols)
        feature_spec_path.write_text(
            json.dumps(
                {
                    "seed": effective_seed,
                    "selected_feature_cols": selected_feature_cols,
                    "feature_stats": feature_stats,
                    "feature_selection": args.feature_selection,
                    "feature_top_k": args.feature_top_k,
                    "min_morph_features": args.min_morph_features,
                    "include_static_embeddings": args.include_static_embeddings,
                    "dropped_feature_cols": dropped_feature_cols,
                    "feature_importance": feature_importance,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    train_df = apply_feature_stats(train_df, feature_stats)
    dev_df = apply_feature_stats(dev_df, feature_stats)
    test_df = apply_feature_stats(test_df, feature_stats)

    morph_feature_count = sum(is_morph_feature(col) for col in selected_feature_cols)
    logger.info(
        f"Using {len(selected_feature_cols)} scaled features "
        f"(morph={morph_feature_count}, static_emb={args.include_static_embeddings})"
    )
    logger.info(f"Selected features: {selected_feature_cols[:40]}")
    morph_input_dim = len(selected_feature_cols)

    # Tokenizer and char vocab
    tokenizer = ContextTokenizer(
        model_name=transformer_model,
        max_length=config.get("max_seq_length", 128)
    )
    char_vocab = CharVocab()

    # Datasets
    train_dataset = DualDataset(train_df, tokenizer, char_vocab, selected_feature_cols)
    dev_dataset = DualDataset(dev_df, tokenizer, char_vocab, selected_feature_cols)
    test_dataset = DualDataset(test_df, tokenizer, char_vocab, selected_feature_cols)

    # Dataloaders
    batch_size = args.batch_size or config.get("batch_size", 16)
    if args.class_balanced_batches:
        balanced_sampler = ClassBalancedBatchSampler(
            train_dataset.labels,
            batch_size=batch_size,
            num_classes=len(CEFR_LABELS),
            seed=effective_seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=balanced_sampler,
            collate_fn=collate_dual_encoder,
        )
        logger.info(
            f"Using class-balanced batch sampler: batches={len(balanced_sampler)}, "
            f"batch_size={batch_size}, classes={len(CEFR_LABELS)}"
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_dual_encoder,
        )
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_dual_encoder)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_dual_encoder)

    # Class weights
    class_weights = compute_class_weights(train_df, device) if use_class_weights else None
    if class_weights is not None:
        logger.info(f"Class weights: {dict(zip(CEFR_LABELS, class_weights.cpu().tolist()))}")

    # Model
    freeze_transformer = args.freeze_bert or config.get("freeze_transformer", False)
    model = DualEncoderCEFR(
        transformer_model=transformer_model,
        freeze_transformer=freeze_transformer,
        freeze_except_top_n=0 if freeze_transformer else freeze_top_n,
        pooling=args.pooling,
        char_vocab_size=char_vocab.vocab_size,
        char_embed_dim=config.get("char_embed_dim", 64),
        char_num_filters=config.get("char_num_filters", 128),
        char_kernel_sizes=config.get("char_kernel_sizes", [2, 3, 4]),
        morph_input_dim=morph_input_dim,
        morph_hidden_dim=config.get("morph_hidden_dim", 64),
        morph_output_dim=config.get("morph_output_dim", 128),
        fusion_type=fusion_type,
        fusion_output_dim=config.get("fusion_output_dim", 256),
        num_classes=5,
        dropout=dropout
    )
    tokenizer.resize_embeddings(model.context_encoder)
    model = model.to(device)

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        load_result = model.load_state_dict(checkpoint, strict=False)
        logger.info(
            f"Loaded checkpoint from {args.checkpoint_path} "
            f"(missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys})"
        )

    # Discriminative learning rates: lower for transformer, higher for heads
    transformer_lr = config.get("learning_rate", 2e-5)
    head_lr = transformer_lr * 25  # 5e-4 when base is 2e-5

    optimizer = torch.optim.AdamW([
        {"params": model.context_encoder.parameters(), "lr": transformer_lr},
        {"params": model.morph_encoder.parameters(), "lr": head_lr},
        {"params": model.fusion.parameters(), "lr": head_lr},
        {"params": model.classifier.parameters(), "lr": head_lr},
        {"params": model.ordinal_head.parameters(), "lr": head_lr},
    ], weight_decay=config.get("weight_decay", 0.01))

    logger.info(f"Discriminative LR: transformer={transformer_lr}, heads={head_lr}")

    epochs = args.epochs or config.get("epochs", 20)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    best_dev_f1 = 0
    patience_counter = 0
    patience = config.get("early_stopping_patience", 7)
    if not args.eval_only:
        for epoch in range(epochs):
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, device,
                class_weights=class_weights,
                label_smoothing_sigma=label_smoothing_sigma,
                use_ordinal=use_ordinal,
                ordinal_weight=args.ordinal_weight,
                loss_type=args.loss_type,
                focal_gamma=args.focal_gamma,
                emd_weight=args.emd_weight,
                r_drop_alpha=args.r_drop_alpha,
            )
            dev_metrics, _, _ = evaluate(model, dev_loader, device, use_ordinal=use_ordinal)

            logger.info(
                f"Epoch {epoch+1}/{epochs}: loss={train_loss:.4f}, "
                f"dev_acc={dev_metrics['accuracy']:.4f}, dev_f1={dev_metrics['macro_f1']:.4f}"
                f"{' *' if dev_metrics['macro_f1'] > best_dev_f1 else ''}"
            )

            # Early stopping
            if dev_metrics['macro_f1'] > best_dev_f1:
                best_dev_f1 = dev_metrics['macro_f1']
                patience_counter = 0
                torch.save(model.state_dict(), model_dir / "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    elif args.checkpoint_path is None:
        raise ValueError("--eval-only requires --checkpoint-path")

    # Load best and evaluate
    dev_metrics, dev_preds, dev_probs = evaluate(model, dev_loader, device, use_ordinal=use_ordinal)
    test_metrics, test_preds, test_probs = evaluate(model, test_loader, device, use_ordinal=use_ordinal)

    # Print summary
    print_metrics_summary(
        model_name=model_name,
        train_size=len(train_df),
        dev_size=len(dev_df),
        test_size=len(test_df),
        dev_metrics=dev_metrics,
        test_metrics=test_metrics,
        output_dir=Path(paths["results"]["dir"])
    )

    # Save outputs
    results_dir = Path(paths["results"]["dir"])
    save_metrics(
        test_metrics,
        f"{model_name}_test",
        results_dir / "metrics",
        extra_info={
            "seed": effective_seed,
            "feature_selection": args.feature_selection,
            "feature_top_k": args.feature_top_k,
            "min_morph_features": args.min_morph_features,
            "transformer_model": transformer_model,
            "loss_type": args.loss_type,
            "focal_gamma": args.focal_gamma,
            "emd_weight": args.emd_weight,
            "class_balanced_batches": args.class_balanced_batches,
            "data_dir": str(processed_dir),
        },
    )

    # Save dev predictions for stacking
    dev_df_out = dev_df.copy()
    dev_df_out['pred_cefr'] = dev_preds
    dev_df_out['gold_cefr'] = dev_df_out['cefr']
    for i, label in enumerate(CEFR_LABELS):
        dev_df_out[f'prob_{label}'] = dev_probs[:, i]
    save_predictions(dev_df_out, model_name, results_dir / "predictions", split="dev")

    # Save predictions
    test_df_out = test_df.copy()
    test_df_out['pred_cefr'] = test_preds
    test_df_out['gold_cefr'] = test_df_out['cefr']
    for i, label in enumerate(CEFR_LABELS):
        test_df_out[f'prob_{label}'] = test_probs[:, i]
    save_predictions(test_df_out, model_name, results_dir / "predictions")

    update_run_index(
        model_name=model_name,
        config="training.yaml",
        dev_metrics=dev_metrics,
        test_metrics=test_metrics,
        index_path=Path(paths["results"]["run_index"])
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

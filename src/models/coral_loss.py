"""
CORAL (Consistent Rank Logits) ordinal regression for CEFR classification.

Instead of K classes with cross-entropy, learns K-1 binary classifiers:
  "Is the label > A1?", "Is the label > A2?", etc.

Reference:
  Cao, Mirjalili, Raschka (2020) "Rank Consistent Ordinal Regression for
  Neural Networks with Application to Age Estimation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralOrdinalClassifier(nn.Module):
    """CORAL ordinal classifier head.

    Replaces standard Linear(hidden, num_classes) with:
    - Shared feature weights W (hidden → 1)
    - K-1 independent biases (one per ordinal threshold)
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1

        # Shared weights (rank consistent)
        self.fc = nn.Linear(input_dim, 1, bias=False)

        # Independent biases for each threshold
        self.biases = nn.Parameter(torch.zeros(self.num_thresholds))

    def forward(self, x):
        """Forward pass.

        Args:
            x: [batch, hidden_dim] feature vector

        Returns:
            ordinal_logits: [batch, K-1] logits for each threshold
        """
        # Shared projection: [batch, 1]
        shared = self.fc(x)

        # Add independent biases: [batch, K-1]
        ordinal_logits = shared + self.biases.unsqueeze(0)

        return ordinal_logits


def coral_loss(ordinal_logits, labels, num_classes):
    """CORAL loss: sum of binary cross-entropy for ordinal thresholds.

    Args:
        ordinal_logits: [batch, K-1] logits for each threshold
        labels: [batch] integer labels (0 to K-1)
        num_classes: K

    Returns:
        loss: scalar
    """
    num_thresholds = num_classes - 1

    # Create ordinal targets: [batch, K-1]
    # For label k: targets = [1, 1, ..., 1, 0, 0, ..., 0]
    #              first k ones, rest zeros
    ordinal_targets = torch.zeros(labels.size(0), num_thresholds,
                                  device=labels.device, dtype=torch.float32)

    for i in range(num_thresholds):
        ordinal_targets[:, i] = (labels > i).float()

    # Binary cross-entropy for each threshold
    loss = F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_targets)

    return loss


def coral_predict(ordinal_logits):
    """Convert ordinal logits to class predictions.

    Args:
        ordinal_logits: [batch, K-1]

    Returns:
        predictions: [batch] integer class predictions (0 to K-1)
        probabilities: [batch, K] class probabilities
    """
    # Sigmoid to get P(label > k) for each threshold
    cumulative_probs = torch.sigmoid(ordinal_logits)  # [batch, K-1]

    # Convert cumulative to per-class probabilities
    # P(class = 0) = 1 - P(label > 0)
    # P(class = k) = P(label > k-1) - P(label > k)  for k in 1..K-2
    # P(class = K-1) = P(label > K-2)
    num_classes = ordinal_logits.size(1) + 1
    batch_size = ordinal_logits.size(0)

    probs = torch.zeros(batch_size, num_classes, device=ordinal_logits.device)
    probs[:, 0] = 1 - cumulative_probs[:, 0]
    for k in range(1, num_classes - 1):
        probs[:, k] = cumulative_probs[:, k - 1] - cumulative_probs[:, k]
    probs[:, num_classes - 1] = cumulative_probs[:, num_classes - 2]

    # Clamp to avoid negative probabilities from numerical issues
    probs = probs.clamp(min=0)
    probs = probs / probs.sum(dim=1, keepdim=True)

    # Predictions: class with highest probability
    predictions = probs.argmax(dim=1)

    return predictions, probs


class CombinedOrdinalLoss(nn.Module):
    """Combined loss: CE + lambda * CORAL ordinal loss.

    Uses standard cross-entropy for classification accuracy
    plus CORAL loss for ordinal consistency.
    """

    def __init__(self, num_classes, ordinal_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.ordinal_weight = ordinal_weight

    def forward(self, logits, ordinal_logits, labels):
        """
        Args:
            logits: [batch, K] standard classification logits
            ordinal_logits: [batch, K-1] ordinal logits
            labels: [batch] integer labels
        """
        ce_loss = F.cross_entropy(logits, labels)
        ord_loss = coral_loss(ordinal_logits, labels, self.num_classes)

        return ce_loss + self.ordinal_weight * ord_loss

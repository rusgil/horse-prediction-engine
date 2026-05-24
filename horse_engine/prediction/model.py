"""
Logistic regression prediction model.

Horse racing is a multi-runner competition. Strategy:
  1. Compute a raw logistic score per runner via w·x + b
  2. Apply softmax across the entire field → normalised win probabilities
  3. Place probability = sum of top-3 normalised scores

Retraining uses gradient descent on historical binary outcomes (win/loss).
"""
from __future__ import annotations

import json
import math
import logging
from datetime import datetime

from horse_engine.prediction.features import DEFAULT_WEIGHTS, DEFAULT_PLACE_WEIGHTS, FEATURE_NAMES, NUM_FEATURES

log = logging.getLogger(__name__)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def softmax(scores: list[float]) -> list[float]:
    mx = max(scores) if scores else 0
    exps = [math.exp(s - mx) for s in scores]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(scores)] * len(scores)
    return [e / total for e in exps]


class HorseModel:
    """Logistic regression with per-feature weights, trained on historical results."""

    def __init__(self, weights: list[float] | None = None, bias: float = 0.0):
        self.weights = list(weights or DEFAULT_WEIGHTS)
        self.bias = bias
        assert len(self.weights) == NUM_FEATURES, (
            f"Expected {NUM_FEATURES} weights, got {len(self.weights)}"
        )

    def raw_score(self, feature_vector: list[float]) -> float:
        """Dot product + bias. Higher = more likely to win."""
        return sum(w * x for w, x in zip(self.weights, feature_vector)) + self.bias

    def predict_field(
        self, feature_vectors: list[list[float]]
    ) -> tuple[list[float], list[float]]:
        """
        Returns (win_probs, place_probs) for the whole field.
        Both are normalised across the field and sum to 1 / ~3 respectively.
        """
        if not feature_vectors:
            return [], []
        raw = [self.raw_score(fv) for fv in feature_vectors]
        win_probs = softmax(raw)

        # Place prob = probability of finishing in top 3 (rough estimate)
        # Use softmax on higher-temperature scores, then scale
        temp = 0.5
        place_raw = [r * temp for r in raw]
        place_base = softmax(place_raw)
        n = len(feature_vectors)
        places = 3 if n >= 8 else 2 if n >= 5 else 1
        place_probs = [round(min(p * places, 0.95), 4) for p in place_base]

        return [round(p, 4) for p in win_probs], place_probs

    # ── Training ─────────────────────────────────────────────────────────

    def train(
        self,
        training_data: list[tuple[list[float], int]],
        learning_rate: float = 0.01,
        epochs: int = 500,
        l2: float = 0.001,
    ) -> dict:
        """
        Gradient descent on binary cross-entropy.
        training_data: list of (feature_vector, label) where label=1 if winner
        Returns dict of training stats.
        """
        if not training_data:
            return {"error": "no training data"}

        n = len(training_data)
        log.info("Training on %d examples, %d features, %d epochs", n, NUM_FEATURES, epochs)

        for epoch in range(epochs):
            total_loss = 0.0
            grad_w = [0.0] * NUM_FEATURES
            grad_b = 0.0

            for fv, label in training_data:
                y_hat = sigmoid(self.raw_score(fv))
                err = y_hat - label
                total_loss += -(label * math.log(y_hat + 1e-9) + (1 - label) * math.log(1 - y_hat + 1e-9))

                for j in range(NUM_FEATURES):
                    grad_w[j] += err * fv[j]
                grad_b += err

            # Update weights with L2 regularisation
            for j in range(NUM_FEATURES):
                self.weights[j] -= learning_rate * (grad_w[j] / n + l2 * self.weights[j])
            self.bias -= learning_rate * grad_b / n

            if epoch % 100 == 0:
                log.debug("Epoch %d loss=%.4f", epoch, total_loss / n)

        accuracy = self._accuracy(training_data)
        log.info("Training complete. Accuracy: %.1f%%", accuracy * 100)
        return {
            "examples": n,
            "epochs": epochs,
            "accuracy": round(accuracy, 4),
            "weights": dict(zip(FEATURE_NAMES, [round(w, 6) for w in self.weights])),
        }

    def _accuracy(self, training_data: list[tuple[list[float], int]]) -> float:
        correct = 0
        for fv, label in training_data:
            pred = 1 if self.raw_score(fv) > 0 else 0
            if pred == label:
                correct += 1
        return correct / len(training_data) if training_data else 0.0

    def to_dict(self) -> dict:
        return {
            "weights": dict(zip(FEATURE_NAMES, self.weights)),
            "bias": self.bias,
            "updated_at": datetime.utcnow().isoformat(),
        }

    @classmethod
    def from_weights_dict(cls, weights_dict: dict[str, float], bias: float = 0.0) -> "HorseModel":
        weights = [weights_dict.get(name, DEFAULT_WEIGHTS[i]) for i, name in enumerate(FEATURE_NAMES)]
        return cls(weights=weights, bias=bias)


class PlaceModel(HorseModel):
    """
    Logistic regression trained on P(position ≤ 3) labels.
    Same feature vector as HorseModel; different default weights and DB table.
    Used to rank legs 2 and 3 of trifecta picks.
    """

    def __init__(self, weights: list[float] | None = None, bias: float = 0.0):
        if weights is None:
            weights = list(DEFAULT_PLACE_WEIGHTS)
        # Bypass HorseModel.__init__ assertion by calling grandparent directly,
        # then set attributes manually so assertion uses correct defaults.
        object.__setattr__(self, "weights", list(weights))
        object.__setattr__(self, "bias", bias)
        assert len(self.weights) == NUM_FEATURES, (
            f"PlaceModel: expected {NUM_FEATURES} weights, got {len(self.weights)}"
        )

    @classmethod
    def from_weights_dict(cls, weights_dict: dict[str, float], bias: float = 0.0) -> "PlaceModel":
        weights = [weights_dict.get(name, DEFAULT_PLACE_WEIGHTS[i]) for i, name in enumerate(FEATURE_NAMES)]
        return cls(weights=weights, bias=bias)

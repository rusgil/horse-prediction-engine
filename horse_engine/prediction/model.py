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

from horse_engine.prediction.features import (
    DEFAULT_WEIGHTS, DEFAULT_PLACE_WEIGHTS, DEFAULT_EXOTIC_WEIGHTS,
    FEATURE_NAMES, NUM_FEATURES, WIN_MASK_INDICES,
)

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

    # Feature indices the win model should ignore. PlaceModel overrides this
    # to be empty. See features.WIN_MASK_INDICES for the list and rationale.
    _mask_indices: frozenset = WIN_MASK_INDICES

    def __init__(self, weights: list[float] | None = None, bias: float = 0.0):
        self.weights = list(weights or DEFAULT_WEIGHTS)
        self.bias = bias
        assert len(self.weights) == NUM_FEATURES, (
            f"Expected {NUM_FEATURES} weights, got {len(self.weights)}"
        )

    def _mask_fv(self, fv: list[float]) -> list[float]:
        """Zero out masked features. Cheap when nothing's masked (PlaceModel)."""
        if not self._mask_indices:
            return fv
        return [0.0 if i in self._mask_indices else v for i, v in enumerate(fv)]

    def raw_score(self, feature_vector: list[float]) -> float:
        """Dot product + bias. Higher = more likely to win."""
        fv = self._mask_fv(feature_vector)
        return sum(w * x for w, x in zip(self.weights, fv)) + self.bias

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
        sample_weights: list[float] | None = None,
    ) -> dict:
        """
        Gradient descent on binary cross-entropy with optional sample weighting.
        training_data: list of (feature_vector, label) where label=1 if winner
        sample_weights: per-example weights (e.g. exp(-days_ago/30)) — defaults to uniform
        Returns dict of training stats.
        """
        if not training_data:
            return {"error": "no training data"}

        # Same masking as train_race_grouped — keep gradient view consistent
        # with raw_score's masked view so weight updates respect the mask.
        if self._mask_indices:
            training_data = [(self._mask_fv(fv), label) for (fv, label) in training_data]

        n = len(training_data)
        weights_arr = sample_weights if sample_weights and len(sample_weights) == n else [1.0] * n
        w_total = sum(weights_arr) or 1.0
        log.info("Training on %d examples, %d features, %d epochs", n, NUM_FEATURES, epochs)

        for epoch in range(epochs):
            total_loss = 0.0
            grad_w = [0.0] * NUM_FEATURES
            grad_b = 0.0

            for (fv, label), sw in zip(training_data, weights_arr):
                y_hat = sigmoid(self.raw_score(fv))
                err = (y_hat - label) * sw
                total_loss += sw * -(label * math.log(y_hat + 1e-9) + (1 - label) * math.log(1 - y_hat + 1e-9))

                for j in range(NUM_FEATURES):
                    grad_w[j] += err * fv[j]
                grad_b += err

            # Update weights with L2 regularisation
            for j in range(NUM_FEATURES):
                self.weights[j] -= learning_rate * (grad_w[j] / w_total + l2 * self.weights[j])
            self.bias -= learning_rate * grad_b / w_total

            if epoch % 100 == 0:
                log.debug("Epoch %d loss=%.4f", epoch, total_loss / w_total)

        metrics = self._metrics(training_data)
        log.info(
            "Training complete. log_loss=%.4f top1_hit=%.1f%%",
            metrics["log_loss"], metrics["top1_hit_rate"] * 100,
        )
        return {
            "examples": n,
            "epochs": epochs,
            "log_loss": metrics["log_loss"],
            "top1_hit_rate": metrics["top1_hit_rate"],
            "accuracy": metrics["top1_hit_rate"],  # keep key for backwards-compat
            "weights": dict(zip(FEATURE_NAMES, [round(w, 6) for w in self.weights])),
        }

    def _metrics(self, training_data: list[tuple[list[float], int]]) -> dict:
        """
        Return meaningful training metrics.
        - log_loss: cross-entropy on training set (lower is better; tracks calibration)
        - top1_hit_rate: fraction of races where the highest-scored horse was the winner
          (meaningful for racing; trivial binary accuracy is not)
        """
        total_loss = 0.0
        race_scores: dict[int, tuple[float, int]] = {}  # group_id → (best_score, label)

        # log-loss
        for fv, label in training_data:
            y_hat = sigmoid(self.raw_score(fv))
            total_loss += -(label * math.log(y_hat + 1e-9) + (1 - label) * math.log(1 - y_hat + 1e-9))

        n = len(training_data)
        log_loss = total_loss / n if n else 0.0

        # top-1 hit rate: among all winner examples, did the model rank them #1 in their group?
        # We approximate by checking if each winner example has the highest raw score.
        winner_data = [(fv, self.raw_score(fv)) for fv, label in training_data if label == 1]
        hits = sum(1 for fv, score in winner_data if score > 0)
        top1_hit_rate = hits / len(winner_data) if winner_data else 0.0

        return {"log_loss": round(log_loss, 4), "top1_hit_rate": round(top1_hit_rate, 4)}

    def train_race_grouped(
        self,
        race_groups: list[list[tuple]],
        learning_rate: float = 0.01,
        epochs: int = 500,
        l2: float = 0.001,
        sample_weights: list[float] | None = None,
    ) -> dict:
        """
        Race-level conditional logit (multinomial cross-entropy).
        race_groups: List of races; each race is List[(feature_vector, label)] where label=1 for winner.
        sample_weights: per-race weights (e.g. exp(-days_ago/30)) — defaults to uniform.

        Directly optimizes -log(softmax_p_of_winner) which matches the softmax inference step,
        unlike per-runner binary CE which ignores competition within the field.
        """
        valid = [r for r in race_groups if sum(t[1] for t in r) == 1 and len(r) >= 2]
        if not valid:
            return {"error": "no races with exactly one winner and ≥2 runners"}

        # Pre-mask each runner's FV up front so gradients computed against
        # fv[k] are zero for masked features. raw_score also masks, but the
        # manual gradient loop below reads fv[k] directly. Without this,
        # masked features would still get weight updates while predict_field
        # ignored them — divergent train/predict behaviour.
        if self._mask_indices:
            valid = [[(self._mask_fv(fv), label) for (fv, label) in race] for race in valid]

        n_races = len(valid)
        race_w = sample_weights if sample_weights and len(sample_weights) == n_races else [1.0] * n_races
        w_total = sum(race_w) or 1.0
        log.info("Race-grouped training on %d races, %d features, %d epochs", n_races, NUM_FEATURES, epochs)

        for epoch in range(epochs):
            total_loss = 0.0
            grad_w = [0.0] * NUM_FEATURES
            grad_b = 0.0

            for race, rw in zip(valid, race_w):
                fvs = [t[0] for t in race]
                labels = [t[1] for t in race]
                winner_idx = next(i for i, l in enumerate(labels) if l == 1)

                scores = [self.raw_score(fv) for fv in fvs]
                probs = softmax(scores)

                total_loss += rw * -math.log(probs[winner_idx] + 1e-9)

                for i, (fv, p) in enumerate(zip(fvs, probs)):
                    err = (p - (1.0 if i == winner_idx else 0.0)) * rw
                    for k in range(NUM_FEATURES):
                        grad_w[k] += err * fv[k]
                    grad_b += err

            for j in range(NUM_FEATURES):
                self.weights[j] -= learning_rate * (grad_w[j] / w_total + l2 * self.weights[j])
            self.bias -= learning_rate * grad_b / w_total

            if epoch % 100 == 0:
                log.debug("Epoch %d loss=%.4f", epoch, total_loss / w_total)

        hits = sum(
            1 for race in valid
            if [t[1] for t in race][
                max(range(len(race)), key=lambda i: self.raw_score(race[i][0]))
            ] == 1
        )
        top1_rate = hits / n_races if n_races else 0.0
        log.info("Race-grouped complete. top1=%.1f%% (%d/%d races)", top1_rate * 100, hits, n_races)
        return {
            "races": n_races,
            "epochs": epochs,
            "top1_hit_rate": round(top1_rate, 4),
            "accuracy": round(top1_rate, 4),
            "weights": dict(zip(FEATURE_NAMES, [round(w, 6) for w in self.weights])),
        }

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

    # Place model ablation showed no feature was net-harmful — keep all 41
    # features active. The mask only applies to the win model.
    _mask_indices: frozenset = frozenset()

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


class ExoticModel(HorseModel):
    """
    Logistic regression trained for exotic bet coverage (trifecta, first four, quinella, exacta).

    Differs from PlaceModel in two ways:
      1. Default weights tuned for exotic betting (consistency over peak-speed signals)
      2. Training uses a race-grouped exotic loss: standard BCE per runner PLUS separate
         box alignment gradients for trifecta (top 3) and first four (top 4). Both push
         missed finishers up and wrongly-included runners down.
    Only trains on field_size >= 7 races.
    """

    # Exotic model is for placing in top-3/top-4, behaviour closer to PlaceModel
    # than to the win model. The win ablation doesn't apply — keep all features.
    _mask_indices: frozenset = frozenset()

    def __init__(self, weights: list[float] | None = None, bias: float = 0.0):
        if weights is None:
            weights = list(DEFAULT_EXOTIC_WEIGHTS)
        object.__setattr__(self, "weights", list(weights))
        object.__setattr__(self, "bias", bias)
        assert len(self.weights) == NUM_FEATURES, (
            f"ExoticModel: expected {NUM_FEATURES} weights, got {len(self.weights)}"
        )

    @classmethod
    def from_weights_dict(cls, weights_dict: dict[str, float], bias: float = 0.0) -> "ExoticModel":
        weights = [weights_dict.get(name, DEFAULT_EXOTIC_WEIGHTS[i]) for i, name in enumerate(FEATURE_NAMES)]
        return cls(weights=weights, bias=bias)

    def train_exotic(
        self,
        race_groups: list[list[tuple]],
        learning_rate: float = 0.01,
        epochs: int = 500,
        l2: float = 0.001,
        tri_lambda: float = 0.4,
        ff_lambda: float = 0.3,
    ) -> dict:
        """
        Race-grouped exotic training covering trifecta (top 3) and first four (top 4).

        race_groups: list of races; each race is a list of tuples:
            (feature_vector, label, position)
            - label = 1 if horse finished in top 3 (for BCE)
            - position = finishing position int, or None if unavailable
        tri_lambda: gradient weight for trifecta box alignment.
        ff_lambda:  gradient weight for first-four box alignment.
        """
        if not race_groups:
            return {"error": "no training data"}

        n_total = sum(len(race) for race in race_groups)
        n_races = len(race_groups)
        log.info(
            "[exotic] Training on %d races / %d runners, %d epochs, tri_lambda=%.2f ff_lambda=%.2f",
            n_races, n_total, epochs, tri_lambda, ff_lambda,
        )

        def _box_grad(probs, fvs, actual_set, predicted_set, lam, grad_w, grad_b):
            if actual_set and predicted_set != actual_set:
                missed = actual_set - predicted_set
                wrong  = predicted_set - actual_set
                for i in missed:
                    extra = -lam * (1.0 - probs[i])
                    for k in range(NUM_FEATURES):
                        grad_w[k] += extra * fvs[i][k]
                    grad_b += extra
                for i in wrong:
                    extra = lam * probs[i]
                    for k in range(NUM_FEATURES):
                        grad_w[k] += extra * fvs[i][k]
                    grad_b += extra
            return grad_b

        for epoch in range(epochs):
            total_loss = 0.0
            grad_w = [0.0] * NUM_FEATURES
            grad_b = 0.0

            for race in race_groups:
                fvs    = [t[0] for t in race]
                labels = [t[1] for t in race]
                positions = [t[2] if len(t) > 2 else None for t in race]
                scores = [self.raw_score(fv) for fv in fvs]
                probs  = [sigmoid(s) for s in scores]

                # Standard BCE gradient per runner (uses top-3 label)
                for j in range(len(race)):
                    err = probs[j] - labels[j]
                    total_loss += -(
                        labels[j] * math.log(probs[j] + 1e-9)
                        + (1 - labels[j]) * math.log(1 - probs[j] + 1e-9)
                    )
                    for k in range(NUM_FEATURES):
                        grad_w[k] += err * fvs[j][k]
                    grad_b += err

                if len(race) < 7:
                    continue

                ranked = sorted(range(len(race)), key=lambda i: probs[i], reverse=True)

                # Trifecta (top-3) box alignment gradient
                has_positions = any(p is not None for p in positions)
                if has_positions:
                    actual_top3 = {i for i, p in enumerate(positions) if p is not None and p in (1, 2, 3)}
                else:
                    actual_top3 = {i for i, lbl in enumerate(labels) if lbl == 1}

                if len(actual_top3) == 3:
                    predicted_top3 = set(ranked[:3])
                    grad_b = _box_grad(probs, fvs, actual_top3, predicted_top3, tri_lambda, grad_w, grad_b)

                # First-four box alignment gradient
                if has_positions:
                    actual_top4 = {i for i, p in enumerate(positions) if p is not None and p in (1, 2, 3, 4)}
                    if len(actual_top4) == 4 and len(race) >= 8:
                        predicted_top4 = set(ranked[:4])
                        grad_b = _box_grad(probs, fvs, actual_top4, predicted_top4, ff_lambda, grad_w, grad_b)

            for j in range(NUM_FEATURES):
                self.weights[j] -= learning_rate * (grad_w[j] / n_total + l2 * self.weights[j])
            self.bias -= learning_rate * grad_b / n_total

            if epoch % 100 == 0:
                log.debug("[exotic] Epoch %d loss=%.4f", epoch, total_loss / n_total)

        # Measure box hit rates on training data
        tri_hits = ff_hits = tri_eligible = ff_eligible = 0
        for race in race_groups:
            if len(race) < 7:
                continue
            fvs       = [t[0] for t in race]
            labels    = [t[1] for t in race]
            positions = [t[2] if len(t) > 2 else None for t in race]
            scores    = [self.raw_score(fv) for fv in fvs]
            ranked    = sorted(range(len(race)), key=lambda i: scores[i], reverse=True)

            has_pos = any(p is not None for p in positions)
            actual_top3 = (
                {i for i, p in enumerate(positions) if p is not None and p in (1, 2, 3)}
                if has_pos else {i for i, lbl in enumerate(labels) if lbl == 1}
            )
            if len(actual_top3) == 3:
                tri_eligible += 1
                if set(ranked[:3]) == actual_top3:
                    tri_hits += 1

            if has_pos and len(race) >= 8:
                actual_top4 = {i for i, p in enumerate(positions) if p is not None and p in (1, 2, 3, 4)}
                if len(actual_top4) == 4:
                    ff_eligible += 1
                    if set(ranked[:4]) == actual_top4:
                        ff_hits += 1

        tri_rate = tri_hits / tri_eligible if tri_eligible else 0.0
        ff_rate  = ff_hits  / ff_eligible  if ff_eligible  else 0.0
        log.info(
            "[exotic] Training complete. Tri box: %.1f%% (%d/%d)  FF box: %.1f%% (%d/%d)",
            tri_rate * 100, tri_hits, tri_eligible,
            ff_rate  * 100, ff_hits,  ff_eligible,
        )

        return {
            "races": n_races,
            "runners": n_total,
            "epochs": epochs,
            "box_hit_rate": round(tri_rate, 4),       # kept for backward compat
            "tri_box_hit_rate": round(tri_rate, 4),
            "ff_box_hit_rate": round(ff_rate, 4),
            "weights": dict(zip(FEATURE_NAMES, [round(w, 6) for w in self.weights])),
        }

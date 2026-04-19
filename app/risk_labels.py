"""Map binary P(scam) + heuristic flags to TICKET-001 3-class risk (legit / warning / fraud)."""

from __future__ import annotations

from typing import Final, Literal

PredictedLabel = Literal["legit", "warning", "fraud"]

CLASS_LEGIT: Final[int] = 0
CLASS_WARNING: Final[int] = 1
CLASS_FRAUD: Final[int] = 2


def class_from_softmax_triple(
    p_legit: float,
    p_warning: float,
    p_fraud: float,
) -> tuple[int, PredictedLabel, float]:
    """
    Map native 3-class probabilities to TICKET-001 class, label, and confidence.

    ``confidence`` is ``max(p_legit, p_warning, p_fraud)`` (winner probability).
    """
    probs = (float(p_legit), float(p_warning), float(p_fraud))
    cls = int(max(range(3), key=lambda k: probs[k]))
    label: PredictedLabel = ("legit", "warning", "fraud")[cls]
    confidence = max(probs)
    return cls, label, confidence


def class_from_softmax_triple_with_p_fraud_threshold(
    p_legit: float,
    p_warning: float,
    p_fraud: float,
    *,
    p_fraud_threshold: float,
) -> tuple[int, PredictedLabel, float]:
    """
    Production-style rule: assign **fraud** when ``p_fraud >= p_fraud_threshold``;
    otherwise assign **legit** vs **warning** by comparing only ``p_legit`` and ``p_warning``,
    breaking ties toward **legit** (consistent with argmax preferring lower index on ties).

    ``confidence`` is the probability of the reported class: ``p_fraud`` if fraud, else
    ``p_legit`` or ``p_warning`` for the chosen non-fraud label.
    """
    pl, pw, pf = float(p_legit), float(p_warning), float(p_fraud)
    t = float(p_fraud_threshold)
    if pf >= t:
        return CLASS_FRAUD, "fraud", pf
    if pl >= pw:
        return CLASS_LEGIT, "legit", pl
    return CLASS_WARNING, "warning", pw


def map_binary_to_risk(
    p_scam: float,
    heuristic_codes: list[str],
    *,
    warn_threshold: float,
    fraud_threshold: float,
) -> tuple[int, PredictedLabel, tuple[float, float, float], float]:
    """
    Derive 3-way class from scalar scam probability and optional heuristic codes.

    Precedence: fraud if p >= fraud_threshold; else warning if p >= warn_threshold
    or any heuristic matched; else legit.

    Per-class probabilities are one-hot for the chosen class. ``confidence`` is
    max(p_scam, 1 - p_scam) (certainty of the underlying binary head).
    """
    p = max(0.0, min(1.0, float(p_scam)))
    confidence = max(p, 1.0 - p)
    has_heuristic = len(heuristic_codes) > 0

    if p >= fraud_threshold:
        label: PredictedLabel = "fraud"
        cls = CLASS_FRAUD
        probs = (0.0, 0.0, 1.0)
    elif p >= warn_threshold or has_heuristic:
        label = "warning"
        cls = CLASS_WARNING
        probs = (0.0, 1.0, 0.0)
    else:
        label = "legit"
        cls = CLASS_LEGIT
        probs = (1.0, 0.0, 0.0)

    return cls, label, probs, confidence

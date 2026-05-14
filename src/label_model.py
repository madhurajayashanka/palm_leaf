"""
Snorkel-style generative label model for combining noisy labelling-function votes.

Background
----------
Given an L x M matrix of LF outputs (L = tokens, M = LFs) with values in
{-1 (abstain), 0 (O), 1 (STOP)}, we want to recover a probabilistic label
P(y_i = STOP) for every token without access to any ground truth.

We implement a simple but principled estimator:

1.  Per-LF accuracy is estimated against a high-precision anchor LF
    (`line_end`).  On tokens where the anchor abstains, we use majority-vote
    of the remaining LFs as a soft pseudo-target.

2.  Each LF's accuracy is converted to a log-odds weight
        w_j = log( a_j / (1 - a_j) )
    so that LFs more accurate than chance pull the posterior toward their
    vote and LFs at chance contribute zero weight.

3.  The posterior for token i is
        logit(P(y_i = STOP)) = Σ_j w_j · v_ij
    where v_ij ∈ {-1, 0, +1}.  Abstentions contribute zero.

4.  P(STOP) = sigmoid(logit).

This is mathematically equivalent to a Naive-Bayes ensemble of the LFs
under the assumption of conditional independence given the latent label,
which is the same assumption Ratner et al. (2017) make in Snorkel's
"label model".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from labeling_functions import LABEL_ABSTAIN, LABEL_O, LABEL_STOP


@dataclass
class LabelModelFit:
    lf_names: List[str]
    accuracies: List[float]
    weights: List[float]
    coverage: List[float] = field(default_factory=list)
    polarity: List[str] = field(default_factory=list)
    anchor: str = "line_end"


def _signed(votes: np.ndarray) -> np.ndarray:
    """Convert {-1, 0, 1} vote matrix to a "signed vote" with abstain=0.

        O      ->  -1
        STOP   ->  +1
        abstain ->  0
    """
    out = np.zeros_like(votes, dtype=np.float64)
    out[votes == LABEL_STOP] = 1.0
    out[votes == LABEL_O] = -1.0
    return out


def fit_label_model(
    votes: np.ndarray,
    lf_names: Sequence[str],
    anchor: str = "line_end",
    accuracy_floor: float = 0.55,
    accuracy_ceiling: float = 0.99,
) -> LabelModelFit:
    """Estimate per-LF accuracy and convert to log-odds weights.

    Parameters
    ----------
    votes : (L, M) int array of LF outputs.  -1 = abstain.
    lf_names : list of length M with LF identifiers (must contain `anchor`).
    anchor : LF assumed to be high-precision; used as pseudo-truth where it
             does not abstain. (Default: `line_end`.)
    accuracy_floor : Minimum accuracy to keep an LF (below = uninformative).
    accuracy_ceiling : Cap to prevent any one LF from dominating.
    """
    if anchor not in lf_names:
        raise ValueError(f"anchor LF {anchor!r} not in {lf_names}")

    anchor_idx = lf_names.index(anchor)
    L, M = votes.shape

    # ------------------------------------------------------------------
    # 1)  Build pseudo-targets:
    #     * where the anchor fires (line_end -> STOP) -> y_hat = 1
    #     * where the anchor abstains -> use majority of the non-anchor LFs
    #                                    counting abstain as zero
    # ------------------------------------------------------------------
    signed = _signed(votes)  # (L, M)
    non_anchor_idx = [j for j in range(M) if j != anchor_idx]
    majority = signed[:, non_anchor_idx].sum(axis=1)
    # > 0 -> STOP, < 0 -> O, == 0 -> O (default to O since most tokens are O)
    pseudo = np.where(majority > 0, 1, 0)

    anchor_fires = votes[:, anchor_idx] == LABEL_STOP
    pseudo = np.where(anchor_fires, 1, pseudo)

    # ------------------------------------------------------------------
    # 2)  Per-LF accuracy on non-abstain tokens vs. the pseudo-target.
    # ------------------------------------------------------------------
    accuracies: List[float] = []
    coverages: List[float] = []
    polarities: List[str] = []
    for j in range(M):
        non_abstain = votes[:, j] != LABEL_ABSTAIN
        cov = non_abstain.mean() if L > 0 else 0.0
        if non_abstain.sum() == 0:
            acc = 0.5
        else:
            lf_pred = (votes[non_abstain, j] == LABEL_STOP).astype(int)
            target = pseudo[non_abstain]
            acc = float((lf_pred == target).mean())
        acc = float(np.clip(acc, accuracy_floor, accuracy_ceiling))
        accuracies.append(acc)
        coverages.append(float(cov))
        # polarity: STOP-only / O-only / both
        unique_votes = set(np.unique(votes[non_abstain, j]).tolist()) if non_abstain.any() else set()
        if unique_votes == {LABEL_STOP}:
            polarities.append("STOP-only")
        elif unique_votes == {LABEL_O}:
            polarities.append("O-only")
        else:
            polarities.append("both")

    # ------------------------------------------------------------------
    # 3)  Log-odds weights.
    # ------------------------------------------------------------------
    weights = [math.log(a / (1.0 - a)) for a in accuracies]

    return LabelModelFit(
        lf_names=list(lf_names),
        accuracies=accuracies,
        weights=weights,
        coverage=coverages,
        polarity=polarities,
        anchor=anchor,
    )


def predict_proba(votes: np.ndarray, fit: LabelModelFit, prior_stop: float = None) -> np.ndarray:
    """Return P(y_i = STOP) for every row of `votes`.

    The prior log-odds correct for the marginal STOP frequency; if not
    provided we infer it from the anchor LF firing rate.
    """
    signed = _signed(votes)  # (L, M)
    w = np.asarray(fit.weights, dtype=np.float64)
    logit = signed @ w  # (L,)

    if prior_stop is None:
        anchor_idx = fit.lf_names.index(fit.anchor)
        prior_stop = float((votes[:, anchor_idx] == LABEL_STOP).mean())
        prior_stop = float(np.clip(prior_stop, 0.05, 0.5))

    prior_logit = math.log(prior_stop / (1.0 - prior_stop))
    logit = logit + prior_logit
    return 1.0 / (1.0 + np.exp(-logit))


def predict_label(votes: np.ndarray, fit: LabelModelFit, threshold: float = 0.5, prior_stop: float = None) -> np.ndarray:
    """Hard-label predictor for compatibility with downstream consumers."""
    proba = predict_proba(votes, fit, prior_stop=prior_stop)
    return (proba >= threshold).astype(int)


def summary(fit: LabelModelFit) -> str:
    """Pretty-printed report of estimated accuracies and weights."""
    lines = ["Label Model Fit", "=" * 60,
             f"{'LF':<14}{'Coverage':>10}{'Polarity':>12}{'Accuracy':>11}{'Weight':>11}"]
    for name, acc, w, cov, pol in zip(
        fit.lf_names, fit.accuracies, fit.weights, fit.coverage, fit.polarity
    ):
        lines.append(f"{name:<14}{cov:>10.3f}{pol:>12}{acc:>11.3f}{w:>11.3f}")
    lines.append("=" * 60)
    lines.append(f"anchor LF: {fit.anchor}")
    return "\n".join(lines)

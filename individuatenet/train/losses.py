"""Reward for training IndividuateNet:  U(T) - fluency penalty.

The generator maximises identity information while penalising mutable-attribute
leakage and preserving fluency:

    max_phi  I_hat(T;Y) - lambda * I_hat(T;A) - beta * D_fluency(q_phi).

`IndividuationReward` scores a batch of generated captions (as embeddings +
token log-probs) and returns the per-sample reward used by REINFORCE, plus its
components for logging. Torch imported at top (cluster-only module).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RewardWeights:
    lambda_attr: float = 1.0
    beta_fluency: float = 0.1
    temperature: float = 0.07


class IndividuationReward:
    def __init__(self, weights: RewardWeights):
        self.w = weights

    def identity_term(self, text_emb: torch.Tensor, id_anchor: torch.Tensor) -> torch.Tensor:
        """Per-sample InfoNCE-style identity alignment (proxy for I(T;Y))."""
        t = torch.nn.functional.normalize(text_emb, dim=-1)
        y = torch.nn.functional.normalize(id_anchor, dim=-1)
        logits = (t @ y.T) / self.w.temperature
        logp = torch.log_softmax(logits, dim=-1)
        return logp.diag()  # log p(correct identity | caption)

    def attribute_term(self, text_emb: torch.Tensor, attr_pred, attr_true) -> torch.Tensor:
        """Per-sample attribute-leakage penalty (proxy for I(T;A)).

        `attr_pred` is q(a|t) from a frozen CLUB net; higher predictability of
        mutable attributes from the caption -> larger penalty.
        """
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            attr_pred, attr_true, reduction="none"
        ).sum(dim=-1)
        # low BCE (attributes predictable) => high leakage => negative reward
        return -bce

    def fluency_term(self, base_logprob: torch.Tensor, policy_logprob: torch.Tensor):
        """KL-to-base surrogate: penalise drifting from the base LM."""
        return -(policy_logprob - base_logprob)

    def __call__(
        self,
        text_emb: torch.Tensor,
        id_anchor: torch.Tensor,
        attr_pred: torch.Tensor,
        attr_true: torch.Tensor,
        base_logprob: torch.Tensor,
        policy_logprob: torch.Tensor,
    ):
        i_ty = self.identity_term(text_emb, id_anchor)
        i_ta = self.attribute_term(text_emb, attr_pred, attr_true)
        flu = self.fluency_term(base_logprob, policy_logprob)
        # attribute_term (= -BCE) INCREASES with I(T;A): low BCE means the
        # attributes are predictable from the caption => high leakage => larger
        # i_ta. Subtract it so the generator is pushed to *reduce* leakage.
        reward = i_ty - self.w.lambda_attr * i_ta + self.w.beta_fluency * flu
        components = {
            "i_ty": i_ty.mean().item(),
            "i_ta": (-i_ta).mean().item(),
            "fluency": flu.mean().item(),
            "reward": reward.mean().item(),
        }
        return reward, components

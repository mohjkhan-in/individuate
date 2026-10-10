"""Parametric torch MI estimators (paper-grade backend, used on the cluster).

- `InfoNCECritic`  separable bilinear critic; `estimate_i_ty` trains it and
  returns the InfoNCE lower bound on I(T;Y).
- `CLUBEstimator`  variational q(a|t) (or q(a|t,y)); `estimate_i_ta` returns the
  CLUB upper bound on the (conditional) attribute-leakage term.

Torch is imported lazily inside the functions so the rest of the package (and
the offline numpy path) works without torch installed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TorchUConfig:
    temperature: float = 0.07
    critic_steps: int = 2000
    critic_lr: float = 1e-3
    hidden: int = 256
    device: str = "cpu"


def _build_infonce_critic(dim_t: int, dim_y: int, hidden: int):
    import torch.nn as nn

    class InfoNCECritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.g_t = nn.Sequential(
                nn.Linear(dim_t, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )
            self.g_y = nn.Sequential(
                nn.Linear(dim_y, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )

        def forward(self, t, y):
            ht = nn.functional.normalize(self.g_t(t), dim=-1)
            hy = nn.functional.normalize(self.g_y(y), dim=-1)
            return ht @ hy.T  # (B, B) similarity matrix

    return InfoNCECritic()


def estimate_i_ty(text_emb, id_anchor, cfg: TorchUConfig | None = None) -> float:
    """Train an InfoNCE critic and return the lower bound on I(T;Y) (nats).

    `id_anchor` is the paired identity representation (e.g. mean image embedding
    for each sample's identity), aligned row-wise with `text_emb`.
    """
    import torch

    cfg = cfg or TorchUConfig()
    device = torch.device(cfg.device)
    t = torch.as_tensor(text_emb, dtype=torch.float32, device=device)
    y = torch.as_tensor(id_anchor, dtype=torch.float32, device=device)

    critic = _build_infonce_critic(t.shape[1], y.shape[1], cfg.hidden).to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    targets = torch.arange(t.shape[0], device=device)

    critic.train()
    for _ in range(cfg.critic_steps):
        opt.zero_grad()
        logits = critic(t, y) / cfg.temperature
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        opt.step()

    critic.eval()
    with torch.no_grad():
        logits = critic(t, y) / cfg.temperature
        ce = torch.nn.functional.cross_entropy(logits, targets)
        bound = torch.log(torch.tensor(float(t.shape[0]))) - ce
    return float(bound.item())


def _build_club_net(dim_t: int, dim_a: int, hidden: int, dim_cond: int = 0):
    import torch.nn as nn

    class CLUBNet(nn.Module):
        """Predicts p_mean and log-variance of a given [t, cond]."""

        def __init__(self) -> None:
            super().__init__()
            din = dim_t + dim_cond
            self.mu = nn.Sequential(
                nn.Linear(din, hidden), nn.ReLU(), nn.Linear(hidden, dim_a)
            )
            self.logvar = nn.Sequential(
                nn.Linear(din, hidden), nn.ReLU(), nn.Linear(hidden, dim_a), nn.Tanh()
            )

        def forward(self, t):
            return self.mu(t), self.logvar(t)

    return CLUBNet()


def estimate_i_ta(
    text_emb,
    attr,
    cond=None,
    cfg: TorchUConfig | None = None,
) -> float:
    """Train a CLUB variational net and return the upper bound on I(T;A|cond).

    Pass `cond` (e.g. an identity embedding / one-hot) to obtain the conditional
    bound I(T;A|Y); leave it None for the marginal I(T;A).
    """
    import torch

    cfg = cfg or TorchUConfig()
    device = torch.device(cfg.device)
    t = torch.as_tensor(text_emb, dtype=torch.float32, device=device)
    a = torch.as_tensor(attr, dtype=torch.float32, device=device)
    dim_cond = 0
    if cond is not None:
        c = torch.as_tensor(cond, dtype=torch.float32, device=device)
        t = torch.cat([t, c], dim=1)
        dim_cond = c.shape[1]

    net = _build_club_net(t.shape[1] - dim_cond, a.shape[1], cfg.hidden, dim_cond).to(device)
    # note: net input dim is t.shape[1] (already includes cond)
    net.mu[0] = torch.nn.Linear(t.shape[1], cfg.hidden).to(device)
    net.logvar[0] = torch.nn.Linear(t.shape[1], cfg.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.critic_lr)

    def log_likelihood(mu, logvar, target):
        return -0.5 * (((target - mu) ** 2) / logvar.exp() + logvar).sum(dim=1)

    net.train()
    for _ in range(cfg.critic_steps):
        opt.zero_grad()
        mu, logvar = net(t)
        loss = -log_likelihood(mu, logvar, a).mean()  # maximise likelihood of q
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        mu, logvar = net(t)
        pos = log_likelihood(mu, logvar, a)
        perm = torch.randperm(a.shape[0], device=device)
        neg = log_likelihood(mu, logvar, a[perm])
        club = (pos - neg).mean()
    return float(torch.clamp(club, min=0.0).item())

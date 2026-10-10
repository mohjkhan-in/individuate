# IndividuateNet

Reference implementation for **"Learning to Individuate: Information-Maximising
Caption Generation for Zero-Shot Face Enrolment."**

The repo provides three things:

1. **`U` estimator** — the individuation functional
   `U(T) = I(T;Y) - I(T;A|Y)`, with an InfoNCE lower bound on `I(T;Y)` and a
   CLUB upper bound on the conditional attribute-leakage term. Two backends:
   a training-free **numpy** proxy (runs anywhere, offline) and a parametric
   **torch** version (paper-grade, for the cluster).
2. **IndividuateNet generator** — a prefix-tuned language model
   `q_phi(T | c)` trained to maximise `U` (identity information) while
   penalising mutable-attribute leakage and preserving fluency.
3. **Eval harness** — verification/identification metrics (EER, ROC-AUC with
   DeLong CIs, Rank-k), the zero-shot enrolment protocol (**E1**), and the
   training-free `U`-based outcome predictor (**E2**).

## Offline-safe design

Real runs need CLIP/FaRL backbones and the face datasets. To let you validate
the *harness* without any of that, every stage has a **synthetic regime**
(`data/synthetic.py`) with planted identity/attribute structure and a dial for
caption individuation. `run_e1.py --synthetic` and `run_e2.py --synthetic`
produce end-to-end numbers using **numpy only** (no torch, no downloads).

## Layout

```
individuatenet/
  data/        synthetic regimes, caption builders, real dataset loaders
  mi/          InfoNCE / CLUB estimators (numpy + torch) and the U functional
  models/      frozen encoders, IndividuateNet generator
  train/       generator training loop + losses
  eval/        metrics, enrolment protocol (E1), U-predictor (E2)
  utils/       config, seeding
scripts/       validate_estimators, estimate_u, run_e1, run_e2
tests/         synthetic MI + metrics sanity checks
configs/       default / celeba / vggface2
```

## Quick start (offline smoke, numpy only)

```bash
pip install numpy pyyaml scipy          # torch not required for --synthetic
python scripts/validate_estimators.py   # MI estimators on planted Gaussians
python scripts/run_e1.py --synthetic     # E1 main-table smoke
python scripts/run_e2.py --synthetic     # E2 U-predicts-outcome smoke
```

## Real runs (cluster)

```bash
pip install -e .                        # pulls torch, open_clip_torch, transformers
python scripts/estimate_u.py  --config configs/celeba.yaml
python scripts/run_e1.py      --config configs/celeba.yaml
python scripts/train_generator.py --config configs/celeba.yaml  # IndividuateNet
```

Set `INET_OFFLINE=1` to force local-only backbone/tokenizer loading.

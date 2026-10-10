"""End-to-end offline smoke: the E1/E2 harness runs and shows the expected trend."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from individuatenet.data.synthetic import SOURCE_INDIVIDUATION, make_regime, split_context_probe
from individuatenet.eval import zero_shot_enrolment
from individuatenet.eval.predict_u import fit_u_predictor
from individuatenet.mi import estimate_u


def test_enrolment_improves_with_individuation():
    reg = make_regime(n_ids=100, imgs_per_id=20, seed=0)
    _, probe_idx = split_context_probe(reg, test_frac=0.5, seed=0)
    aucs = {}
    for source, p in SOURCE_INDIVIDUATION.items():
        enrol = reg.caption_embeddings(p, per_image=False)
        res = zero_shot_enrolment(
            enrol, reg.ids, reg.image_emb[probe_idx], reg.id_labels[probe_idx], seed=0
        )
        aucs[source] = res.auc
    assert aucs["generated"] > aucs["attribute"]


def test_u_predicts_auc_positive_spearman():
    us, aucs, nimg = [], [], []
    for imgs in (5, 10, 20):
        for p in (0.0, 0.5, 1.0):
            reg = make_regime(n_ids=80, imgs_per_id=imgs, seed=0)
            text = reg.caption_embeddings(p, per_image=True)
            us.append(estimate_u(text, reg.image_emb, reg.id_labels, reg.attr,
                                 backend="numpy", seed=0).u)
            _, probe_idx = split_context_probe(reg, seed=0)
            enrol = reg.caption_embeddings(p, per_image=False)
            res = zero_shot_enrolment(enrol, reg.ids, reg.image_emb[probe_idx],
                                      reg.id_labels[probe_idx], seed=0)
            aucs.append(res.auc)
            nimg.append(imgs)
    fit = fit_u_predictor(us, aucs, nimg)
    assert fit.spearman_u_auc > 0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all e1/e2 smoke tests passed")

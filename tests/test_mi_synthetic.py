"""Sanity checks for the numpy MI estimators and the U functional."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from individuatenet.data.synthetic import SOURCE_INDIVIDUATION, make_regime
from individuatenet.mi import estimate_u
from individuatenet.mi.np_estimators import club_upper_bound, infonce_lower_bound


def test_infonce_positive_and_bounded():
    reg = make_regime(n_ids=60, imgs_per_id=15, seed=0)
    lb = infonce_lower_bound(reg.image_emb, reg.image_emb, reg.id_labels)
    # identity is recoverable from image embeddings: bound well above 0 and
    # below the log(num_ids) ceiling.
    assert lb > 0.5
    assert lb <= np.log(len(reg.ids)) + 1e-6


def test_club_nonnegative():
    reg = make_regime(n_ids=50, imgs_per_id=12, seed=1)
    ub = club_upper_bound(reg.image_emb, reg.attr, reg.id_labels)
    assert ub >= 0.0


def test_u_increases_with_individuation():
    reg = make_regime(n_ids=80, imgs_per_id=20, seed=2)
    us = {}
    for source, p in SOURCE_INDIVIDUATION.items():
        text = reg.caption_embeddings(p, per_image=True)
        us[source] = estimate_u(text, reg.image_emb, reg.id_labels, reg.attr,
                                backend="numpy", seed=2).u
    # individuating captions should carry more identity info than attribute ones
    assert us["generated"] > us["attribute"]
    assert us["rich"] > us["attribute"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all MI tests passed")

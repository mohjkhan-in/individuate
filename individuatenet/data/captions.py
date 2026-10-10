"""Caption builders for real datasets (attribute / discriminative / rich).

These turn a per-image attribute dict into natural-language descriptions at
three individuation levels used as baselines in E1:

- ``attribute``      : enumerates mutable attributes (at chance on VGGFace2).
- ``discriminative`` : keeps only the attributes that distinguish this identity
                       from its nearest neighbours (mild individuation).
- ``rich``           : VLM-prompted individuating description (strong baseline).

The learned ``generated`` source comes from IndividuateNet, not this module.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

# A compact, ordered vocabulary compatible with CelebA / MAAD-Face style flags.
_ATTR_PHRASES = {
    "Male": "a man",
    "Young": "a young person",
    "Smiling": "smiling",
    "Eyeglasses": "wearing glasses",
    "Wearing_Hat": "wearing a hat",
    "Bald": "bald",
    "Bangs": "with bangs",
    "Black_Hair": "with black hair",
    "Blond_Hair": "with blond hair",
    "Brown_Hair": "with brown hair",
    "Gray_Hair": "with gray hair",
    "Straight_Hair": "with straight hair",
    "Wavy_Hair": "with wavy hair",
    "Mustache": "with a mustache",
    "No_Beard": "clean-shaven",
    "Goatee": "with a goatee",
    "Pale_Skin": "with pale skin",
    "Chubby": "with a round face",
    "Narrow_Eyes": "with narrow eyes",
    "Big_Nose": "with a large nose",
    "Big_Lips": "with full lips",
    "High_Cheekbones": "with high cheekbones",
    "Arched_Eyebrows": "with arched eyebrows",
    "Heavy_Makeup": "wearing heavy makeup",
}


def _active(attr: Dict[str, int]) -> List[str]:
    return [k for k, v in attr.items() if int(v) > 0 and k in _ATTR_PHRASES]


def attribute_caption(attr: Dict[str, int]) -> str:
    phrases = [_ATTR_PHRASES[k] for k in _active(attr)]
    if not phrases:
        return "a person."
    head, *tail = phrases
    body = head + ("" if head.startswith(("a ", "an ")) else " person")
    if tail:
        body += ", " + ", ".join(tail)
    return body + "."


def discriminative_caption(
    attr: Dict[str, int], distinctive_keys: Sequence[str]
) -> str:
    """Describe only the identity-distinctive attributes (mild individuation)."""
    keep = {k: attr.get(k, 0) for k in distinctive_keys}
    phrases = [_ATTR_PHRASES[k] for k in _active(keep)]
    if not phrases:
        return attribute_caption(attr)
    return "notably, " + ", ".join(phrases) + "."


def rich_caption(attr: Dict[str, int], free_text: str | None = None) -> str:
    """VLM-prompted individuating description; falls back to attribute text."""
    if free_text:
        return free_text.strip().rstrip(".") + "."
    return attribute_caption(attr)


def build_captions(
    attr: Dict[str, int],
    source: str = "attribute",
    distinctive_keys: Sequence[str] | None = None,
    free_text: str | None = None,
) -> str:
    if source == "attribute":
        return attribute_caption(attr)
    if source == "discriminative":
        return discriminative_caption(attr, distinctive_keys or [])
    if source == "rich":
        return rich_caption(attr, free_text)
    raise ValueError(f"unknown caption source {source!r}")

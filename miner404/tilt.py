"""The tilt that makes the level front view look most like the photo. No model call: the outline of each tilt against the photo's."""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
from PIL import Image

from . import measure_own

TILT_LINE = "if (flat.y < 0.5 * Math.max(flat.x, flat.z)) root.rotation.x = 15 * Math.PI / 180;"
TILTS = (0, 8, 15, 22)
MIN_GAIN = 0.001  # outline overlap over no tilt; below this the module keeps its own tilt
USED = 22


def _square(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    box = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = box.shape
    side = max(height, width)
    out = np.zeros((side, side), dtype=bool)
    top, left = (side - height) // 2, (side - width) // 2
    out[top:top + height, left:left + width] = box
    return np.asarray(Image.fromarray(out.astype(np.uint8) * 255).resize((128, 128))) > 127


def tilted(source: str, angle: int) -> str:
    return source.replace(TILT_LINE, f"root.rotation.x = {angle} * Math.PI / 180;")


def tilted_if_flat(source: str, angle: int) -> str:
    """The tilt for flat objects only: anything taller stays level, as the fixed front views read it better."""
    return source.replace(TILT_LINE, TILT_LINE.replace("= 15 *", f"= {angle} *"))


def best_tilt(image: bytes, source: str, render: Callable[[str], bytes]) -> int | None:
    """The angle to use, or None to leave the module alone: no usable photo outline, or no clear gain."""
    if source.count(TILT_LINE) != 1:
        return None
    photo = measure_own.photo_mask(image)
    if photo is None:
        return None
    target = _square(np.asarray(photo[0], dtype=bool))
    scores: dict[int, float] = {}
    for angle in TILTS:
        rgb = Image.open(io.BytesIO(render(tilted(source, angle)))).convert("RGB").resize((measure_own.S, measure_own.S))
        mask = measure_own.render_mask(np.asarray(rgb).astype(np.float32))
        if mask.sum() <= 20:
            return None
        shown = _square(mask)
        scores[angle] = float((target & shown).sum() / max((target | shown).sum(), 1))
    best = max(TILTS, key=lambda angle: (scores[angle], -angle))
    return best if best == USED and scores[best] - scores[0] >= MIN_GAIN else None

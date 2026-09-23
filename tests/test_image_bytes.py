"""``laya.vlm`` states may carry encoded image bytes (the autoresearch data pool keeps images in memory that way)."""
import io

import numpy as np
from PIL import Image

from laya.vlm import split_state


def _png(color, size=(20, 10)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_split_state_decodes_bytes_like_a_path(tmp_path):
    raw = _png((10, 200, 30))
    path = tmp_path / "x.png"
    path.write_bytes(raw)
    from_bytes, text = split_state({"image": raw, "note": "hi"})
    from_path, _ = split_state({"image": str(path), "note": "hi"})
    assert np.array_equal(np.asarray(from_bytes[0]), np.asarray(from_path[0])) and from_bytes[0].size == (20, 10)
    assert "hi" in text
    two, _ = split_state({"images": [raw, bytearray(_png((0, 0, 0)))]})
    assert [im.getpixel((0, 0)) for im in two] == [(10, 200, 30), (0, 0, 0)]

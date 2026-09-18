"""Bounded encoded-image cache; original and decoded images are never retained."""

import io

from fastapi import HTTPException
from PIL import Image


def image_bytes(store, dicom_id, size, quality=90, format="jpeg"):
    image = store.images.get(dicom_id)
    if image is None:
        raise HTTPException(404, "Unknown image")
    path = image.image_path.resolve()
    if not path.is_relative_to(store.cxr_root.resolve()) or not path.is_file():
        raise HTTPException(404, "Image is unavailable")
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, size, quality, format)
    cache = store.image_memory
    payload = cache.get(key)
    if payload is not None:
        return payload
    # Fixed lock stripes coalesce repeated requests without an unbounded lock map.
    with cache.key_locks[hash(key) % len(cache.key_locks)]:
        payload = cache.get(key)
        if payload is not None:
            return payload
        with cache.render_slots:
            with Image.open(path) as im:
                im.thumbnail((size, size))
                with io.BytesIO() as buf, im.convert("RGB") as rgb:
                    rgb.save(buf, format.upper(), quality=quality)
                    payload = buf.getvalue()
            cache.put(key, payload)
            return payload

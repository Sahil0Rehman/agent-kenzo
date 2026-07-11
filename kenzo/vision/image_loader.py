"""Image loading/validation/encoding for the Phase 8 vision pipeline.

Accepts a file path OR raw bytes, validates it's one of the supported
formats (png/jpg/jpeg/webp -- per the spec's explicit list), and produces a
`LoadedImage` ready to embed in a multimodal `litellm.completion()` message
as a base64 data URL. Format is detected from the file extension (path
input) or magic bytes (raw-bytes input, e.g. an image the VS Code extension
or `kenzo server` received over the wire with no filename attached) --
NEVER trusted from a caller-supplied label alone, since a wrong MIME type
silently sent to a vision model produces confusing, hard-to-debug failures
rather than a clear error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

SUPPORTED_FORMATS: frozenset[str] = frozenset({"png", "jpg", "jpeg", "webp"})

# Generous but bounded -- large enough for any real screenshot, small enough
# to reject an accidental multi-hundred-MB upload before it reaches a model
# API (which would either reject it anyway or incur a large, wasted cost).
MAX_IMAGE_BYTES = 20 * 1024 * 1024

_FORMAT_TO_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}

_EXTENSION_TO_FORMAT = {
    ".png": "png",
    ".jpg": "jpg",
    ".jpeg": "jpeg",
    ".webp": "webp",
}


class UnsupportedImageError(ValueError):
    """Raised when an image's format can't be determined or isn't supported."""


@dataclass
class LoadedImage:
    data: bytes
    format: str  # one of SUPPORTED_FORMATS
    mime_type: str
    source: str  # the path, or "<bytes>" for in-memory input -- for logging/errors only


def _sniff_format(data: bytes) -> str | None:
    """Best-effort magic-byte format detection. Returns None if unrecognized."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def load_image(image: Union[str, Path, bytes], *, filename_hint: str = "") -> LoadedImage:
    """Load and validate an image from a file path or raw bytes.

    Raises `UnsupportedImageError` for a missing file, an unreadable path, an
    oversized image, or a format that isn't png/jpg/jpeg/webp (detected from
    the extension for a path, magic bytes for raw bytes -- `filename_hint`,
    if given alongside raw bytes, is used ONLY as a fallback when magic-byte
    sniffing is inconclusive, never trusted over it).
    """
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise UnsupportedImageError(f"image file not found: {path}")
        if not path.is_file():
            raise UnsupportedImageError(f"'{path}' is not a file")

        extension_format = _EXTENSION_TO_FORMAT.get(path.suffix.lower())
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise UnsupportedImageError(f"could not read '{path}': {exc}") from exc

        sniffed_format = _sniff_format(data)
        # Magic bytes win when they disagree with the extension (a
        # mislabeled/renamed file) -- but a sniff failure still falls back to
        # a valid extension rather than rejecting a genuinely fine image
        # this sniffer just doesn't recognize the header shape for.
        fmt = sniffed_format or extension_format
        source = str(path)
    else:
        data = image
        fmt = _sniff_format(data) or _EXTENSION_TO_FORMAT.get(Path(filename_hint).suffix.lower() if filename_hint else "")
        source = filename_hint or "<bytes>"

    if not data:
        raise UnsupportedImageError(f"image is empty: {source}")
    if len(data) > MAX_IMAGE_BYTES:
        raise UnsupportedImageError(
            f"image is too large ({len(data)} bytes, max {MAX_IMAGE_BYTES}): {source}"
        )
    if fmt is None or fmt not in SUPPORTED_FORMATS:
        raise UnsupportedImageError(
            f"unsupported or undetectable image format for '{source}' -- supported: "
            f"{', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    return LoadedImage(data=data, format=fmt, mime_type=_FORMAT_TO_MIME[fmt], source=source)


def to_data_url(loaded: LoadedImage) -> str:
    """Render as a base64 data URL suitable for a multimodal `litellm
    .completion()` message's `image_url.url` field."""
    import base64

    encoded = base64.b64encode(loaded.data).decode("ascii")
    return f"data:{loaded.mime_type};base64,{encoded}"

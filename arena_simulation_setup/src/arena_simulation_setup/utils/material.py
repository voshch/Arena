import re
from collections.abc import Iterator
from pathlib import Path

import colourings
from PIL import Image


def Color(color: object) -> colourings.Color:
    """Convert a color representation to a Colourings Color object.

    Args:
        color (typing.Any): Color representation (e.g., hex string, RGB tuple).

    Returns:
        colourings.Color: Corresponding Colourings Color object.
    """
    try:
        return colourings.Color(color)
    except Exception as e:
        exc = e

    try:
        t, c = color.split('(', 1)
        return colourings.Color(**{t: tuple(map(float, c.rstrip(')').split(',')))})  # type: ignore
    except BaseException:
        raise exc from None


class ImgUtil:
    @classmethod
    def tint(cls, img: Path | Image.Image, tint: object) -> Image.Image:
        """Apply a tint to an image.

        Args:
            img (Path | Image.Image): Image or path to image.
            tint (typing.Any): Tint color.

        Returns:
            Image.Image: Tinted image, RGB unless the source carries a non-uniform alpha channel.
        """

        if isinstance(img, Path):
            img = Image.open(img)
        if img.mode in ("P", "PA"):
            img = img.convert("RGBA")

        tint_color = Color(tint).get_rgba()

        strength = tint_color[3]
        orig_alpha = img.getchannel("A") if "A" in img.getbands() else None
        base_rgb = img.convert("RGB")
        overlay_rgb = Image.new(
            "RGB",
            img.size,
            (
                int(tint_color[0] * 255),
                int(tint_color[1] * 255),
                int(tint_color[2] * 255),
            ),
        )
        blended_rgb = Image.blend(base_rgb, overlay_rgb, strength)

        if orig_alpha is None or orig_alpha.getextrema() == (255, 255):
            return blended_rgb
        return Image.merge("RGBA", (*blended_rgb.split(), orig_alpha))


class MdlUtil:
    """.mdl material helper"""

    def __init__(self, path: Path):
        self.path = path

    def _texture_paths(self, slot: str) -> Iterator[Path]:
        """Yields paths to texture files bound to the given OmniPBR slot (e.g. 'diffuse_texture')."""
        pattern = re.compile(rf'{slot}:\s*texture_2d\("([^"]+)"')
        with open(self.path) as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    yield self.path.parent / Path(match.group(1))

    def texture(self, slot: str) -> Path | None:
        """First texture bound to the given slot, or None if the slot is empty/absent."""
        return next(self._texture_paths(slot), None)

    @property
    def diffuse_texture_paths(self) -> Iterator[Path]:
        """Yields paths to diffuse texture files referenced by the .mdl file.

        Yields:
            Path: Path to a texture file.
        """
        return self._texture_paths('diffuse_texture')

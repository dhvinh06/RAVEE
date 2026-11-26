"""Aspect-preserving preview sizing, independent of the Tk event loop."""

from PIL import Image


def fit_preview_size(source_width, source_height, available_width, available_height):
    """Use the largest whole-frame size that fits, including upscaling.

    Letterboxing is only the remainder required by the two aspect ratios.
    Unlike thumbnail(), a low-resolution video can fill a large desktop panel.
    """
    if source_width <= 0 or source_height <= 0:
        raise ValueError('Video dimensions must be positive')
    width, height = max(1, int(available_width)), max(1, int(available_height))
    scale = min(width / source_width, height / source_height)
    return (max(1, min(width, round(source_width * scale))),
            max(1, min(height, round(source_height * scale))))


def resize_preview_image(image, available_width, available_height):
    size = fit_preview_size(*image.size, available_width, available_height)
    return image.resize(size, Image.Resampling.LANCZOS)

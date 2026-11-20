"""Calibration (offset, dark, flat), registration and stacking of images with NumPy/SciPy."""

import logging
from pathlib import Path
from typing import Iterable, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import fft as scipy_fft
from scipy import ndimage

from image_io import read_image, read_image_binned

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ImageArray: TypeAlias = NDArray[np.float32]
MasterFrame: TypeAlias = ArrayLike | str | Path
CombineMode: TypeAlias = Literal["sum", "mean"]
Shift: TypeAlias = tuple[float, float]  # (décalage vertical, décalage horizontal) en pixels

# Rapport signal/bruit du pic de corrélation de phase, (pic - médiane) / écart-type, en
# dessous duquel le recalage est jugé douteux. Mesuré : 5 à 8 pour des images sans rapport
# (quelle que soit leur taille), 12 à 45 pour de vraies paires, même très bruitées.
REGISTRATION_MIN_SNR = 10.0


class Image:
    """Load, calibrate, register and stack supported images using NumPy and SciPy.

    The processor retains the original frames and their current combined or
    reduced image so later processing steps can reuse them.
    """

    def __init__(self) -> None:
        """Initialize the processor with no loaded image data."""
        self.paths: list[str] = []
        self.frames: list[ImageArray] = []
        self.raw_data: ImageArray | None = None
        self.data: ImageArray | None = None
        self.applied_corrections: list[str] = []  # masters réellement appliqués
        self.registration_shifts: list[Shift] = []  # décalage de chaque image vs la première
        self.binning: int = 1  # 2 si les pixels ont été sommés 2×2 à la lecture (JPEG, couleur)

    def load_stack(
        self,
        paths: Iterable[str | Path],
        offset: MasterFrame | None = None,
        dark: MasterFrame | None = None,
        flat: MasterFrame | None = None,
        register: bool = True,
    ) -> ImageArray:
        """Load images, correct each one, align them on the first and sum them.

        Each frame is corrected (offset, dark, flat) *before* the sum, then
        registered on the first frame with sub-pixel phase correlation, so the
        corrections are applied once per frame and not once to the sum.
        Frames are processed one at a time to limit memory use.

        Args:
            paths: Paths to supported raster or FITS images; the first is the
                registration reference.
            offset: Optional master offset (2D array or image path).
            dark: Optional master dark, already offset-subtracted.
            flat: Optional master flat, already offset/dark-subtracted.
            register: Align frames on the first one before summing.

        Returns:
            The summed, calibrated image as a two-dimensional float32 array.

        Raises:
            ValueError: If no path is given or the frames differ in shape.
        """
        paths = [str(path) for path in paths]
        if not paths:
            raise ValueError("Select at least one image.")
        logger.info(
            "Chargement de %d image(s) : %s", len(paths), ", ".join(Path(p).name for p in paths)
        )

        first, self.binning = read_image_binned(paths[0])
        if self.binning > 1:
            logger.info("Pixels sommés %d×%d à la lecture (JPEG/couleur)", self.binning, self.binning)
        masters = self._usable_masters(first.shape, offset=offset, dark=dark, flat=flat)
        self.applied_corrections = [name for name, value in masters.items() if value is not None]
        skipped = [name for name, value in masters.items() if value is None]
        logger.info(
            "Corrections : %s ; non appliquées : %s",
            ", ".join(self.applied_corrections) or "aucune",
            ", ".join(skipped) or "aucune",
        )
        flat_fit = _normalized_flat(masters["flat"]) if masters["flat"] is not None else None

        total = np.zeros(first.shape, dtype=np.float32)
        reference: ImageArray | None = None
        self.registration_shifts = [(0.0, 0.0)]
        for index, path in enumerate(paths):
            frame = first if index == 0 else read_image(path)
            if frame.shape != first.shape:
                raise ValueError(
                    f"{Path(path).name} : {frame.shape[1]} × {frame.shape[0]} px, "
                    f"attendu {first.shape[1]} × {first.shape[0]} (comme la première image)."
                )
            frame = _sanitize(calibrate_frame(frame, masters["offset"], masters["dark"], flat_fit))
            logger.info(
                "Image %d/%d %s : médiane=%.1f max=%.1f après correction",
                index + 1, len(paths), Path(path).name, float(np.median(frame)), float(frame.max()),
            )
            if index == 0:
                reference = frame
            elif register:
                dy, dx, snr = _phase_correlation_shift(reference, frame)
                logger.info(
                    "Recalage de %s sur la première image : dy=%+.2f dx=%+.2f px (pic à %.0f σ)",
                    Path(path).name, dy, dx, snr,
                )
                if snr < REGISTRATION_MIN_SNR:
                    logger.warning(
                        "Recalage de %s peu fiable (pic à %.1f σ < %.0f σ) : images trop différentes ?",
                        Path(path).name, snr, REGISTRATION_MIN_SNR,
                    )
                self.registration_shifts.append((dy, dx))
                frame = _shift_image(frame, (dy, dx))
            else:
                self.registration_shifts.append((0.0, 0.0))
            total += frame

        self.paths = paths
        self.frames = []
        self.raw_data = total
        self.data = total.copy()
        logger.info(
            "Somme de %d image(s) %s : médiane=%.1f max=%.1f",
            len(paths), "recalées" if register and len(paths) > 1 else "",
            float(np.median(total)), float(total.max()),
        )
        return self.data.copy()

    def _usable_masters(
        self, shape: tuple[int, int], **masters: MasterFrame | None
    ) -> dict[str, ImageArray | None]:
        """Validate masters against the frame shape; unusable ones become ``None`` with a warning."""
        usable: dict[str, ImageArray | None] = {}
        for name, master in masters.items():
            if master is None:
                usable[name] = None
                continue
            array = read_image(master) if isinstance(master, (str, Path)) else np.asarray(master, np.float32)
            if array.ndim != 2 or array.shape != shape:
                logger.warning(
                    "Master %s ignoré : %s au lieu de %d × %d "
                    "(format différent ? JPEG/TIFF sont binnés 2×2 à la lecture)",
                    name, "dimensions " + " × ".join(str(n) for n in reversed(array.shape)),
                    shape[1], shape[0],
                )
                usable[name] = None
            else:
                usable[name] = array
        return usable

    def load(
        self, paths: Iterable[str | Path], combine: CombineMode = "sum"
    ) -> ImageArray:
        """Load equal-sized images and combine them by sum or mean (no calibration).

        Args:
            paths: Paths to supported raster or FITS images.
            combine: Combination method applied to the loaded frames.

        Returns:
            The combined image as a two-dimensional float32 array.

        Raises:
            ValueError: If no paths are provided, the method is invalid, or the
                source images have different dimensions.
        """
        paths = [str(path) for path in paths]
        if not paths:
            raise ValueError("Select at least one image.")
        if combine not in ("sum", "mean"):
            raise ValueError("combine must be 'sum' or 'mean'.")

        frames = [read_image(path) for path in paths]
        shape = frames[0].shape
        if any(frame.shape != shape for frame in frames[1:]):
            raise ValueError("All image frames must have identical dimensions.")

        combined = np.zeros(shape, dtype=np.float32)
        for frame in frames:
            combined += frame
        if combine == "mean":
            combined /= len(frames)

        self.paths = paths
        self.frames = frames
        self.raw_data = combined
        self.data = combined.copy()
        logger.info("Chargé %d image(s), forme=%s", len(frames), shape)
        return combined.copy()

    def reduce(
        self,
        offset: MasterFrame | None = None,
        dark: MasterFrame | None = None,
        flat: MasterFrame | None = None,
    ) -> ImageArray:
        """Apply optional offset and dark subtraction and flat-field correction.

        This works on the combined image: for several frames, prefer
        :meth:`load_stack`, which corrects each frame before summing.

        Args:
            offset: Optional master offset image or 2D pixel array.
            dark: Optional master dark image or 2D pixel array.
            flat: Optional master flat image or 2D pixel array.

        Returns:
            The reduced image as a two-dimensional float32 array.

        Raises:
            RuntimeError: If no source image has been loaded.
            ValueError: If a master image has invalid dimensions or flat values.
        """
        if self.raw_data is None:
            raise RuntimeError("Load at least one image before reducing.")

        arrays = {
            name: None if master is None else self._master_array(master, name)
            for name, master in (("offset", offset), ("dark", dark), ("flat", flat))
        }
        flat_fit = _normalized_flat(arrays["flat"]) if arrays["flat"] is not None else None
        reduced = calibrate_frame(self.raw_data, arrays["offset"], arrays["dark"], flat_fit)
        self.data = reduced
        logger.info("Réduction de l'image terminée")
        return reduced

    def register(self, reference_index: int = 0) -> ImageArray:
        """Align loaded frames to a reference and sum the registered images.

        Args:
            reference_index: Index of the frame used as the alignment reference.

        Returns:
            The registered sum as a two-dimensional float32 array.

        Raises:
            RuntimeError: If no image frames have been loaded.
            IndexError: If the reference index is outside the loaded frame list.
            ValueError: If any frame dimensions differ from the reference.
        """
        if not self.frames:
            raise RuntimeError("Load at least one image before registration.")
        if not 0 <= reference_index < len(self.frames):
            raise IndexError("reference_index is outside the loaded frame list.")

        reference = self.frames[reference_index]
        registered = []
        for index, frame in enumerate(self.frames):
            if frame.shape != reference.shape:
                raise ValueError("All image frames must have identical dimensions.")
            if index == reference_index:
                registered.append(frame.copy())
                continue
            dy, dx, snr = _phase_correlation_shift(reference, frame)
            logger.info("Recalage de l'image %d : dy=%+.2f dx=%+.2f px (pic à %.0f σ)", index, dy, dx, snr)
            registered.append(_shift_image(frame, (dy, dx)))

        self.frames = registered
        result = np.sum(registered, axis=0, dtype=np.float32)
        self.raw_data = result
        self.data = result.copy()
        return result.copy()

    def _master_array(self, master: MasterFrame, name: str) -> ImageArray:
        """Load or validate a calibration frame against the source dimensions.

        Args:
            master: Calibration image path or two-dimensional pixel array.
            name: Calibration type used in error messages.

        Returns:
            A validated two-dimensional float32 calibration array.
        """
        if isinstance(master, (str, Path)):
            array = read_image(master)
        else:
            array = np.asarray(master, dtype=np.float32)
        if array.ndim != 2 or not array.size:
            raise ValueError(f"The {name} frame must be a non-empty 2D array.")
        if array.shape != self.raw_data.shape:
            raise ValueError(
                f"The {name} frame has shape {array.shape}; expected {self.raw_data.shape}."
            )
        return array


def _normalized_flat(flat: ImageArray) -> tuple[ImageArray, NDArray[np.bool_]]:
    """Flat divided by its median, and the mask of its usable (finite, non-zero) pixels.

    Raises:
        ValueError: If the flat has no finite non-zero pixel.
    """
    valid = np.isfinite(flat) & (flat != 0)
    if not np.any(valid):
        raise ValueError("Flat field contains no finite non-zero pixels.")
    normalized = np.zeros_like(flat, dtype=np.float32)
    normalized[valid] = flat[valid] / np.median(flat[valid])
    return normalized, valid


def calibrate_frame(
    frame: ImageArray,
    offset: ImageArray | None = None,
    dark: ImageArray | None = None,
    flat: tuple[ImageArray, NDArray[np.bool_]] | None = None,
) -> ImageArray:
    """Return ``(frame - offset - dark) / flat``; pixels with an unusable flat become 0.

    Args:
        frame: Raw 2D image (not modified).
        offset: Master offset, same shape as ``frame``.
        dark: Master dark, same shape, already offset-subtracted.
        flat: Output of :func:`_normalized_flat` for the master flat.
    """
    corrected = frame.astype(np.float32, copy=True)
    if offset is not None:
        corrected -= offset
    if dark is not None:
        corrected -= dark
    if flat is not None:
        normalized, valid = flat
        result = np.zeros_like(corrected)
        np.divide(corrected, normalized, out=result, where=valid)
        corrected = result
    return corrected


def _sanitize(frame: ImageArray) -> ImageArray:
    """Replace NaN/inf by 0 (they would corrupt the FFT and the sum) and warn once."""
    bad = ~np.isfinite(frame)
    count = int(np.count_nonzero(bad))
    if count:
        logger.warning("%d pixel(s) non finis remplacés par 0", count)
        frame = np.where(bad, np.float32(0), frame)
    return frame


def _phase_correlation_shift(
    reference: ImageArray, moving: ImageArray
) -> tuple[float, float, float]:
    """Estimate the translation that aligns ``moving`` on ``reference`` (sub-pixel).

    The mean level is removed and a Hann window applied, so the image borders and
    the sky background do not dominate the correlation.

    Args:
        reference: Reference image.
        moving: Image whose translation is measured.

    Returns:
        ``(dy, dx, snr)``: the vertical and horizontal shifts, in pixels, to apply
        to ``moving``, and the signal-to-noise ratio of the correlation peak.
    """
    window = np.outer(np.hanning(reference.shape[0]), np.hanning(reference.shape[1])).astype(np.float32)

    def prepare(image: ImageArray) -> ImageArray:
        image = np.nan_to_num(image.astype(np.float32, copy=False))
        return (image - np.median(image)) * window

    cross_power = scipy_fft.rfft2(prepare(reference)) * np.conj(scipy_fft.rfft2(prepare(moving)))
    magnitude = np.abs(cross_power)
    cross_power = np.divide(
        cross_power, magnitude,
        out=np.zeros_like(cross_power), where=magnitude > np.finfo(np.float32).eps,
    )
    correlation = scipy_fft.irfft2(cross_power, s=reference.shape)
    peak = np.array(np.unravel_index(np.argmax(correlation), correlation.shape))
    shifts = peak.astype(np.float64)

    for axis, axis_size in enumerate(reference.shape):
        if shifts[axis] > axis_size // 2:
            shifts[axis] -= axis_size
        before_index, after_index = peak.copy(), peak.copy()
        before_index[axis] = (peak[axis] - 1) % axis_size
        after_index[axis] = (peak[axis] + 1) % axis_size
        before = correlation[tuple(before_index)]
        center = correlation[tuple(peak)]
        after = correlation[tuple(after_index)]
        denominator = before - 2 * center + after
        if denominator != 0:
            shifts[axis] += 0.5 * (before - after) / denominator

    snr = (correlation[tuple(peak)] - np.median(correlation)) / max(float(correlation.std()), 1e-30)
    return float(shifts[0]), float(shifts[1]), float(snr)


def _shift_image(image: ImageArray, shift: Shift) -> ImageArray:
    """Translate an image by ``(dy, dx)`` pixels (bilinear interpolation, zero fill)."""
    return ndimage.shift(image, shift, order=1, mode="constant", cval=0.0)


def _bilinear_sample(
    image: ImageArray,
    source_x: NDArray[np.float32],
    source_y: NDArray[np.float32],
) -> ImageArray:
    """Sample an image at broadcastable coordinates, filling outside with 0.

    Args:
        image: Two-dimensional source image.
        source_x: Horizontal source coordinates, broadcastable with ``source_y``.
        source_y: Vertical source coordinates, broadcastable with ``source_x``.

    Returns:
        Float32 pixel values sampled at the requested coordinates.
    """
    height, width = image.shape
    x0_float = np.floor(source_x)
    y0_float = np.floor(source_y)
    x0 = x0_float.astype(np.int32)
    y0 = y0_float.astype(np.int32)
    dx = source_x - x0_float
    dy = source_y - y0_float
    result = np.zeros(
        np.broadcast_shapes(source_x.shape, source_y.shape), dtype=np.float32
    )

    for y_offset, y_weight in ((0, 1 - dy), (1, dy)):
        sample_y = y0 + y_offset
        valid_y = (sample_y >= 0) & (sample_y < height)
        for x_offset, x_weight in ((0, 1 - dx), (1, dx)):
            sample_x = x0 + x_offset
            valid = valid_y & (sample_x >= 0) & (sample_x < width)
            clipped_y = np.clip(sample_y, 0, height - 1)
            clipped_x = np.clip(sample_x, 0, width - 1)
            result += (
                image[clipped_y, clipped_x]
                * y_weight
                * x_weight
                * valid
            )
    return result

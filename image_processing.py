"""Image calibration and registration using NumPy."""

import logging
from pathlib import Path
from typing import Iterable, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from image_io import read_image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ImageArray: TypeAlias = NDArray[np.float32]
MasterFrame: TypeAlias = ArrayLike | str | Path
CombineMode: TypeAlias = Literal["sum", "mean"]


class Image:
    """Load, calibrate, and register supported images using NumPy.

    The processor retains both the original frames and their current combined
    or reduced image so later processing steps can reuse them.
    """

    def __init__(self) -> None:
        """Initialize the processor with no loaded image data."""
        self.paths: list[str] = []
        self.frames: list[ImageArray] = []
        self.raw_data: ImageArray | None = None
        self.data: ImageArray | None = None

    def load(
        self, paths: Iterable[str | Path], combine: CombineMode = "sum"
    ) -> ImageArray:
        """Load equal-sized images and combine them by sum or mean.

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
        result = combined.copy()
        self.data = result
        logger.info("Loaded %d image(s), shape=%s", len(frames), shape)
        return result

    def reduce(
        self,
        offset: MasterFrame | None = None,
        dark: MasterFrame | None = None,
        flat: MasterFrame | None = None,
    ) -> ImageArray:
        """Apply optional offset and dark subtraction and flat-field correction.

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

        reduced = self.raw_data.copy()
        if offset is not None:
            reduced -= self._master_array(offset, "offset")
        if dark is not None:
            reduced -= self._master_array(dark, "dark")
        if flat is not None:
            flat_data = self._master_array(flat, "flat")
            valid = np.isfinite(flat_data) & (flat_data != 0)
            if not np.any(valid):
                raise ValueError("Flat field contains no finite non-zero pixels.")
            normalized_flat = np.zeros_like(flat_data, dtype=np.float32)
            normalized_flat[valid] = flat_data[valid] / np.median(flat_data[valid])
            corrected = np.zeros_like(reduced, dtype=np.float32)
            np.divide(reduced, normalized_flat, out=corrected, where=valid)
            reduced = corrected

        self.data = reduced
        logger.info("Image reduction complete")
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
            shift = _phase_correlation_shift(reference, frame)
            logger.info("Registering frame %d with shift %s", index, shift)
            registered.append(_shift_image(frame, shift))

        self.frames = registered
        result = np.sum(registered, axis=0, dtype=np.float32)
        self.raw_data = result
        registered_sum = result.copy()
        self.data = registered_sum
        return registered_sum

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
        return self._validated_shape(array, name)

    def _validated_shape(self, array: ImageArray, name: str) -> ImageArray:
        """Ensure a calibration array matches the loaded science image.

        Args:
            array: Calibration data to validate.
            name: Calibration type used in error messages.

        Returns:
            The unchanged array after its dimensions have been validated.

        Raises:
            RuntimeError: If no science image has been loaded.
            ValueError: If the calibration dimensions differ from the science
                image dimensions.
        """
        if self.raw_data is None:
            raise RuntimeError("Load an image before validating calibration frames.")
        if array.shape != self.raw_data.shape:
            raise ValueError(
                f"The {name} frame has shape {array.shape}; "
                f"expected {self.raw_data.shape}."
            )
        return array


def _phase_correlation_shift(
    reference: ImageArray, moving: ImageArray
) -> tuple[float, float]:
    """Estimate the subpixel translation that aligns a frame to a reference.

    Args:
        reference: Reference image used for phase correlation.
        moving: Image whose translation is being measured.

    Returns:
        The vertical and horizontal shifts, in pixels, to apply to ``moving``.
    """
    reference_fft = np.fft.fftn(reference)
    moving_fft = np.fft.fftn(moving)
    cross_power = reference_fft * moving_fft.conj()
    magnitude = np.abs(cross_power)
    cross_power = np.divide(
        cross_power,
        magnitude,
        out=np.zeros_like(cross_power),
        where=magnitude > np.finfo(np.float32).eps,
    )
    correlation = np.fft.ifftn(cross_power).real
    peak = np.array(np.unravel_index(np.argmax(correlation), correlation.shape))
    shifts = peak.astype(np.float64)

    for axis, axis_size in enumerate(reference.shape):
        if shifts[axis] > axis_size // 2:
            shifts[axis] -= axis_size

        before_index = peak.copy()
        after_index = peak.copy()
        before_index[axis] = (peak[axis] - 1) % axis_size
        after_index[axis] = (peak[axis] + 1) % axis_size
        before = correlation[tuple(before_index)]
        center = correlation[tuple(peak)]
        after = correlation[tuple(after_index)]
        denominator = before - 2 * center + after
        if denominator != 0:
            shifts[axis] += 0.5 * (before - after) / denominator

    return float(shifts[0]), float(shifts[1])


def _shift_image(
    image: ImageArray,
    shift: tuple[float, float],
    block_rows: int = 64,
) -> ImageArray:
    """Translate an image using bilinear interpolation and zero fill.

    Args:
        image: Two-dimensional source image.
        shift: Vertical and horizontal shifts to apply, in pixels.
        block_rows: Number of output rows sampled in each processing block.

    Returns:
        A translated image with the same shape as the input.
    """
    height, width = image.shape
    source_x = np.arange(width, dtype=np.float32)[None, :] - shift[1]
    output = np.zeros_like(image, dtype=np.float32)
    for row_start in range(0, height, block_rows):
        row_end = min(row_start + block_rows, height)
        source_y = (
            np.arange(row_start, row_end, dtype=np.float32)[:, None] - shift[0]
        )
        output[row_start:row_end] = _bilinear_sample(image, source_x, source_y)
    return output


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

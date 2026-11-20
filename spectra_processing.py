"""Star Analyser (SA100/SA200) geometry, extraction and wavelength calibration."""

from dataclasses import dataclass
import logging
import math
from typing import Sequence, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from image_processing import _bilinear_sample

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ImageArray: TypeAlias = NDArray[np.float32]
Point: TypeAlias = tuple[float, float]
PointInput: TypeAlias = Sequence[float]


@dataclass(frozen=True)
class SpectrumResult:
    """Container for calibrated flux and intermediate SA100 extraction data.

    Attributes:
        wavelengths: Calibrated wavelengths in Angstroms.
        flux: Sky-subtracted spectral flux values.
        rectified_image: Input image after rotation to align the spectrum.
        spectrum_strip: Rectified pixels summed to produce the spectrum.
        sky_strip: Rectified background pixels used for sky subtraction.
        angle_degrees: Applied image rotation angle.
        dispersion_angstrom_per_pixel: Linear wavelength scale.
        spectrum_rows: Rows ``[low, high[`` of the spectrum band in the rectified image.
        sky_rows: Rows ``[low, high[`` of the sky band in the rectified image.
        x_start: First column of both bands in the rectified image.
    """

    wavelengths: NDArray[np.float64]
    flux: NDArray[np.float32]
    rectified_image: ImageArray
    spectrum_strip: ImageArray
    sky_strip: ImageArray
    angle_degrees: float
    dispersion_angstrom_per_pixel: float
    spectrum_rows: tuple[int, int] = (0, 0)
    sky_rows: tuple[int, int] = (0, 0)
    x_start: int = 0


class Spectra:
    """Rectify SA100 images, extract a spectrum, and calibrate wavelengths."""

    def __init__(
        self,
        focal_mm: float = 50.0,
        pixel_size_um: float = 4.65,
        grating_period_nm: float = 5000.0,
        binning: int = 1,
        spectrum_height: int = 10,
        sky_height: int = 20,
        lambda_min: float = 3500.0,
        lambda_max: float = 7500.0,
        order_zero_offset_px: int = 0,
    ) -> None:
        """Configure the optical geometry and extraction band widths.

        The grating sits in front of the lens, so the grating-to-sensor distance is the
        focal length. Wavelengths follow ``λ = d · sin(atan(x / f))`` where ``x`` is the
        distance to the zero order on the sensor.

        Args:
            focal_mm: Focal length of the lens in millimeters.
            pixel_size_um: Native sensor pixel pitch in micrometers.
            grating_period_nm: Grating period in nanometers (1e6 / lines per mm).
            binning: Sensor pixels per image pixel along each axis (2 for 2x2-binned images).
            spectrum_height: Thickness of the spectrum binning band in pixels.
            sky_height: Thickness of the sky-background band in pixels.
            lambda_min: Inclusive lower bound of the wavelength window in Å.
            lambda_max: Exclusive upper bound of the wavelength window in Å.
            order_zero_offset_px: Pixel offset used as the wavelength origin.

        Raises:
            ValueError: If an optical setting or wavelength range is invalid.
        """
        self.focal_mm = float(focal_mm)
        self.pixel_size_um = float(pixel_size_um)
        self.grating_period_nm = float(grating_period_nm)
        self.binning = int(binning)
        self.spectrum_height = int(spectrum_height)
        self.sky_height = int(sky_height)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.order_zero_offset_px = int(order_zero_offset_px)
        self.result: SpectrumResult | None = None
        self._validate_settings()

    def process(
        self,
        image: ArrayLike,
        order_zero: PointInput,
        dispersion_point: PointInput,
        sky_point: PointInput,
    ) -> SpectrumResult:
        """Rectify an image, subtract sky background, and calibrate its spectrum.

        Args:
            image: Two-dimensional detector image.
            order_zero: Pixel coordinates of the zero-order reference.
            dispersion_point: Pixel coordinates along the spectral trace.
            sky_point: Pixel coordinates defining the sky-background band.

        Returns:
            Calibrated wavelengths, flux, and intermediate extraction arrays.

        Raises:
            ValueError: If the image or selected points are invalid, extraction
                bands fall outside the image, or no pixels match the wavelength
                range.
        """
        data = np.asarray(image, dtype=np.float32)
        if data.ndim != 2 or min(data.shape) < 10:
            raise ValueError("SA100 processing requires a 2D image at least 10x10.")

        logger.info(
            "Extraction : image %d × %d px (min=%.1f médiane=%.1f max=%.1f)",
            data.shape[1], data.shape[0],
            float(np.nanmin(data)), float(np.nanmedian(data)), float(np.nanmax(data)),
        )
        x0, y0 = _point(order_zero, "order-zero point")
        x1, y1 = _point(dispersion_point, "dispersion point")
        xs, ys = _point(sky_point, "sky point")
        logger.info(
            "Points choisis : ordre 0=(%.1f, %.1f)  trace=(%.1f, %.1f)  ciel=(%.1f, %.1f)",
            x0, y0, x1, y1, xs, ys,
        )
        angle = compute_angle(x0, y0, x1, y1)

        rectified = _rotate_image(data, angle)

        height, width = rectified.shape
        center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
        x0_rotated, y0_rotated = rotate_point(x0, y0, center_x, center_y, angle)
        xs_rotated, sky_y_rotated = rotate_point(xs, ys, center_x, center_y, angle)
        logger.info(
            "Image tournée de %.2f° (angle de la trace) autour de (%.1f, %.1f) : ordre 0 → (%.1f, %.1f), "
            "ciel → y=%.1f",
            angle, center_x, center_y, x0_rotated, y0_rotated, sky_y_rotated,
        )
        logger.debug("Point de ciel après rotation : x=%.1f (non utilisé)", xs_rotated)

        spectrum_strip, sky_strip = crop_spectrum(
            rectified,
            x0_rotated,
            y0_rotated,
            sky_y_rotated,
            self.spectrum_height,
            self.sky_height,
        )
        if spectrum_strip.shape[0] == 0 or sky_strip.shape[0] == 0:
            raise ValueError("The spectrum or sky selection lies outside the image.")

        spectrum_1d = np.sum(spectrum_strip, axis=0, dtype=np.float64)
        sky_1d = np.sum(sky_strip, axis=0, dtype=np.float64)
        sky_scale = spectrum_strip.shape[0] / sky_strip.shape[0]
        raw_flux = spectrum_1d - sky_1d * sky_scale
        flux = np.clip(raw_flux, 0.0, None)
        logger.info(
            "Flux : somme de %d ligne(s) moins ciel (%d ligne(s), facteur %.3f) ; "
            "%d valeur(s) négative(s) ramenée(s) à 0 ; max=%.1f médiane=%.1f",
            spectrum_strip.shape[0], sky_strip.shape[0], sky_scale,
            int(np.count_nonzero(raw_flux < 0)), float(flux.max()), float(np.median(flux)),
        )

        x_start = int(round(x0_rotated))
        # Distance de chaque colonne à l'ordre zéro (sous-pixel : le centroïde n'est pas entier).
        offsets = x_start + np.arange(flux.size, dtype=np.float64) - x0_rotated - self.order_zero_offset_px
        wavelengths = star_analyser_wavelengths(
            offsets, self.grating_period_nm, self.focal_mm, self.pixel_size_um * self.binning
        )
        keep = (wavelengths > self.lambda_min) & (wavelengths < self.lambda_max)
        logger.info(
            "Fenêtre λ ]%.0f, %.0f[ Å : %d pixel(s) conservé(s) sur %d (λ calculées %.0f à %.0f Å)",
            self.lambda_min, self.lambda_max, int(np.count_nonzero(keep)), flux.size,
            float(wavelengths[0]), float(wavelengths[-1]),
        )
        if not np.any(keep):
            raise ValueError(
                "No pixels fall in the configured wavelength range; "
                "check the SA100 parameters and image."
            )

        kept = wavelengths[keep]
        dispersion = float((kept[-1] - kept[0]) / max(kept.size - 1, 1))
        self.result = SpectrumResult(
            wavelengths=kept,
            flux=flux[keep].astype(np.float32),
            rectified_image=rectified,
            spectrum_strip=spectrum_strip,
            sky_strip=sky_strip,
            angle_degrees=angle,
            dispersion_angstrom_per_pixel=dispersion,
            spectrum_rows=strip_rows(height, y0_rotated, self.spectrum_height),
            sky_rows=strip_rows(height, sky_y_rotated, self.sky_height),
            x_start=x_start,
        )
        logger.info(
            "Spectre extrait : %d échantillons, %.3f Å/px, λ %.0f–%.0f Å",
            self.result.flux.size, dispersion,
            float(self.result.wavelengths[0]), float(self.result.wavelengths[-1]),
        )
        return self.result

    def _validate_settings(self) -> None:
        """Validate optical parameters and extraction-band dimensions.

        Raises:
            ValueError: If any setting is non-positive or the wavelength range
                is empty or reversed.
        """
        if self.focal_mm <= 0 or self.pixel_size_um <= 0:
            raise ValueError("Focal length and pixel size must be positive.")
        if self.binning < 1:
            raise ValueError("Binning must be at least 1.")
        if self.grating_period_nm <= 0:
            raise ValueError("Grating period must be positive.")
        if self.spectrum_height < 1 or self.sky_height < 1:
            raise ValueError("Extraction strip thicknesses must be positive.")
        if self.lambda_max <= self.lambda_min:
            raise ValueError("lambda_max must be greater than lambda_min.")


def compute_angle(x0: float, y0: float, x1: float, y1: float) -> float:
    """Calculate the image-coordinate angle from order zero to the trace.

    Args:
        x0: Horizontal coordinate of the order-zero point.
        y0: Vertical coordinate of the order-zero point.
        x1: Horizontal coordinate of the dispersion point.
        y1: Vertical coordinate of the dispersion point.

    Returns:
        The direction angle in degrees.

    Raises:
        ValueError: If both points are identical.
    """
    if x0 == x1 and y0 == y1:
        raise ValueError("Order zero and dispersion point must be different.")
    angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
    logger.info("Angle de la trace : %.2f° (de l'ordre 0 vers le 2e point)", angle)
    return angle


def rotate_point(
    x: float,
    y: float,
    center_x: float,
    center_y: float,
    angle_degrees: float,
) -> Point:
    """Rotate a point using the image-coordinate convention of rectification.

    Args:
        x: Horizontal coordinate of the point.
        y: Vertical coordinate of the point.
        center_x: Horizontal rotation-center coordinate.
        center_y: Vertical rotation-center coordinate.
        angle_degrees: Rotation angle in degrees.

    Returns:
        Rotated horizontal and vertical coordinates.
    """
    radians = math.radians(-angle_degrees)
    dx, dy = x - center_x, y - center_y
    return (
        center_x + dx * math.cos(radians) - dy * math.sin(radians),
        center_y + dx * math.sin(radians) + dy * math.cos(radians),
    )


def strip_rows(height: int, center_y: float, thickness: int) -> tuple[int, int]:
    """Rows ``[low, high[`` of a horizontal band centred on ``center_y``, clipped to the image.

    Args:
        height: Image height in pixels.
        center_y: Vertical centre of the band.
        thickness: Band thickness in pixels (odd values are centred on a pixel row).

    Returns:
        The first row and the row after the last one.
    """
    low = int(round(center_y)) - thickness // 2
    return max(0, low), min(height, low + thickness)


def crop_spectrum(
    image: ImageArray,
    x0_rotated: float,
    y0_rotated: float,
    sky_y_rotated: float,
    spectrum_height: int,
    sky_height: int,
) -> tuple[ImageArray, ImageArray]:
    """Extract the spectral and sky strips from a rectified image.

    Args:
        image: Rectified two-dimensional detector image.
        x0_rotated: Horizontal start coordinate of the spectrum.
        y0_rotated: Vertical center of the spectral trace.
        sky_y_rotated: Vertical center of the sky-background band.
        spectrum_height: Thickness of the spectrum strip in pixels.
        sky_height: Thickness of the sky strip in pixels.

    Returns:
        Copies of the spectrum strip and sky strip, respectively.

    Raises:
        ValueError: If the order-zero coordinate lies outside the image.
    """
    height, width = image.shape
    x_start = int(round(x0_rotated))
    if not 0 <= x_start < width:
        raise ValueError("The order-zero point is outside the image.")

    spectrum_low, spectrum_high = strip_rows(height, y0_rotated, spectrum_height)
    sky_low, sky_high = strip_rows(height, sky_y_rotated, sky_height)
    logger.info(
        "Bandes (image tournée) : spectre y=[%d, %d[ (épaisseur %d, centre %d), "
        "ciel y=[%d, %d[ (épaisseur %d, centre %d), départ x=%d",
        spectrum_low, spectrum_high, spectrum_height, round(y0_rotated),
        sky_low, sky_high, sky_height, round(sky_y_rotated), x_start,
    )
    if spectrum_high - spectrum_low < spectrum_height:
        logger.warning("Bande du spectre tronquée par le bord de l'image")
    if sky_high - sky_low < sky_height:
        logger.warning("Bande de ciel tronquée par le bord de l'image")
    if spectrum_low < sky_high and sky_low < spectrum_high:
        logger.warning("Les bandes spectre et ciel se chevauchent : le ciel contient du signal")

    return (
        image[spectrum_low:spectrum_high, x_start:].copy(),
        image[sky_low:sky_high, x_start:].copy(),
    )


def star_analyser_wavelengths(
    offsets_px: ArrayLike,
    grating_period_nm: float,
    focal_mm: float,
    pixel_pitch_um: float,
) -> NDArray[np.float64]:
    """Wavelengths (Å) at given pixel distances from the zero order.

    ``λ = d · sin(atan(x / f))`` with ``x = offset · pixel pitch``: exact for a grating in
    front of a lens (no linear approximation). Negative offsets give negative wavelengths.

    Args:
        offsets_px: Distances to the zero order, in image pixels.
        grating_period_nm: Grating period ``d`` in nanometers.
        focal_mm: Focal length ``f`` in millimeters.
        pixel_pitch_um: Size of one *image* pixel in micrometers (native pitch × binning).

    Returns:
        Wavelengths in Angstroms.

    Raises:
        ValueError: If a geometry value is not positive.
    """
    if grating_period_nm <= 0 or focal_mm <= 0 or pixel_pitch_um <= 0:
        raise ValueError("Grating period, focal length and pixel pitch must be positive.")
    x_mm = np.asarray(offsets_px, dtype=np.float64) * (pixel_pitch_um / 1000.0)
    wavelengths = grating_period_nm * np.sin(np.arctan2(x_mm, focal_mm)) * 10.0
    logger.info(
        "Calibration : d=%.0f nm, f=%.1f mm, pixel image %.2f µm → %.3f Å/px à l'origine, "
        "%.3f Å/px au bout (λ %.0f–%.0f Å)",
        grating_period_nm, focal_mm, pixel_pitch_um,
        grating_period_nm * (pixel_pitch_um / 1000.0) / focal_mm * 10.0,
        float(np.mean(np.diff(wavelengths))) if wavelengths.size > 1 else 0.0,
        float(wavelengths.min(initial=0.0)), float(wavelengths.max(initial=0.0)),
    )
    return wavelengths


def refine_centroid(
    image: ImageArray, x: float, y: float, radius: int, iterations: int = 6
) -> tuple[float, float]:
    """Refine a clicked position to the intensity-weighted centroid of the star around it.

    The window (``2·radius+1`` pixels wide) is re-centred on the centroid at each iteration.
    The local background is the window median, and only pixels clearly above it
    (3 robust σ) contribute, so noise does not pull the result toward the window centre.

    Args:
        image: 2D image.
        x: Clicked horizontal position (pixels).
        y: Clicked vertical position (pixels).
        radius: Half-size of the search window; 0 disables the refinement.
        iterations: Maximum number of re-centring steps.

    Returns:
        The refined ``(x, y)``, or the input position if there is no usable signal.
    """
    if radius <= 0:
        logger.debug("Affinage du centroïde désactivé (rayon 0)")
        return x, y
    height, width = image.shape
    cx, cy = x, y
    for step in range(iterations):
        x_low, x_high = max(0, int(round(cx)) - radius), min(width, int(round(cx)) + radius + 1)
        y_low, y_high = max(0, int(round(cy)) - radius), min(height, int(round(cy)) + radius + 1)
        window = image[y_low:y_high, x_low:x_high].astype(np.float64)
        if window.size == 0 or not np.isfinite(window).all():
            logger.warning("Affinage impossible en (%.1f, %.1f) : fenêtre vide ou non finie", x, y)
            return x, y
        background = float(np.median(window))
        sigma = 1.4826 * float(np.median(np.abs(window - background)))
        weights = np.where(window > background + 3.0 * sigma, window - background, 0.0)
        total = float(weights.sum())
        if total <= 0:
            logger.warning(
                "Affinage impossible en (%.1f, %.1f) : aucun signal au-dessus du fond "
                "dans une fenêtre de ±%d px", x, y, radius,
            )
            return x, y
        ys, xs = np.mgrid[y_low:y_high, x_low:x_high]
        new_x, new_y = float((weights * xs).sum() / total), float((weights * ys).sum() / total)
        moved = math.hypot(new_x - cx, new_y - cy)
        cx, cy = new_x, new_y
        if moved < 0.01:
            break
    shift = math.hypot(cx - x, cy - y)
    logger.info(
        "Ordre zéro affiné au centroïde : clic (%.1f, %.1f) → (%.2f, %.2f), décalage %.2f px "
        "(fenêtre ±%d px, %d itération(s))",
        x, y, cx, cy, shift, radius, step + 1,
    )
    if shift > 0.75 * radius:
        logger.warning(
            "L'ordre zéro a été déplacé de %.1f px (> 75 %% du rayon %d) : étoile mal centrée sur le clic ?",
            shift, radius,
        )
    return cx, cy


def _rotate_image(
    image: ImageArray, angle_degrees: float, block_rows: int = 64
) -> ImageArray:
    """Rotate an image within its original bounds using bilinear interpolation.

    Args:
        image: Two-dimensional source image.
        angle_degrees: Rotation angle in degrees.
        block_rows: Number of output rows processed at a time.

    Returns:
        Rotated float32 image with the same shape as the input.
    """
    height, width = image.shape
    logger.debug("Rotation bilinéaire %.2f° par blocs de %d lignes", angle_degrees, block_rows)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    radians = math.radians(angle_degrees)
    cos_angle = math.cos(radians)
    sin_angle = math.sin(radians)
    output = np.zeros_like(image, dtype=np.float32)
    output_x = np.arange(width, dtype=np.float32)[None, :] - center_x

    for row_start in range(0, height, block_rows):
        row_end = min(row_start + block_rows, height)
        output_y = (
            np.arange(row_start, row_end, dtype=np.float32)[:, None] - center_y
        )
        source_x = center_x + cos_angle * output_x - sin_angle * output_y
        source_y = center_y + sin_angle * output_x + cos_angle * output_y
        output[row_start:row_end] = _bilinear_sample(
            image,
            source_x,
            source_y,
        )
    return output


def _point(point: PointInput, name: str) -> Point:
    """Validate and convert one finite image point to floating-point values.

    Args:
        point: Sequence containing horizontal and vertical coordinates.
        name: Point label used in validation errors.

    Returns:
        Point coordinates as a pair of Python floats.

    Raises:
        ValueError: If the sequence does not contain two finite coordinates.
    """
    if len(point) != 2:
        raise ValueError(f"{name} must contain x and y coordinates.")
    x, y = float(point[0]), float(point[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{name} coordinates must be finite.")
    return x, y

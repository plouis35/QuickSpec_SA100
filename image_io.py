"""Image file readers for QuickSpec's supported raster and FITS formats."""

import gzip
import logging
from pathlib import Path
from typing import BinaryIO, TypeAlias

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PillowImage

logger = logging.getLogger(__name__)

ImageArray: TypeAlias = NDArray[np.float32]
FitsHeader: TypeAlias = dict[str, str]
ImagePath: TypeAlias = str | Path
RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FITS_EXTENSIONS = {".fit", ".fits", ".fts"}
COLOR_BIN_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff"}
FITS_BLOCK_SIZE = 2880
FITS_CARD_SIZE = 80

_FITS_DTYPES = {
    8: np.dtype("u1"),
    16: np.dtype(">i2"),
    32: np.dtype(">i4"),
    64: np.dtype(">i8"),
    -32: np.dtype(">f4"),
    -64: np.dtype(">f8"),
}
_GRAYSCALE_MODES = {"1", "L", "I", "I;16", "I;16B", "I;16L", "F"}


def read_image(path: ImagePath) -> ImageArray:
    """Read one supported raster or FITS image into a 2D float32 array.

    Args:
        path: Path to PNG, JPG, TIFF, FITS, or gzip-compressed FITS data.

    Returns:
        Image pixels in row-major order as a two-dimensional float32 array.

    Raises:
        FileNotFoundError: If the input path does not exist.
        ValueError: If the extension or file contents are unsupported.
    """
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    name = image_path.name.lower()
    if name.endswith(".gz"):
        stem_extension = Path(image_path.stem).suffix.lower()
        if stem_extension not in FITS_EXTENSIONS:
            raise ValueError(f"Unsupported compressed image format: {image_path}")
        return read_fits(image_path, compressed=True)

    extension = image_path.suffix.lower()
    if extension in FITS_EXTENSIONS:
        return read_fits(image_path)
    if extension not in RASTER_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {image_path.suffix or image_path}")
    return _read_raster(image_path, bin_color=extension in COLOR_BIN_EXTENSIONS)


def read_fits(path: ImagePath, compressed: bool = False) -> ImageArray:
    """Read a simple two-dimensional primary-HDU FITS image.

    Args:
        path: Path to an uncompressed or gzip-compressed FITS file.
        compressed: Whether to decompress the FITS stream using gzip.

    Returns:
        Image pixels as a two-dimensional float32 array, with FITS scaling
        applied and integer ``BLANK`` pixels represented as NaN.

    Raises:
        ValueError: If the header/data are invalid or the FITS layout is not a
            single, uncompressed 2D primary image.
    """
    image_path = Path(path)
    opener = gzip.open if compressed else open
    try:
        with opener(image_path, "rb") as source:
            header = _read_fits_header(source, image_path)
            if header.get("SIMPLE") != "T":
                raise ValueError("Only standard primary-HDU FITS images are supported.")

            bitpix = _fits_integer(header, "BITPIX", image_path)
            if bitpix not in _FITS_DTYPES:
                raise ValueError(f"Unsupported FITS BITPIX={bitpix}: {image_path}")
            naxis = _fits_integer(header, "NAXIS", image_path)
            if naxis != 2:
                raise ValueError(
                    f"Only simple 2D FITS images are supported (NAXIS={naxis})."
                )
            width = _fits_integer(header, "NAXIS1", image_path)
            height = _fits_integer(header, "NAXIS2", image_path)
            if width <= 0 or height <= 0:
                raise ValueError("FITS image dimensions must be positive.")

            dtype = _FITS_DTYPES[bitpix]
            count = width * height
            data_size = count * dtype.itemsize
            payload = source.read(data_size)
            if len(payload) != data_size:
                raise ValueError(f"FITS image data is truncated: {image_path}")
            raw = np.frombuffer(payload, dtype=dtype, count=count).reshape(height, width)
            padding = source.read(-data_size % FITS_BLOCK_SIZE)
            if len(padding) != -data_size % FITS_BLOCK_SIZE:
                raise ValueError(f"FITS image data is truncated: {image_path}")
            if source.read(1):
                raise ValueError("Only single-HDU FITS images are supported.")

            if bitpix > 0 and "BLANK" in header:
                blank = _fits_integer(header, "BLANK", image_path)
                invalid = raw == blank
            else:
                invalid = None

            bscale = _fits_float(header, "BSCALE", 1.0, image_path)
            bzero = _fits_float(header, "BZERO", 0.0, image_path)
            if invalid is None and bscale == 1.0 and bzero == 0.0:
                image = raw.astype(np.float32)
            else:
                image = raw.astype(np.float64)
                image *= bscale
                image += bzero
                if invalid is not None:
                    image[invalid] = np.nan
                image = image.astype(np.float32)
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        raise ValueError(f"Unable to read FITS image {image_path}: {exc}") from exc

    if not image.size:
        raise ValueError(f"FITS image is empty: {image_path}")
    return image


def _read_raster(path: Path, bin_color: bool) -> ImageArray:
    """Read a Pillow-supported raster image and convert it to grayscale.

    Args:
        path: Raster image path.
        bin_color: Whether color pixel values should be summed in 2x2 blocks.

    Returns:
        A two-dimensional float32 grayscale array.
    """
    try:
        with PillowImage.open(path) as source:
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError(f"Multi-page raster images are not supported: {path}")
            source.load()
            if source.mode in _GRAYSCALE_MODES:
                pixels = np.asarray(source)
                bin_pixels = path.suffix.lower() in {".jpg", ".jpeg"}
            else:
                rgb = np.asarray(source.convert("RGB"), dtype=np.float32)
                pixels = (
                    rgb[..., 0] * 0.2126
                    + rgb[..., 1] * 0.7152
                    + rgb[..., 2] * 0.0722
                )
                bin_pixels = bin_color
            if bin_pixels:
                pixels = _bin_2x2(pixels, path)
    except (OSError, ValueError, SyntaxError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(
            ("Multi-page raster", "Image must be at least")
        ):
            raise
        raise ValueError(f"Unable to decode image {path}: {exc}") from exc

    if pixels.ndim != 2 or not pixels.size:
        raise ValueError(f"Image must contain a non-empty 2D array: {path}")
    return np.asarray(pixels, dtype=np.float32)


def _bin_2x2(pixels: NDArray, path: Path) -> NDArray[np.float64]:
    """Sum each complete 2x2 pixel block and discard incomplete edges.

    Args:
        pixels: Two-dimensional array to bin.
        path: Source path used to identify invalid dimensions in errors.

    Returns:
        A two-dimensional float64 array of summed pixel blocks.
    """
    height, width = pixels.shape[0] // 2, pixels.shape[1] // 2
    if height == 0 or width == 0:
        raise ValueError(f"Image must be at least 2x2 pixels for binning: {path}")
    return pixels[: height * 2, : width * 2].reshape(
        height, 2, width, 2
    ).sum(axis=(1, 3), dtype=np.float64)


def _read_fits_header(source: BinaryIO, path: Path) -> FitsHeader:
    """Read FITS header cards through the END card and its padded block.

    Args:
        source: Binary stream positioned at the start of the FITS header.
        path: Source path included in malformed-header errors.

    Returns:
        Mapping of FITS keywords to their parsed textual values.
    """
    header: FitsHeader = {}
    found_end = False
    while not found_end:
        block = source.read(FITS_BLOCK_SIZE)
        if len(block) != FITS_BLOCK_SIZE:
            raise ValueError(f"FITS header is truncated or missing END: {path}")
        for offset in range(0, FITS_BLOCK_SIZE, FITS_CARD_SIZE):
            card = block[offset : offset + FITS_CARD_SIZE].decode("ascii")
            keyword = card[:8].strip()
            if keyword == "END":
                found_end = True
                break
            if keyword and card[8:10] == "= ":
                header[keyword] = _parse_fits_value(card[10:])
    return header


def _parse_fits_value(field: str) -> str:
    """Parse the value field of a standard FITS header card.

    Args:
        field: Text following the equals sign in an 80-character FITS card.

    Returns:
        The value with surrounding whitespace and comments removed.
    """
    value = field.split("/", 1)[0].strip()
    if value.startswith("'"):
        return value[1 : value.rfind("'")].rstrip()
    return value


def _fits_integer(header: FitsHeader, key: str, path: Path) -> int:
    """Return a required integer FITS keyword.

    Args:
        header: Parsed FITS header values.
        key: Keyword to retrieve.
        path: Source path included in validation errors.

    Returns:
        The keyword value converted to an integer.
    """
    try:
        value = header[key]
        if isinstance(value, str) and value.startswith("'"):
            raise ValueError
        return int(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid FITS {key}: {path}") from exc


def _fits_float(
    header: FitsHeader, key: str, default: float, path: Path
) -> float:
    """Return a floating-point FITS keyword or its default value.

    Args:
        header: Parsed FITS header values.
        key: Keyword to retrieve.
        default: Value to use when the keyword is absent.
        path: Source path included in validation errors.

    Returns:
        The keyword or default converted to a Python float.
    """
    try:
        value = header.get(key, default)
        return float(str(value).replace("D", "E").replace("d", "e"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid FITS {key}: {path}") from exc

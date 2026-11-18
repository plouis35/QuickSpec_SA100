"""Tests for raster and simple FITS image readers."""

import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image as PillowImage

from image_io import read_image
from image_processing import Image


class ImageIoTests(unittest.TestCase):
    """Verify raster decoding and minimal FITS image parsing."""

    def setUp(self) -> None:
        """Create a temporary directory for image fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)

    def test_tiff_grayscale_preserves_16_bit_values_and_resolution(self) -> None:
        """Preserve TIFF grayscale values and image dimensions."""
        expected = np.array([[0, 1000], [30000, 65535]], dtype=np.uint16)
        path = self.directory / "gray16.tiff"
        PillowImage.fromarray(expected).save(path)

        result = read_image(path)

        np.testing.assert_array_equal(result, expected.astype(np.float32))

    def test_color_tiff_is_converted_and_summed_in_2_by_2_bins(self) -> None:
        """Convert color TIFF pixels to luminance and sum each 2x2 block."""
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[:2, :2] = (100, 0, 0)
        rgb[:2, 2:] = (0, 100, 0)
        rgb[2:, :2] = (0, 0, 100)
        rgb[2:, 2:] = (100, 100, 100)
        path = self.directory / "color.tiff"
        PillowImage.fromarray(rgb).save(path)

        result = read_image(path)

        expected = np.array(
            [
                [4 * 100 * 0.2126, 4 * 100 * 0.7152],
                [4 * 100 * 0.0722, 4 * 100],
            ],
            dtype=np.float32,
        )
        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_jpeg_is_binned_instead_of_loaded_at_full_resolution(self) -> None:
        """Bin decoded JPEG luminance into 2x2 pixel blocks."""
        rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        rgb[:, :2] = (220, 30, 10)
        rgb[:, 2:4] = (10, 200, 30)
        rgb[:, 4:] = (20, 40, 230)
        path = self.directory / "color.jpg"
        PillowImage.fromarray(rgb).save(path, quality=100, subsampling=0)

        with PillowImage.open(path) as source:
            decoded = np.asarray(source.convert("RGB"), dtype=np.float32)
        grayscale = (
            decoded[..., 0] * 0.2126
            + decoded[..., 1] * 0.7152
            + decoded[..., 2] * 0.0722
        )
        expected = grayscale.reshape(2, 2, 3, 2).sum(axis=(1, 3))

        result = read_image(path)

        self.assertEqual(result.shape, (2, 3))
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_png_color_keeps_existing_full_resolution_behavior(self) -> None:
        """Keep color PNG conversion at its original spatial resolution."""
        rgb = np.full((4, 6, 3), (10, 20, 30), dtype=np.uint8)
        path = self.directory / "color.png"
        PillowImage.fromarray(rgb).save(path)

        result = read_image(path)

        self.assertEqual(result.shape, (4, 6))
        np.testing.assert_allclose(result, 10 * 0.2126 + 20 * 0.7152 + 30 * 0.0722)

    def test_fits_integer_image_applies_scaling_blank_and_axis_order(self) -> None:
        """Apply FITS scaling, blank pixels, and row/column axis ordering."""
        raw = np.array([[-2, 1, 2], [3, 4, 5]], dtype=np.int16)
        path = self.directory / "scaled.fits"
        self._write_fits(path, raw, bitpix=16, bscale=2, bzero=10, blank=-2)

        result = read_image(path)

        self.assertEqual(result.shape, (2, 3))
        np.testing.assert_allclose(
            result,
            [[np.nan, 12, 14], [16, 18, 20]],
            equal_nan=True,
        )

    def test_gzip_fits_reads_floating_point_data(self) -> None:
        """Read a gzip-compressed FITS floating-point primary image."""
        raw = np.array([[1.25, -2.5], [3.0, 4.5]], dtype=np.float32)
        path = self.directory / "image.fits.gz"
        payload = self._fits_bytes(raw, bitpix=-32)
        with gzip.open(path, "wb") as target:
            target.write(payload)

        np.testing.assert_array_equal(read_image(path), raw)

    def test_fits_rejects_non_2d_and_truncated_data(self) -> None:
        """Reject FITS cubes and incomplete image payloads."""
        raw = np.ones((2, 2), dtype=np.int16)
        three_d = self.directory / "cube.fits"
        three_d.write_bytes(self._fits_bytes(raw, bitpix=16, naxis=3))
        with self.assertRaisesRegex(ValueError, "NAXIS=3"):
            read_image(three_d)

        truncated = self.directory / "truncated.fits"
        payload = self._fits_bytes(raw, bitpix=16)
        truncated.write_bytes(payload[:2880] + payload[2880:2887])
        with self.assertRaisesRegex(ValueError, "truncated"):
            read_image(truncated)

    def test_fits_rejects_additional_hdus(self) -> None:
        """Reject FITS files containing additional HDUs."""
        raw = np.ones((2, 2), dtype=np.int16)
        path = self.directory / "multiple-hdu.fits"
        path.write_bytes(self._fits_bytes(raw, bitpix=16) + b"extra HDU")

        with self.assertRaisesRegex(ValueError, "single-HDU"):
            read_image(path)

    def test_image_processor_loads_fits_through_io_module(self) -> None:
        """Verify the image processor uses the FITS I/O implementation."""
        raw = np.array([[1, 2], [3, 4]], dtype=np.int16)
        path = self.directory / "integrated.fits"
        self._write_fits(path, raw, bitpix=16)

        result = Image().load([path])

        np.testing.assert_array_equal(result, raw.astype(np.float32))

    def test_unsupported_format_is_rejected(self) -> None:
        """Reject files with unsupported image extensions."""
        path = self.directory / "image.cr2"
        path.touch()
        with self.assertRaisesRegex(ValueError, "Unsupported image format"):
            read_image(path)

    def _write_fits(
        self,
        path: Path,
        data: np.ndarray,
        bitpix: int,
        bscale: float | None = None,
        bzero: float | None = None,
        blank: int | None = None,
    ) -> None:
        """Write a FITS fixture with the requested header and image data."""
        path.write_bytes(
            self._fits_bytes(
                data,
                bitpix=bitpix,
                bscale=bscale,
                bzero=bzero,
                blank=blank,
            )
        )

    @staticmethod
    def _fits_bytes(
        data: np.ndarray,
        bitpix: int,
        bscale: float | None = None,
        bzero: float | None = None,
        blank: int | None = None,
        naxis: int = 2,
    ) -> bytes:
        """Build a padded primary-HDU FITS byte stream for tests."""
        data = np.asarray(data)
        cards = [
            ("SIMPLE", "T"),
            ("BITPIX", str(bitpix)),
            ("NAXIS", str(naxis)),
            ("NAXIS1", str(data.shape[-1])),
            ("NAXIS2", str(data.shape[-2])),
        ]
        if bscale is not None:
            cards.append(("BSCALE", str(bscale)))
        if bzero is not None:
            cards.append(("BZERO", str(bzero)))
        if blank is not None:
            cards.append(("BLANK", str(blank)))
        cards.append(("END", None))
        header = b"".join(
            (
                f"{key:<8}".ljust(8)
                if value is None
                else f"{key:<8}= {value:>20}"
            ).ljust(80).encode("ascii")
            for key, value in cards
        )
        header += b" " * (-len(header) % 2880)

        dtypes = {
            8: ">u1",
            16: ">i2",
            32: ">i4",
            64: ">i8",
            -32: ">f4",
            -64: ">f8",
        }
        payload = data.astype(dtypes[bitpix]).tobytes()
        payload += b"\0" * (-len(payload) % 2880)
        return header + payload


if __name__ == "__main__":
    unittest.main()

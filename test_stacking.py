"""Tests des masters (offset, dark, flat), du recalage scipy et de la somme d'images."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image as PillowImage

from calibration import MASTER_EXTENSIONS, find_master, load_masters
from image_processing import Image, REGISTRATION_MIN_SNR, _phase_correlation_shift
from settings import AppParameters
from spectra_processing import strip_rows


def save_gray16(path: Path, data: np.ndarray) -> Path:
    """Écrit un PNG 16 bits en niveaux de gris (lu tel quel, sans binning)."""
    PillowImage.fromarray(np.asarray(data, dtype=np.uint16)).save(path)
    return path


def star_field(height: int = 160, width: int = 240, dy: float = 0.0, dx: float = 0.0,
               noise: float = 3.0, seed: int = 0) -> np.ndarray:
    """Champ d'étoiles gaussiennes (positions fixes), décalé de (dy, dx), avec bruit."""
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[:height, :width]
    image = np.full((height, width), 200.0)
    for _ in range(15):
        y, x, amplitude = rng.uniform(20, height - 20), rng.uniform(20, width - 20), rng.uniform(300, 3000)
        image += amplitude * np.exp(-((yy - y - dy) ** 2 + (xx - x - dx) ** 2) / (2 * 2.0**2))
    return (image + np.random.default_rng(seed).normal(0, noise, image.shape)).astype(np.float32)


class MasterLookupTests(unittest.TestCase):
    """Recherche des fichiers master par nom, avec ou sans extension."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_extensions_are_fits_png_jpeg(self) -> None:
        self.assertEqual(set(MASTER_EXTENSIONS), {".fits", ".fit", ".fts", ".png", ".jpg", ".jpeg"})

    def test_finds_png_without_extension(self) -> None:
        save_gray16(self.directory / "master_dark.png", np.zeros((4, 4)))
        self.assertEqual(find_master(self.directory, "master_dark"), self.directory / "master_dark.png")

    def test_name_with_extension_and_uppercase_extension(self) -> None:
        save_gray16(self.directory / "master_flat.PNG", np.ones((4, 4)))
        self.assertIsNotNone(find_master(self.directory, "master_flat"))
        self.assertIsNotNone(find_master(self.directory, "master_flat.PNG"))

    def test_fits_has_priority_over_png(self) -> None:
        (self.directory / "master_offset.fits").write_bytes(b"x")
        save_gray16(self.directory / "master_offset.png", np.zeros((4, 4)))
        with self.assertLogs("calibration", "WARNING"):
            found = find_master(self.directory, "master_offset")
        self.assertEqual(found.name, "master_offset.fits")

    def test_missing_master_logs_a_warning_and_is_none(self) -> None:
        save_gray16(self.directory / "master_dark.png", np.full((4, 4), 5))
        with self.assertLogs("calibration", "WARNING") as logs:
            masters = load_masters(
                self.directory,
                {"offset": "master_offset", "dark": "master_dark", "flat": "master_flat"},
            )
        self.assertIsNone(masters["offset"])
        self.assertIsNone(masters["flat"])
        np.testing.assert_array_equal(masters["dark"], np.full((4, 4), 5, dtype=np.float32))
        self.assertEqual(sum("introuvable" in line for line in logs.output), 2)

    def test_empty_name_disables_without_warning(self) -> None:
        with self.assertNoLogs("calibration", "WARNING"):
            masters = load_masters(self.directory, {"offset": "", "dark": "", "flat": ""})
        self.assertEqual(masters, {"offset": None, "dark": None, "flat": None})

    def test_missing_directory_warns_once(self) -> None:
        with self.assertLogs("calibration", "WARNING") as logs:
            masters = load_masters(self.directory / "absent", {"offset": "a", "dark": "b"})
        self.assertEqual(masters, {"offset": None, "dark": None})
        self.assertEqual(len(logs.output), 1)


class StackTests(unittest.TestCase):
    """Correction image par image, recalage et somme."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_each_frame_is_corrected_before_the_sum(self) -> None:
        """somme = Σ (image − offset − dark) / flat normalisé, pas (Σ image − offset − dark)."""
        rng = np.random.default_rng(0)
        frames = [rng.integers(1000, 2000, (12, 16)) for _ in range(3)]
        offset = np.full((12, 16), 100.0)
        dark = np.full((12, 16), 20.0)
        flat = rng.uniform(0.5, 1.5, (12, 16)).astype(np.float32)
        paths = [save_gray16(self.directory / f"f{i}.png", f) for i, f in enumerate(frames)]

        result = Image().load_stack(paths, offset=offset, dark=dark, flat=flat, register=False)

        normalized = flat / np.median(flat)
        expected = sum((f - 120.0) / normalized for f in frames)
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_wrong_size_master_is_ignored_with_a_warning(self) -> None:
        path = save_gray16(self.directory / "a.png", np.full((10, 10), 50))
        processor = Image()
        with self.assertLogs("image_processing", "WARNING") as logs:
            result = processor.load_stack([path], offset=np.ones((4, 4)), dark=np.full((10, 10), 5))
        np.testing.assert_allclose(result, 45.0)
        self.assertEqual(processor.applied_corrections, ["dark"])
        self.assertTrue(any("offset ignoré" in line for line in logs.output))

    def test_frames_of_different_size_are_rejected(self) -> None:
        first = save_gray16(self.directory / "a.png", np.zeros((10, 10)))
        second = save_gray16(self.directory / "b.png", np.zeros((10, 12)))
        with self.assertRaisesRegex(ValueError, "b.png"):
            Image().load_stack([first, second])

    def test_shifted_frames_are_registered_on_the_first(self) -> None:
        """Trois champs décalés (sous-pixel compris) se recalent à moins de 0,1 px."""
        shifts = [(0.0, 0.0), (3.4, -2.6), (-5.0, 7.25)]
        paths = [
            save_gray16(self.directory / f"s{i}.png", star_field(dy=dy, dx=dx, seed=i))
            for i, (dy, dx) in enumerate(shifts)
        ]
        processor = Image()
        stacked = processor.load_stack(paths)

        for (dy, dx), (est_dy, est_dx) in zip(shifts[1:], processor.registration_shifts[1:]):
            self.assertAlmostEqual(est_dy, -dy, delta=0.1)
            self.assertAlmostEqual(est_dx, -dx, delta=0.1)
        # le pic des étoiles reste net : la somme recalée est plus piquée que la somme brute
        unregistered = Image().load_stack(paths, register=False)
        self.assertGreater(float(stacked[20:-20, 20:-20].max()), float(unregistered[20:-20, 20:-20].max()))

    def test_unrelated_images_trigger_a_registration_warning(self) -> None:
        a = save_gray16(self.directory / "a.png", np.random.default_rng(1).normal(2000, 50, (120, 160)))
        b = save_gray16(self.directory / "b.png", np.random.default_rng(2).normal(2000, 50, (120, 160)))
        with self.assertLogs("image_processing", "WARNING") as logs:
            Image().load_stack([a, b])
        self.assertTrue(any("peu fiable" in line for line in logs.output))

    def test_phase_correlation_snr_is_high_for_real_pairs(self) -> None:
        snr = _phase_correlation_shift(star_field(seed=1), star_field(dy=2.0, dx=3.0, seed=2))[2]
        self.assertGreater(snr, 2 * REGISTRATION_MIN_SNR)

    def test_non_finite_pixels_are_replaced_and_reported(self) -> None:
        data = np.full((10, 10), 100.0, dtype=np.float32)
        data[2, 3] = np.nan
        offset = np.zeros((10, 10), dtype=np.float32)
        with self.assertLogs("image_processing", "WARNING"):
            result = Image().load_stack([self._fits(data)], offset=offset)
        self.assertEqual(result[2, 3], 0.0)
        self.assertTrue(np.isfinite(result).all())

    def _fits(self, data: np.ndarray) -> Path:
        """FITS 32 bits flottants minimal."""
        header = {"SIMPLE": "T", "BITPIX": "-32", "NAXIS": "2",
                  "NAXIS1": str(data.shape[1]), "NAXIS2": str(data.shape[0])}
        cards = [f"{key:<8}= {value:>20}".ljust(80) for key, value in header.items()] + ["END".ljust(80)]
        block = "".join(cards).ljust(-(-len(cards) * 80 // 2880) * 2880).encode("ascii")
        payload = data.astype(">f4").tobytes()
        path = self.directory / "nan.fits"
        path.write_bytes(block + payload + bytes(-len(payload) % 2880))
        return path


class BandAndParameterTests(unittest.TestCase):
    """Épaisseur des bandes et nouveaux paramètres."""

    def test_strip_rows_supports_even_and_odd_thickness(self) -> None:
        self.assertEqual(strip_rows(100, 40, 10), (35, 45))
        self.assertEqual(strip_rows(100, 40, 5), (38, 43))  # centrée sur la rangée 40
        self.assertEqual(strip_rows(100, 2, 10), (0, 7))  # bornée par le bord de l'image

    def test_new_parameters_have_defaults_and_old_files_still_load(self) -> None:
        parameters = AppParameters.from_mapping({"focal_mm": 85, "line_size": 2})
        self.assertEqual((parameters.spectrum_height, parameters.sky_height), (10, 20))
        self.assertEqual(
            (parameters.offset_name, parameters.dark_name, parameters.flat_name),
            ("master_offset", "master_dark", "master_flat"),
        )
        self.assertEqual(parameters.calibration_dir, "")

    def test_band_thickness_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            AppParameters(spectrum_height=0)
        with self.assertRaises(ValueError):
            AppParameters(sky_height=-3)


if __name__ == "__main__":
    unittest.main()

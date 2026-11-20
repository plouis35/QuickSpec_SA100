"""Tests du mode Star Analyser : formule de dispersion, binning et affinage de l'ordre zéro."""

import unittest

import numpy as np

from settings import AppParameters
from spectra_processing import Spectra, refine_centroid, star_analyser_wavelengths


def gaussian_star(shape=(60, 80), x=40.3, y=30.7, sigma=1.5, noise=0.5, seed=0):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    rng = np.random.default_rng(seed)
    star = 1000.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    return (star + 50.0 + rng.normal(0, noise, shape)).astype(np.float32)


class WavelengthTests(unittest.TestCase):
    def test_halpha_position_sa200_canon6d_binned(self):
        # d=5000 nm, f=50 mm, pixel 6.54 µm × 2 : Hα (6563 Å) vers 506 px de l'ordre zéro.
        x_px = np.array([506.0])
        wl = star_analyser_wavelengths(x_px, 5000.0, 50.0, 13.08)[0]
        self.assertAlmostEqual(wl, 6563.0, delta=15.0)

    def test_exact_formula(self):
        wl = star_analyser_wavelengths([100.0], 5000.0, 50.0, 10.0)[0]
        expected = 5000.0 * np.sin(np.arctan(1.0 / 50.0)) * 10.0
        self.assertAlmostEqual(wl, expected, places=6)

    def test_binning_doubles_pixel_scale(self):
        single = star_analyser_wavelengths([100.0], 5000.0, 50.0, 6.54)[0]
        binned = star_analyser_wavelengths([100.0], 5000.0, 50.0, 13.08)[0]
        self.assertAlmostEqual(binned / single, 2.0, delta=0.01)

    def test_invalid_geometry(self):
        with self.assertRaises(ValueError):
            star_analyser_wavelengths([1.0], 5000.0, 0.0, 6.54)

    def test_spectra_uses_binning(self):
        image = np.full((80, 400), 100.0, dtype=np.float32)
        image[38:42, 50:] += 500.0
        results = []
        for binning in (1, 2):
            sp = Spectra(focal_mm=50, pixel_size_um=6.54, grating_period_nm=5000.0,
                         binning=binning, spectrum_height=4, sky_height=10,
                         lambda_min=100.0, lambda_max=20000.0)
            results.append(sp.process(image, (50.0, 40.0), (150.0, 40.0), (150.0, 55.0)))
        self.assertAlmostEqual(
            results[1].dispersion_angstrom_per_pixel / results[0].dispersion_angstrom_per_pixel,
            2.0, delta=0.02,
        )


class CentroidTests(unittest.TestCase):
    def test_refines_to_star_centre(self):
        image = gaussian_star(x=40.3, y=30.7)
        x, y = refine_centroid(image, 43, 28, radius=8)
        self.assertAlmostEqual(x, 40.3, delta=0.15)
        self.assertAlmostEqual(y, 30.7, delta=0.15)

    def test_radius_zero_disables(self):
        self.assertEqual(refine_centroid(gaussian_star(), 43.0, 28.0, radius=0), (43.0, 28.0))

    def test_no_signal_keeps_click(self):
        flat = np.full((40, 40), 100.0, dtype=np.float32)
        self.assertEqual(refine_centroid(flat, 20.0, 20.0, radius=5), (20.0, 20.0))

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            AppParameters(centroid_radius=-1)
        self.assertEqual(AppParameters().centroid_radius, 8)


if __name__ == "__main__":
    unittest.main()

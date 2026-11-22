"""Tests de la lecture de spectres 1D et de la réponse instrumentale."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from response import (
    apply_response,
    compute_response,
    load_response,
    save_response,
    smooth_reference,
)
from settings import AppParameters
from spectrum_io import read_spectrum, write_spectrum


def card(key: str, value: str) -> bytes:
    return f"{key:<8}= {value:>20}".ljust(80).encode("ascii")


def write_fits_1d(path: Path, flux: np.ndarray, crval: float, cdelt: float, extra=()) -> None:
    cards = [card("SIMPLE", "T"), card("BITPIX", "-32"), card("NAXIS", "1"),
             card("NAXIS1", str(flux.size)), card("CRVAL1", repr(crval)),
             card("CDELT1", repr(cdelt)), card("CRPIX1", "1")]
    cards += [card(k, v) for k, v in extra] + [b"END".ljust(80)]
    header = b"".join(cards).ljust(-(-len(cards) * 80 // 2880) * 2880)
    data = flux.astype(">f4").tobytes()
    path.write_bytes(header + data + b"\0" * (-len(data) % 2880))


def reference_star(step=0.9):
    """Continuum bleuté avec raies de Balmer larges : une étoile de type A simplifiée."""
    wl = np.arange(3500.0, 7430.0, step)
    flux = (wl / 5500.0) ** -2.0
    for line, depth in ((4101.7, 0.4), (4340.5, 0.4), (4861.3, 0.5), (6562.8, 0.5)):
        flux = flux * (1 - depth * np.exp(-0.5 * ((wl - line) / 8.0) ** 2))
    return wl, flux


def instrument(wl):
    x = (wl - 5500.0) / 2000.0
    return np.exp(-1.5 * x**2) * (1 + 0.2 * x) + 0.05


class SpectrumIoTests(unittest.TestCase):
    def test_dat_with_header_comma_and_unordered_rows(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "s.csv"
            path.write_text("# commentaire\nlambda,flux\n5000,2.0\n4000;1.0\n4500 1.5\n")
            wl, flux = read_spectrum(path)
        np.testing.assert_array_equal(wl, [4000.0, 4500.0, 5000.0])
        np.testing.assert_array_equal(flux, [1.0, 1.5, 2.0])

    def test_nanometres_are_converted_to_angstroms(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nm.dat"
            path.write_text("400 1\n500 2\n600 3\n")
            wl, _ = read_spectrum(path)
        np.testing.assert_allclose(wl, [4000.0, 5000.0, 6000.0])

    def test_write_then_read_roundtrip(self) -> None:
        wl = np.linspace(4000, 7000, 50)
        flux = np.sin(wl / 500) + 2
        with TemporaryDirectory() as directory:
            path = Path(directory) / "out.dat"
            write_spectrum(path, wl, flux, comment="test")
            wl2, flux2 = read_spectrum(path)
        np.testing.assert_allclose(wl2, wl, atol=1e-4)
        np.testing.assert_allclose(flux2, flux, rtol=1e-5)

    def test_fits_1d_wavelength_axis(self) -> None:
        flux = np.arange(1, 101, dtype=np.float32)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "s.fits"
            write_fits_1d(path, flux, 4000.0, 2.5)
            wl, data = read_spectrum(path)
        self.assertAlmostEqual(wl[0], 4000.0)
        self.assertAlmostEqual(wl[-1], 4000.0 + 99 * 2.5)
        np.testing.assert_allclose(data, flux)

    def test_fits_in_nanometres_and_missing_calibration(self) -> None:
        flux = np.ones(20, dtype=np.float32)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nm.fit"
            write_fits_1d(path, flux, 400.0, 1.0, extra=[("CUNIT1", "'nm'")])
            wl, _ = read_spectrum(path)
            self.assertAlmostEqual(wl[0], 4000.0)
            bad = Path(directory) / "bad.fits"
            bad.write_bytes(path.read_bytes().replace(b"CRVAL1", b"XXXXXX1"))
            with self.assertRaises(ValueError):
                read_spectrum(bad)

    def test_unknown_or_empty_files_raise(self) -> None:
        with TemporaryDirectory() as directory:
            empty = Path(directory) / "e.dat"
            empty.write_text("# rien\n")
            with self.assertRaises(ValueError):
                read_spectrum(empty)
            with self.assertRaises(ValueError):
                read_spectrum(Path(directory) / "x.xyz")


class ResponseTests(unittest.TestCase):
    PARAMS = dict(degree=6, smooth_px=3.0, line_half_width=60.0, lambda_min=3600.0, lambda_max=7400.0)

    def observe(self, ref_wl, ref_flux, grid):
        fwhm = 3.0 * (grid[1] - grid[0])
        smooth = smooth_reference(ref_wl, ref_flux, fwhm)
        rng = np.random.default_rng(3)
        return np.interp(grid, ref_wl, smooth) * instrument(grid) * 1000 * (1 + rng.normal(0, 0.01, grid.size))

    def test_response_recovers_instrument_shape(self) -> None:
        ref_wl, ref_flux = reference_star()
        grid = np.arange(3600.0, 7400.0, 13.1)
        fit = compute_response(grid, self.observe(ref_wl, ref_flux, grid), ref_wl, ref_flux, **self.PARAMS)
        truth = instrument(fit.wavelengths)
        truth /= np.median(truth)
        self.assertLess(np.max(np.abs(fit.response / truth - 1)), 0.05)
        self.assertAlmostEqual(float(np.median(fit.response)), 1.0, places=6)
        self.assertLess(fit.rms_percent, 3.0)

    def test_dividing_the_reference_star_gives_back_a_flat_continuum_shape(self) -> None:
        ref_wl, ref_flux = reference_star()
        grid = np.arange(3600.0, 7400.0, 13.1)
        measured = self.observe(ref_wl, ref_flux, grid)
        fit = compute_response(grid, measured, ref_wl, ref_flux, **self.PARAMS)
        wl, corrected = apply_response(grid, measured, (fit.wavelengths, fit.response))
        expected = np.interp(wl, ref_wl, smooth_reference(ref_wl, ref_flux, 3 * 13.1))
        ratio = corrected / expected
        ratio /= np.median(ratio)
        self.assertLess(np.max(np.abs(ratio - 1)), 0.06)

    def test_insufficient_overlap_raises(self) -> None:
        ref_wl = np.arange(8000.0, 9000.0, 0.9)
        grid = np.arange(3600.0, 7400.0, 13.1)
        with self.assertRaises(ValueError):
            compute_response(grid, np.ones_like(grid), ref_wl, np.ones_like(ref_wl), **self.PARAMS)

    def test_save_load_and_apply_crop_outside_range(self) -> None:
        ref_wl, ref_flux = reference_star()
        grid = np.arange(3600.0, 7400.0, 13.1)
        fit = compute_response(grid, self.observe(ref_wl, ref_flux, grid), ref_wl, ref_flux, **self.PARAMS)
        with TemporaryDirectory() as directory:
            self.assertIsNone(load_response(Path(directory), "master_response"))
            save_response(Path(directory), "master_response", fit)
            loaded = load_response(Path(directory), "master_response")
        self.assertIsNotNone(loaded)
        wide = np.arange(3000.0, 8000.0, 10.0)
        wl, flux = apply_response(wide, np.ones_like(wide), loaded)
        self.assertGreaterEqual(wl[0], loaded[0][0])
        self.assertLessEqual(wl[-1], loaded[0][-1])
        self.assertLess(wl.size, wide.size)
        self.assertTrue(np.all(np.isfinite(flux)))

    def test_parameters_validation_and_defaults(self) -> None:
        defaults = AppParameters()
        self.assertEqual(
            (defaults.response_name, defaults.response_degree, defaults.response_smooth_px,
             defaults.response_line_mask),
            ("master_response", 6, 3.0, 60),
        )
        for bad in ({"response_degree": 0}, {"response_degree": 13},
                    {"response_smooth_px": -1.0}, {"response_line_mask": -5}):
            with self.assertRaises(ValueError):
                AppParameters(**bad)


if __name__ == "__main__":
    unittest.main()

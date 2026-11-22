"""Lecture et écriture de spectres 1D : FITS (CRVAL1/CDELT1) ou texte DAT/CSV à deux colonnes."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from image_io import _FITS_DTYPES, _fits_float, _fits_integer, _read_fits_header

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FITS_EXTENSIONS = (".fits", ".fit", ".fts")
TEXT_EXTENSIONS = (".dat", ".txt", ".csv")
# Extensions (sans point) proposées par le sélecteur de fichiers.
SPECTRUM_EXTENSIONS = [e.lstrip(".") for e in TEXT_EXTENSIONS + FITS_EXTENSIONS]

SpectrumData = tuple[NDArray[np.float64], NDArray[np.float64]]


def read_spectrum(path: str | Path) -> SpectrumData:
    """Lit un spectre 1D et renvoie ``(longueurs d'onde en Å, flux)`` triés par longueur d'onde.

    Args:
        path: Fichier FITS 1D (``.fits``/``.fit``/``.fts``) ou texte (``.dat``/``.txt``/``.csv``).

    Raises:
        ValueError: Si le fichier est illisible ou contient moins de trois points valides.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in FITS_EXTENSIONS:
        wavelengths, flux = _read_fits_spectrum(path)
    elif suffix in TEXT_EXTENSIONS:
        wavelengths, flux = _read_text_spectrum(path)
    else:
        raise ValueError(f"Format de spectre non reconnu : {path.name}")

    valid = np.isfinite(wavelengths) & np.isfinite(flux)
    wavelengths, flux = wavelengths[valid], flux[valid]
    if wavelengths.size < 3:
        raise ValueError(f"Moins de trois points valides dans {path.name}")
    order = np.argsort(wavelengths, kind="stable")
    wavelengths, flux = wavelengths[order], flux[order]
    logger.info(
        "Spectre lu : %s — %d points, λ %.1f–%.1f Å, flux min=%.4g max=%.4g",
        path.name, wavelengths.size, wavelengths[0], wavelengths[-1], flux.min(), flux.max(),
    )
    return wavelengths, flux


def write_spectrum(path: str | Path, wavelengths: NDArray, flux: NDArray, comment: str = "") -> None:
    """Écrit un spectre en texte (colonnes « λ (Å)  flux »), lisible par ``read_spectrum``."""
    path = Path(path)
    lines = [f"# {line}" for line in comment.splitlines()] + ["# wavelength_A flux"]
    lines += [f"{w:.4f} {f:.6e}" for w, f in zip(wavelengths, flux)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Spectre écrit : %s (%d points)", path, len(wavelengths))


def _read_text_spectrum(path: Path) -> SpectrumData:
    """Deux premières colonnes numériques ; en-têtes et commentaires ignorés."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"Lecture impossible de {path.name} : {exc}") from exc
    rows: list[tuple[float, float]] = []
    skipped = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in "#%":
            continue
        tokens = line.replace(",", " ").replace(";", " ").split()
        try:
            rows.append((float(tokens[0]), float(tokens[1])))
        except (ValueError, IndexError):
            skipped += 1
    if skipped:
        logger.info("%s : %d ligne(s) non numérique(s) ignorée(s)", path.name, skipped)
    if not rows:
        raise ValueError(f"Aucune colonne numérique lisible dans {path.name}")
    data = np.array(rows, dtype=np.float64)
    return _to_angstrom(data[:, 0], "", path), data[:, 1]


def _read_fits_spectrum(path: Path) -> SpectrumData:
    """FITS 1D : λ = CRVAL1 + (i + 1 − CRPIX1) · CDELT1 (ou CD1_1)."""
    try:
        with open(path, "rb") as source:
            header = _read_fits_header(source, path)
            if header.get("SIMPLE") != "T":
                raise ValueError("FITS standard attendu.")
            bitpix = _fits_integer(header, "BITPIX", path)
            if bitpix not in _FITS_DTYPES:
                raise ValueError(f"BITPIX={bitpix} non pris en charge.")
            naxis = _fits_integer(header, "NAXIS", path)
            count = _fits_integer(header, "NAXIS1", path)
            if naxis not in (1, 2) or (naxis == 2 and _fits_integer(header, "NAXIS2", path) != 1):
                raise ValueError(f"Spectre 1D attendu (NAXIS={naxis}).")
            dtype = _FITS_DTYPES[bitpix]
            payload = source.read(count * dtype.itemsize)
            if len(payload) != count * dtype.itemsize:
                raise ValueError("Données FITS tronquées.")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Lecture impossible de {path.name} : {exc}") from exc

    flux = np.frombuffer(payload, dtype=dtype, count=count).astype(np.float64)
    flux = flux * _fits_float(header, "BSCALE", 1.0, path) + _fits_float(header, "BZERO", 0.0, path)

    step = header.get("CDELT1") or header.get("CD1_1")
    if "CRVAL1" not in header or step is None:
        raise ValueError(f"{path.name} : CRVAL1 et CDELT1 (ou CD1_1) sont nécessaires à la calibration.")
    crval = _fits_float(header, "CRVAL1", 0.0, path)
    cdelt = _fits_float(header, "CDELT1" if "CDELT1" in header else "CD1_1", 1.0, path)
    crpix = _fits_float(header, "CRPIX1", 1.0, path)
    wavelengths = crval + (np.arange(count) + 1.0 - crpix) * cdelt
    return _to_angstrom(wavelengths, header.get("CUNIT1", ""), path), flux


def _to_angstrom(wavelengths: NDArray[np.float64], unit: str, path: Path) -> NDArray[np.float64]:
    """Convertit en Å d'après l'unité déclarée, sinon d'après l'ordre de grandeur."""
    unit = unit.strip().lower()
    if unit.startswith(("angstrom", "ang", "å")):
        return wavelengths
    if unit == "nm" or unit.startswith("nanom"):
        factor = 10.0
    elif unit in ("um", "micron", "microns") or unit.startswith("micro"):
        factor = 1e4
    elif unit:
        logger.warning("%s : unité « %s » inconnue, supposée en Å", path.name, unit)
        return wavelengths
    else:
        top = float(np.nanmax(np.abs(wavelengths))) if wavelengths.size else 0.0
        if top < 30:
            factor = 1e4
        elif top < 1000:
            factor = 10.0
        else:
            return wavelengths
        logger.warning(
            "%s : pas d'unité déclarée, longueurs d'onde supposées en %s (×%g → Å)",
            path.name, "µm" if factor == 1e4 else "nm", factor,
        )
        return wavelengths * factor
    logger.info("%s : unité %s convertie en Å (×%g)", path.name, unit, factor)
    return wavelengths * factor

"""Réponse instrumentale : calcul à partir d'une étoile de référence, stockage et application.

Principe : réponse(λ) = spectre mesuré de l'étoile / spectre de référence connu (MILES, CALSPEC…),
la référence étant d'abord lissée à la résolution de l'instrument. Le rapport est ajusté par un
polynôme de bas degré (en ignorant les raies de Balmer et les bandes telluriques) pour que la
réponse soit lisse et n'injecte aucune raie de l'étoile de référence dans les autres spectres.
Chaque spectre extrait est ensuite divisé par cette réponse.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from spectrum_io import FITS_EXTENSIONS, TEXT_EXTENSIONS, read_spectrum, write_spectrum

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BALMER_LINES_A = (3835.4, 3889.1, 3970.1, 4101.7, 4340.5, 4861.3, 6562.8)
# Bandes telluriques (O2 B, H2O, O2 A), exclues de l'ajustement : (λ min, λ max) en Å.
TELLURIC_BANDS_A = ((6860.0, 6930.0), (7160.0, 7340.0), (7590.0, 7700.0))
MIN_RESPONSE_FRACTION = 0.05  # en dessous, la réponse est jugée inutilisable (division instable)
# Le DAT que nous écrivons passe en premier ; les autres formats sont acceptés s'ils existent.
RESPONSE_EXTENSIONS = TEXT_EXTENSIONS + FITS_EXTENSIONS

Response = tuple[NDArray[np.float64], NDArray[np.float64]]


@dataclass(frozen=True)
class ResponseFit:
    """Réponse calculée et indicateurs de qualité de l'ajustement."""

    wavelengths: NDArray[np.float64]
    response: NDArray[np.float64]  # normalisée à une médiane de 1
    points_used: int
    rms_percent: float  # dispersion du rapport mesuré/référence autour du polynôme


def smooth_reference(
    wavelengths: NDArray[np.float64], flux: NDArray[np.float64], fwhm_angstrom: float
) -> NDArray[np.float64]:
    """Dégrade la référence à la résolution de l'instrument (gaussienne de largeur à mi-hauteur donnée).

    Les valeurs nulles ou négatives (zones sans données) sont ignorées ; la sortie y vaut 0.
    """
    step = float(np.median(np.diff(wavelengths)))
    valid = flux > 0
    sigma_px = fwhm_angstrom / 2.3548 / step
    if sigma_px < 0.3:
        return np.where(valid, flux, 0.0)
    weight = gaussian_filter1d(valid.astype(np.float64), sigma_px, mode="constant")
    smooth = gaussian_filter1d(np.where(valid, flux, 0.0), sigma_px, mode="constant")
    return np.where(weight > 0.5, smooth / np.maximum(weight, 1e-12), 0.0)


def _excluded(wavelengths: NDArray[np.float64], line_half_width: float) -> NDArray[np.bool_]:
    """Longueurs d'onde à ignorer : raies de Balmer (± demi-largeur) et bandes telluriques."""
    excluded = np.zeros(wavelengths.shape, dtype=bool)
    for line in BALMER_LINES_A:
        excluded |= np.abs(wavelengths - line) <= line_half_width
    for low, high in TELLURIC_BANDS_A:
        excluded |= (wavelengths >= low) & (wavelengths <= high)
    return excluded


def _longest_run(mask: NDArray[np.bool_]) -> slice:
    """Plus longue suite de valeurs vraies consécutives."""
    best = (0, 0)
    start = None
    for index, value in enumerate(np.append(mask, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    return slice(*best)


def compute_response(
    measured_wavelengths: NDArray[np.float64],
    measured_flux: NDArray[np.float64],
    reference_wavelengths: NDArray[np.float64],
    reference_flux: NDArray[np.float64],
    *,
    degree: int,
    smooth_px: float,
    line_half_width: float,
    lambda_min: float,
    lambda_max: float,
) -> ResponseFit:
    """Calcule la réponse instrumentale (flux mesuré / référence) ajustée par un polynôme.

    Args:
        measured_wavelengths: Longueurs d'onde (Å) du spectre extrait de l'étoile de référence.
        measured_flux: Flux de ce spectre (non corrigé).
        reference_wavelengths: Longueurs d'onde (Å) du spectre de référence connu.
        reference_flux: Flux de référence.
        degree: Degré du polynôme.
        smooth_px: Résolution de l'instrument en pixels du spectre mesuré (largeur à mi-hauteur
            de la gaussienne appliquée à la référence).
        line_half_width: Demi-largeur (Å) masquée autour de chaque raie de Balmer.
        lambda_min: Début de la fenêtre utile (Å).
        lambda_max: Fin de la fenêtre utile (Å).

    Raises:
        ValueError: Si le recouvrement des deux spectres est insuffisant pour l'ajustement.
    """
    wl = np.asarray(measured_wavelengths, dtype=np.float64)
    flux = np.asarray(measured_flux, dtype=np.float64)
    window = (wl >= lambda_min) & (wl <= lambda_max) & np.isfinite(flux) & (flux > 0)
    if window.sum() < degree + 5:
        raise ValueError("Spectre mesuré trop court (ou vide) dans la fenêtre de longueurs d'onde.")

    dispersion = float(np.median(np.abs(np.diff(wl[window]))))
    fwhm = smooth_px * dispersion
    logger.info(
        "Réponse : dispersion %.2f Å/px, référence lissée à %.1f px = %.1f Å (FWHM), degré %d, "
        "masque ±%.0f Å autour des raies de Balmer",
        dispersion, smooth_px, fwhm, degree, line_half_width,
    )
    smooth_ref = smooth_reference(
        np.asarray(reference_wavelengths, dtype=np.float64),
        np.asarray(reference_flux, dtype=np.float64),
        fwhm,
    )
    ref_on_grid = np.interp(wl, reference_wavelengths, smooth_ref, left=0.0, right=0.0)
    ref_ok = ref_on_grid > 1e-3 * ref_on_grid.max()

    good = window & ref_ok & ~_excluded(wl, line_half_width)
    if good.sum() < degree + 5:
        raise ValueError(
            f"Recouvrement insuffisant avec la référence ({int(good.sum())} points utilisables, "
            f"{degree + 5} nécessaires)."
        )
    ratio = flux[good] / ref_on_grid[good]
    poly = np.polynomial.Polynomial.fit(wl[good], ratio, degree)
    rms = float(np.std(ratio / poly(wl[good]) - 1.0) * 100.0)

    covered = window & ref_ok
    low, high = wl[good].min(), wl[good].max()
    inside = covered & (wl >= low) & (wl <= high)
    out_wl, response = wl[inside], poly(wl[inside])
    usable = response > MIN_RESPONSE_FRACTION * response.max()
    run = _longest_run(usable)
    out_wl, response = out_wl[run], response[run]
    if out_wl.size < 10:
        raise ValueError("La réponse ajustée est négative ou nulle presque partout : baisse le degré.")
    if run.stop - run.start < usable.size:
        logger.warning(
            "Réponse tronquée à %.0f–%.0f Å (au-delà, elle tombe sous %d %% de son maximum)",
            out_wl[0], out_wl[-1], int(MIN_RESPONSE_FRACTION * 100),
        )
    response = response / np.median(response)
    logger.info(
        "Réponse calculée : %d points d'ajustement, λ %.0f–%.0f Å, dispersion du rapport "
        "autour du polynôme %.1f %%",
        int(good.sum()), out_wl[0], out_wl[-1], rms,
    )
    if rms > 10:
        logger.warning(
            "Le rapport mesuré/référence s'écarte de %.0f %% du polynôme : étoile de référence "
            "mal choisie, mal identifiée, ou degré trop bas ?", rms,
        )
    return ResponseFit(out_wl, response, int(good.sum()), rms)


def save_response(directory: Path, name: str, fit: ResponseFit) -> Path:
    """Enregistre la réponse en ``<directory>/<name>.dat`` (deux colonnes λ en Å, réponse)."""
    path = directory / f"{name}.dat"
    write_spectrum(
        path, fit.wavelengths, fit.response,
        comment=f"QuickSpec master_response (flux mesuré / référence, médiane = 1)\n"
        f"points d'ajustement : {fit.points_used} ; dispersion autour du polynôme : {fit.rms_percent:.1f} %",
    )
    return path


def load_response(directory: Path, name: str) -> Response | None:
    """Charge la réponse ``name`` (avec ou sans extension) ; ``None`` si absente ou illisible."""
    candidates = [directory / name] + [directory / f"{name}{ext}" for ext in RESPONSE_EXTENSIONS]
    path = next((c for c in candidates if c.suffix.lower() in RESPONSE_EXTENSIONS and c.is_file()), None)
    if path is None:
        logger.info("Pas de réponse instrumentale (%s absent de %s) : spectre non corrigé", name, directory)
        return None
    try:
        wavelengths, response = read_spectrum(path)
    except ValueError as exc:
        logger.warning("Réponse instrumentale illisible (%s) : %s", path.name, exc)
        return None
    if not np.all(response > 0):
        logger.warning("Réponse instrumentale %s invalide (valeurs ≤ 0) : ignorée", path.name)
        return None
    return wavelengths, response


def apply_response(
    wavelengths: NDArray[np.float64], flux: NDArray[np.float64], response: Response
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Divise le flux par la réponse ; les points hors de la plage de la réponse sont retirés."""
    response_wl, response_values = response
    inside = (wavelengths >= response_wl[0]) & (wavelengths <= response_wl[-1])
    if not inside.any():
        raise ValueError("Aucune longueur d'onde du spectre dans la plage de la réponse instrumentale.")
    dropped = int((~inside).sum())
    if dropped:
        logger.warning(
            "%d point(s) hors de la plage de la réponse (%.0f–%.0f Å) retiré(s) du spectre",
            dropped, response_wl[0], response_wl[-1],
        )
    values = np.interp(wavelengths[inside], response_wl, response_values)
    logger.info("Spectre divisé par la réponse instrumentale (%d points)", int(inside.sum()))
    return wavelengths[inside], flux[inside] / values

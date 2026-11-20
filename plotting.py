"""Rendu matplotlib du spectre et aperçu PNG de l'image (sans dépendance à Flet)."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import logging
import math
import time
from typing import TypeAlias

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from PIL import Image as PillowImage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

StageSize: TypeAlias = tuple[float, float]
WavelengthLimits: TypeAlias = tuple[float, float]

PLOT_DPI = 110
DEFAULT_PLOT_SIZE: StageSize = (770.0, 440.0)  # 7 x 4 pouces à 110 dpi
MIN_PLOT_SIZE: StageSize = (240.0, 160.0)
MAX_PLOT_SIZE: StageSize = (3000.0, 2000.0)

MAX_SPECTRUM_TRACES = 8
SPECTRUM_COLORS = (
    "#ff0000",
    "#008000",
    "#87cefa",
    "#ffff00",
    "#ffa500",
    "#ee82ee",
    "#0000ff",
    "#a52a2a",
)
ELEMENT_LINE_COLORS = {
    "H": "#d1d5db",
    "He I": "#d1d5db",
    "Na I": "#d1d5db",
    "Mg I": "#d1d5db",
    "Ca II": "#d1d5db",
}
ELEMENT_LINES: tuple[tuple[float, str, str], ...] = (
    (4101.74, "Hδ", "H"),
    (4340.47, "Hγ", "H"),
    (4861.33, "Hβ", "H"),
    (6562.80, "Hα", "H"),
    (4471.48, "He I", "He I"),
    (5875.62, "He I", "He I"),
    (6678.15, "He I", "He I"),
    (7065.19, "He I", "He I"),
    (5889.95, "Na D2", "Na I"),
    (5895.92, "Na D1", "Na I"),
    (5167.32, "Mg b4", "Mg I"),
    (5172.68, "Mg b2", "Mg I"),
    (5183.60, "Mg b1", "Mg I"),
    (3933.66, "Ca K", "Ca II"),
    (3968.47, "Ca H", "Ca II"),
)
VALID_LINE_TYPES = {
    "-", "--", "-.", ":", "None", "none", " ", "",
    "solid", "dashed", "dashdot", "dotted",
}
# Couleurs approximatives du spectre visible (λ en Å -> RVB).
_COLOR_WAVELENGTHS = np.array([3800, 4400, 4900, 5100, 5800, 6000, 7000, 7500], dtype=np.float64)
_COLOR_VALUES = np.array(
    [
        [0.55, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.5, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ],
    dtype=np.float64,
)
_BG = "#111827"


@dataclass(frozen=True)
class SpectrumTrace:
    """Un spectre extrait et son identité graphique."""

    wavelengths: NDArray[np.float64]
    flux: NDArray[np.float32]
    color: str
    label: str


def image_bounds(
    image_shape: tuple[int, int], stage_size: StageSize
) -> tuple[float, float, float, float]:
    """Rectangle occupé par l'image dans une zone d'affichage (ajustement « contain »).

    Args:
        image_shape: Dimensions de l'image ``(hauteur, largeur)`` en pixels.
        stage_size: Dimensions de la zone ``(largeur, hauteur)``.

    Returns:
        ``(gauche, haut, largeur, hauteur)`` du rectangle affiché.

    Raises:
        ValueError: Si une dimension n'est pas strictement positive.
    """
    image_height, image_width = image_shape
    stage_width, stage_height = stage_size
    if min(image_width, image_height, stage_width, stage_height) <= 0:
        raise ValueError("Image and viewport dimensions must be positive.")
    scale = min(stage_width / image_width, stage_height / image_height)
    display_width = image_width * scale
    display_height = image_height * scale
    return (
        (stage_width - display_width) / 2,
        (stage_height - display_height) / 2,
        display_width,
        display_height,
    )


def display_limits(data: ArrayLike) -> tuple[float, float]:
    """Bornes d'affichage (percentiles 0,5 et 99,5 %) d'une image.

    Raises:
        ValueError: Si l'entrée n'est pas 2D ou ne contient aucun pixel fini.
    """
    image = np.asarray(data, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("The image preview requires a 2D grayscale array.")
    finite = image[np.isfinite(image)]
    if not finite.size:
        raise ValueError("The image contains no finite grayscale pixels.")
    low, high = (float(v) for v in np.percentile(finite, (0.5, 99.5)))
    if high <= low:
        high = low + 1
    return low, high


# Curseurs de contraste : position 0–100 → percentile 0–100, avec une échelle dilatée près de
# 100 % (le spectre est une toute petite fraction des pixels : un curseur linéaire ne
# permettrait pas de régler finement entre 99 et 100 %). 0 → 0 %, 100 → 100 %.
CUT_EXPONENT = 3.0
AUTO_CUT_PERCENTILES = (0.5, 99.5)
CUT_SAMPLE_SIZE = 1_000_000


def cut_percentile(position: float) -> float:
    """Percentile (0–100) correspondant à la position d'un curseur (0–100)."""
    return 100.0 * (1.0 - (1.0 - position / 100.0) ** CUT_EXPONENT)


def cut_position(percentile: float) -> float:
    """Position de curseur (0–100) correspondant à un percentile (0–100)."""
    return 100.0 * (1.0 - (1.0 - percentile / 100.0) ** (1.0 / CUT_EXPONENT))


def cut_sample(data: ArrayLike) -> NDArray[np.float32]:
    """Échantillon (au plus 1 M pixels finis, tirage reproductible) pour calculer les percentiles vite."""
    image = np.asarray(data, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size > CUT_SAMPLE_SIZE:
        finite = finite[np.random.default_rng(0).integers(0, finite.size, CUT_SAMPLE_SIZE)]
    return finite


def cut_limits(sample: NDArray[np.float32], low_percentile: float, high_percentile: float) -> tuple[float, float]:
    """Niveaux noir et blanc pour deux percentiles de l'échantillon (blanc > noir garanti)."""
    low, high = (float(v) for v in np.percentile(sample, (low_percentile, high_percentile)))
    if high <= low:
        high = low + 1
    return low, high


def image_png_data_uri(data: ArrayLike, limits: tuple[float, float] | None = None) -> str:
    """Aperçu PNG (niveaux de gris) sous forme de data URI.

    Args:
        data: Image 2D ; le tableau n'est pas modifié.
        limits: Bornes noir/blanc. Par défaut, les percentiles 0,5–99,5 % de
            ``data`` ; on les impose pour afficher deux images avec le même contraste.

    Raises:
        ValueError: Si l'entrée n'est pas 2D ou ne contient aucun pixel fini.
    """
    started = time.perf_counter()
    image = np.asarray(data, dtype=np.float32)
    low, high = limits if limits is not None else display_limits(image)
    if high <= low:
        high = low + 1
    preview = np.nan_to_num(image, nan=low, posinf=high, neginf=low)
    preview = np.clip((preview - low) * (255.0 / (high - low)), 0, 255)
    output = BytesIO()
    PillowImage.fromarray(preview.astype(np.uint8)).save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    logger.debug(
        "Aperçu PNG %dx%d : étirement [%.1f, %.1f]%s, %d ko, %.0f ms",
        image.shape[1], image.shape[0], low, high,
        " (imposé)" if limits is not None else "",
        len(encoded) // 1024, (time.perf_counter() - started) * 1000,
    )
    return f"data:image/png;base64,{encoded}"


def _visible_spectrum_rgb(
    wavelengths: ArrayLike, intensities: ArrayLike
) -> NDArray[np.float64]:
    """Couleur visible de chaque longueur d'onde, modulée par l'intensité (0 à 1)."""
    wl = np.asarray(wavelengths, dtype=np.float64)
    inten = np.asarray(intensities, dtype=np.float64)
    if wl.ndim != 1 or inten.ndim != 1 or wl.size != inten.size:
        raise ValueError("Wavelength and intensity arrays must be matching 1D arrays.")
    colors = np.column_stack(
        [
            np.interp(wl, _COLOR_WAVELENGTHS, _COLOR_VALUES[:, ch], left=0.0, right=0.0)
            for ch in range(3)
        ]
    )
    positive = np.maximum(inten, 0.0)
    peak = float(np.max(positive, initial=0.0))
    if peak > 0:
        colors *= (positive / peak)[:, np.newaxis]
    else:
        colors.fill(0.0)
    return colors


def _as_1d_pair(x: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convertit (λ, flux) en tableaux 1D de même taille."""
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    if xv.ndim != 1 or yv.ndim != 1 or xv.size != yv.size:
        raise ValueError("Wavelength and flux arrays must be matching 1D arrays.")
    return xv, yv


def _draw_element_lines(axes) -> int:
    """Trace les raies H/He/Na/Mg/Ca situées dans la plage visible ; renvoie leur nombre."""
    low, high = axes.get_xlim()
    shown = [line for line in ELEMENT_LINES if low <= line[0] <= high]
    for index, (wavelength, label, element) in enumerate(shown):
        color = ELEMENT_LINE_COLORS[element]
        axes.axvline(wavelength, color=color, linestyle="--", alpha=0.65, linewidth=0.8)
        axes.text(
            wavelength,
            0.97 - (index % 4) * 0.12,
            f"{label} {wavelength:.0f}",
            transform=axes.get_xaxis_transform(),
            color=color,
            fontsize=7,
            rotation=90,
            verticalalignment="top",
        )
    return len(shown)


def _color_band_values(trace: SpectrumTrace | None, low: float, high: float) -> NDArray[np.float64]:
    """Bande RVB (1024 échantillons) d'une trace entre ``low`` et ``high`` Å."""
    grid = np.linspace(low, high, 1024)
    if trace is None:
        return np.zeros((1024, 3))
    wl = np.asarray(trace.wavelengths, dtype=np.float64)
    flux = np.asarray(trace.flux, dtype=np.float64)
    valid = np.isfinite(wl) & np.isfinite(flux)
    if not np.any(valid):
        return np.zeros((1024, 3))
    wl, flux = wl[valid], flux[valid]
    wl, unique = np.unique(wl, return_index=True)  # trié et sans doublon
    flux = flux[unique]
    sampled = np.interp(grid, wl, flux, left=0.0, right=0.0)
    return _visible_spectrum_rgb(grid, sampled)


def _clamp_size(size_px: StageSize) -> StageSize:
    """Borne la taille du graphe pour éviter les figures minuscules ou énormes."""
    return (
        min(max(float(size_px[0]), MIN_PLOT_SIZE[0]), MAX_PLOT_SIZE[0]),
        min(max(float(size_px[1]), MIN_PLOT_SIZE[1]), MAX_PLOT_SIZE[1]),
    )


def spectrum_png_data_uri(
    wavelengths: ArrayLike | None = None,
    flux: ArrayLike | None = None,
    **kwargs,
) -> str:
    """Rendu PNG (data URI) d'un spectre ; mêmes options que `_render_spectrum_plot`."""
    uri, _limits = _render_spectrum_plot(wavelengths, flux, **kwargs)
    return uri


def _render_spectrum_plot(
    wavelengths: ArrayLike | None = None,
    flux: ArrayLike | None = None,
    *,
    line_size: int = 1,
    line_type: str = "-",
    traces: tuple[SpectrumTrace, ...] = (),
    x_limits: WavelengthLimits | None = None,
    show_color_band: bool = False,
    show_element_lines: bool = False,
    size_px: StageSize = DEFAULT_PLOT_SIZE,
) -> tuple[str, tuple[float, float]]:
    """Dessine le graphe et renvoie ``(data URI PNG, (gauche, droite))`` des axes.

    Args:
        wavelengths: Longueurs d'onde (Å) d'un spectre isolé optionnel.
        flux: Flux correspondant.
        line_size: Épaisseur de trait (> 0).
        line_type: Style de trait matplotlib.
        traces: Spectres colorés à superposer.
        x_limits: Plage de longueurs d'onde visible (zoom).
        show_color_band: Affiche la bande de couleurs sous le graphe.
        show_element_lines: Marque les raies H/He/Na/Mg/Ca.
        size_px: Taille de l'image produite en pixels (celle de la zone d'affichage).

    Returns:
        Le PNG et les bornes horizontales des axes (fractions de la figure).

    Raises:
        ValueError: Si un paramètre est invalide.
    """
    started = time.perf_counter()
    if line_size <= 0:
        raise ValueError("line_size must be greater than zero.")
    if line_type not in VALID_LINE_TYPES:
        raise ValueError("line_type must be a supported Matplotlib line style.")
    if x_limits is not None and (
        not all(math.isfinite(v) for v in x_limits) or x_limits[0] >= x_limits[1]
    ):
        raise ValueError("Wavelength limits must be finite and increasing.")

    width_px, height_px = _clamp_size(size_px)
    figure = Figure(figsize=(width_px / PLOT_DPI, height_px / PLOT_DPI), dpi=PLOT_DPI, facecolor=_BG)
    if show_color_band:
        grid = figure.add_gridspec(2, 1, height_ratios=(12, 1), hspace=0.04)
        axes = figure.add_subplot(grid[0])
        color_axes = figure.add_subplot(grid[1], sharex=axes)
    else:
        axes = figure.add_subplot(111)
        color_axes = None
    axes.set_facecolor(_BG)
    axes.set_title("Spectre SA100", color="#f9fafb")
    axes.set_ylabel("Flux", color="#e5e7eb")
    axes.tick_params(colors="#d1d5db")
    axes.grid(color="#374151", alpha=0.8)
    for spine in axes.spines.values():
        spine.set_color("#6b7280")

    plot_traces = list(traces)
    if wavelengths is not None and flux is not None:
        x_values, y_values = _as_1d_pair(wavelengths, flux)
        plot_traces.append(SpectrumTrace(x_values, y_values, "#38bdf8", "Spectre"))

    plotted = 0
    for trace in plot_traces:
        x_values, y_values = _as_1d_pair(trace.wavelengths, trace.flux)
        if x_limits is not None:
            visible = (x_values >= x_limits[0]) & (x_values <= x_limits[1])
            x_values, y_values = x_values[visible], y_values[visible]
        if x_values.size:
            axes.plot(
                x_values, y_values, color=trace.color,
                linewidth=line_size, linestyle=line_type, label=trace.label,
            )
            plotted += 1
    if plotted > 1:
        axes.legend(facecolor="#1f2937", edgecolor="#6b7280", labelcolor="#e5e7eb")
    if x_limits is not None:
        axes.set_xlim(*x_limits)

    marks = _draw_element_lines(axes) if show_element_lines else 0
    if color_axes is not None:
        low, high = axes.get_xlim()
        band = _color_band_values(plot_traces[-1] if plot_traces else None, low, high)
        color_axes.imshow(
            band[np.newaxis, :, :], extent=(low, high, 0, 1),
            aspect="auto", interpolation="nearest",
        )
        color_axes.set_xlim(low, high)
        color_axes.set_yticks([])
        color_axes.set_xlabel("Longueur d’onde (Å)", color="#e5e7eb")
        color_axes.tick_params(axis="x", colors="#d1d5db")
        color_axes.set_facecolor(_BG)
        for spine in color_axes.spines.values():
            spine.set_color("#6b7280")

    figure.subplots_adjust(left=0.11, right=0.97, bottom=0.16, top=0.91)
    left, _bottom, width, _height = axes.get_position().bounds
    output = BytesIO()
    FigureCanvasAgg(figure).print_png(output)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    logger.debug(
        "Graphe %dx%d px : %d trace(s), xlim=%s, bande couleur=%s, %d raie(s), %.0f ms",
        width_px, height_px, plotted, x_limits, show_color_band, marks,
        (time.perf_counter() - started) * 1000,
    )
    return f"data:image/png;base64,{encoded}", (left, left + width)

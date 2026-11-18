"""Flet interface for the QuickSpec SA100 processing engine."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from io import BytesIO
import json
import logging
import math
from pathlib import Path
import random
from typing import Callable, TypeAlias

import flet as ft
import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from PIL import Image as PillowImage

from image_processing import Image
from spectra_processing import Spectra

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")

MARKER_COLORS = ("#ff3030", "#22c55e", "#3b82f6")
MARKER_NAMES = ("ordre zéro", "trace du spectre", "fond de ciel")
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
MAX_SPECTRUM_TRACES = len(SPECTRUM_COLORS)
SPECTRUM_IMAGE_WIDTH = 770.0
SPECTRUM_IMAGE_HEIGHT = 440.0
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
    "-",
    "--",
    "-.",
    ":",
    "None",
    "none",
    " ",
    "",
    "solid",
    "dashed",
    "dashdot",
    "dotted",
}
LOG_LEVEL_COLORS = {
    logging.INFO: "#22c55e",
    logging.WARNING: "#facc15",
    logging.ERROR: "#ef4444",
    logging.CRITICAL: "#ef4444",
}
APPLICATION_LOGGERS = (
    logger,
    logging.getLogger("image_io"),
    logging.getLogger("image_processing"),
    logging.getLogger("spectra_processing"),
)
ImagePoint: TypeAlias = tuple[float, float]
StageSize: TypeAlias = tuple[float, float]
WavelengthLimits: TypeAlias = tuple[float, float]
_active_log_handler: logging.Handler | None = None


@dataclass(frozen=True)
class SpectrumTrace:
    """One extracted spectrum and its plot identity."""

    wavelengths: NDArray[np.float64]
    flux: NDArray[np.float32]
    color: str
    label: str


@dataclass(frozen=True)
class AppParameters:
    """Validated user-configurable parameters for spectrum processing and display."""

    star_analyser: int = 200
    focal_mm: int = 50
    pixel_size_um: float = 4.65
    lambda_min: int = 3500
    lambda_max: int = 7500
    line_size: int = 1
    line_type: str = "-"

    def __post_init__(self) -> None:
        """Reject invalid optical, wavelength, and plot-style settings."""
        for name, value in (
            ("Star Analyser", self.star_analyser),
            ("Focal", self.focal_mm),
            ("lambda_min", self.lambda_min),
            ("lambda_max", self.lambda_max),
            ("line_size", self.line_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer.")
        if self.star_analyser not in (100, 200):
            raise ValueError("Star Analyser must be 100 or 200 traits/mm.")
        if self.focal_mm <= 0:
            raise ValueError("Focal must be an integer greater than zero.")
        if (
            isinstance(self.pixel_size_um, bool)
            or not isinstance(self.pixel_size_um, (int, float))
            or not math.isfinite(self.pixel_size_um)
            or self.pixel_size_um <= 0
        ):
            raise ValueError("Pixel_size must be a finite number greater than zero.")
        if self.lambda_min <= 0 or self.lambda_max <= 0:
            raise ValueError("lambda_min and lambda_max must be greater than zero.")
        if self.lambda_max <= self.lambda_min:
            raise ValueError("lambda_max must be greater than lambda_min.")
        if self.line_size <= 0:
            raise ValueError("line_size must be an integer greater than zero.")
        if self.line_type not in VALID_LINE_TYPES:
            raise ValueError(
                "line_type must be a supported Matplotlib line style."
            )

    @classmethod
    def from_mapping(cls, values: object) -> AppParameters:
        """Create validated parameters from a decoded JSON object.

        Args:
            values: Mapping of parameter names to JSON-compatible values.

        Returns:
            A validated parameter set.

        Raises:
            ValueError: If the value is not a valid parameter mapping.
        """
        if not isinstance(values, dict):
            raise ValueError("The saved parameters must be a JSON object.")
        try:
            star_analyser = values["star_analyser"]
            focal_mm = values["focal_mm"]
            pixel_size_um = values["pixel_size_um"]
            lambda_min = values["lambda_min"]
            lambda_max = values["lambda_max"]
            line_size = values["line_size"]
            line_type = values["line_type"]
        except KeyError as exc:
            raise ValueError("The saved parameter values are invalid.") from exc
        if isinstance(star_analyser, bool) or not isinstance(star_analyser, int):
            raise ValueError("star_analyser must be a JSON integer.")
        if isinstance(focal_mm, bool) or not isinstance(focal_mm, int):
            raise ValueError("focal_mm must be a JSON integer.")
        if isinstance(lambda_min, bool) or not isinstance(lambda_min, int):
            raise ValueError("lambda_min must be a JSON integer.")
        if isinstance(lambda_max, bool) or not isinstance(lambda_max, int):
            raise ValueError("lambda_max must be a JSON integer.")
        if isinstance(line_size, bool) or not isinstance(line_size, int):
            raise ValueError("Saved integer parameters must be JSON integers.")
        if (
            isinstance(pixel_size_um, bool)
            or not isinstance(pixel_size_um, (int, float))
        ):
            raise ValueError("Saved pixel_size must be a JSON number.")
        if not isinstance(line_type, str):
            raise ValueError("Saved line_type must be text.")
        return cls(
            star_analyser=star_analyser,
            focal_mm=focal_mm,
            pixel_size_um=pixel_size_um,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            line_size=line_size,
            line_type=line_type,
        )


class FletLogHandler(logging.Handler):
    """Forward formatted log records to a callback owned by the Flet view."""

    def __init__(self, on_record: Callable[[logging.LogRecord], None]) -> None:
        """Create a handler that forwards each log record to ``on_record``.

        Args:
            on_record: Callback that displays one log record in the UI.
        """
        super().__init__(level=logging.INFO)
        self.on_record = on_record
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        """Forward a log record to the configured callback.

        Args:
            record: Log record emitted by the application or its dependencies.
        """
        self.on_record(record)


def image_bounds(
    image_shape: tuple[int, int], stage_size: StageSize
) -> tuple[float, float, float, float]:
    """Calculate the image rectangle for a contain-fitted viewport.

    Args:
        image_shape: Image dimensions as ``(height, width)`` in pixels.
        stage_size: Viewport dimensions as ``(width, height)`` in logical pixels.

    Returns:
        The displayed rectangle as ``(left, top, width, height)``.

    Raises:
        ValueError: If any image or viewport dimension is not positive.
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


def image_point_from_local(
    local_position: ft.Offset | None,
    image_shape: tuple[int, int],
    stage_size: StageSize,
) -> ImagePoint | None:
    """Map a viewport-local position to image pixels.

    Args:
        local_position: Position relative to the image viewport, or ``None``.
        image_shape: Image dimensions as ``(height, width)`` in pixels.
        stage_size: Viewport dimensions as ``(width, height)`` in logical pixels.

    Returns:
        Image coordinates as ``(x, y)``, or ``None`` outside the displayed image.
    """
    if local_position is None:
        return None
    image_height, image_width = image_shape
    left, top, display_width, display_height = image_bounds(image_shape, stage_size)
    x, y = float(local_position.x), float(local_position.y)
    if not (left <= x <= left + display_width and top <= y <= top + display_height):
        return None
    return (
        (x - left) / display_width * image_width,
        (y - top) / display_height * image_height,
    )


def image_png_data_uri(data: ArrayLike) -> str:
    """Encode a contrast-stretched grayscale preview as a PNG data URI.

    Args:
        data: Two-dimensional image pixels; the input array is not modified.

    Returns:
        A base64-encoded PNG data URI suitable for an Flet image control.

    Raises:
        ValueError: If the input is not a 2D array or has no finite pixels.
    """
    image = np.asarray(data, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("The image preview requires a 2D grayscale array.")
    finite = image[np.isfinite(image)]
    if not finite.size:
        raise ValueError("The image contains no finite grayscale pixels.")
    low, high = np.percentile(finite, (0.5, 99.5))
    if high <= low:
        high = low + 1
    preview = np.nan_to_num(image, nan=low, posinf=high, neginf=low)
    preview = np.clip((preview - low) * (255.0 / (high - low)), 0, 255)
    output = BytesIO()
    PillowImage.fromarray(preview.astype(np.uint8)).save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def spectrum_png_data_uri(
    wavelengths: ArrayLike | None = None,
    flux: ArrayLike | None = None,
    *,
    line_size: int = 1,
    line_type: str = "-",
    traces: tuple[SpectrumTrace, ...] = (),
    x_limits: WavelengthLimits | None = None,
    show_color_band: bool = False,
    show_element_lines: bool = False,
) -> str:
    """Render a labeled SA100 spectrum as a Matplotlib PNG data URI.

    Args:
        wavelengths: Optional one-dimensional wavelength values in Angstroms.
        flux: Optional one-dimensional flux values matching ``wavelengths``.
        line_size: Positive Matplotlib line width.
        line_type: Matplotlib line style string.
        traces: Optional set of color-coded spectra to draw together.
        x_limits: Optional visible wavelength limits for zoomed rendering.
        show_color_band: Whether to show wavelength hues below the spectrum.
        show_element_lines: Whether to mark common H, He, Na, Mg, and Ca lines.

    Returns:
        A base64-encoded PNG data URI for display in an Flet image control.

    Raises:
        ValueError: If array dimensions, line style, or wavelength limits are invalid.
    """
    image_uri, _axes_x_limits = _render_spectrum_plot(
        wavelengths,
        flux,
        line_size=line_size,
        line_type=line_type,
        traces=traces,
        x_limits=x_limits,
        show_color_band=show_color_band,
        show_element_lines=show_element_lines,
    )
    return image_uri


def _visible_spectrum_rgb(
    wavelengths: ArrayLike,
    intensities: ArrayLike,
) -> NDArray[np.float64]:
    """Map visible wavelengths to RGB and scale brightness by intensity.

    Args:
        wavelengths: One-dimensional wavelength samples in Angstroms.
        intensities: Matching spectral intensity samples.

    Returns:
        RGB values in the range from zero to one for each wavelength sample.

    Raises:
        ValueError: If the wavelength and intensity arrays are not matching 1D arrays.
    """
    wavelength_values = np.asarray(wavelengths, dtype=np.float64)
    intensity_values = np.asarray(intensities, dtype=np.float64)
    if (
        wavelength_values.ndim != 1
        or intensity_values.ndim != 1
        or wavelength_values.size != intensity_values.size
    ):
        raise ValueError("Wavelength and intensity arrays must be matching 1D arrays.")

    color_wavelengths = np.array(
        [3800, 4400, 4900, 5100, 5800, 6000, 7000, 7500],
        dtype=np.float64,
    )
    color_values = np.array(
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
    colors = np.column_stack(
        [
            np.interp(
                wavelength_values,
                color_wavelengths,
                color_values[:, channel],
                left=0.0,
                right=0.0,
            )
            for channel in range(3)
        ]
    )
    positive_intensities = np.maximum(intensity_values, 0.0)
    max_intensity = float(np.max(positive_intensities, initial=0.0))
    if max_intensity > 0:
        colors *= (positive_intensities / max_intensity)[:, np.newaxis]
    else:
        colors.fill(0.0)
    return colors


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
) -> tuple[str, tuple[float, float]]:
    """Render the graph and report the horizontal axes bounds.

    Args:
        wavelengths: Optional one-dimensional wavelength values in Angstroms.
        flux: Optional one-dimensional flux values matching ``wavelengths``.
        line_size: Positive Matplotlib line width.
        line_type: Matplotlib line style string.
        traces: Color-coded spectra to draw together.
        x_limits: Optional visible wavelength limits for zoomed rendering.
        show_color_band: Whether to show wavelength hues below the spectrum.
        show_element_lines: Whether to mark common H, He, Na, Mg, and Ca lines.

    Returns:
        The PNG data URI and normalized axes left and right bounds.

    Raises:
        ValueError: If array dimensions, line style, or wavelength limits are invalid.
    """
    if line_size <= 0:
        raise ValueError("line_size must be greater than zero.")
    if line_type not in VALID_LINE_TYPES:
        raise ValueError("line_type must be a supported Matplotlib line style.")
    if x_limits is not None and (
        not all(math.isfinite(value) for value in x_limits)
        or x_limits[0] >= x_limits[1]
    ):
        raise ValueError("Wavelength limits must be finite and increasing.")
    figure = Figure(figsize=(7, 4), dpi=110, facecolor="#111827")
    if show_color_band:
        grid = figure.add_gridspec(2, 1, height_ratios=(12, 1), hspace=0.04)
        axes = figure.add_subplot(grid[0])
        color_axes = figure.add_subplot(grid[1], sharex=axes)
    else:
        axes = figure.add_subplot(111)
        color_axes = None
    axes.set_facecolor("#111827")
    axes.set_title("Spectre SA100", color="#f9fafb")
    axes.set_xlabel("Longueur d’onde (Å)", color="#e5e7eb")
    axes.set_ylabel("Flux", color="#e5e7eb")
    axes.tick_params(colors="#d1d5db")
    axes.grid(color="#374151", alpha=0.8)
    for spine in axes.spines.values():
        spine.set_color("#6b7280")

    plot_traces = list(traces)
    if wavelengths is not None and flux is not None:
        x_values = np.asarray(wavelengths, dtype=np.float64)
        y_values = np.asarray(flux, dtype=np.float64)
        if x_values.ndim != 1 or y_values.ndim != 1 or x_values.size != y_values.size:
            raise ValueError("Wavelength and flux arrays must be matching 1D arrays.")
        plot_traces.append(
            SpectrumTrace(x_values, y_values, "#38bdf8", "Spectre")
        )

    plotted_count = 0
    for trace in plot_traces:
        x_values = np.asarray(trace.wavelengths, dtype=np.float64)
        y_values = np.asarray(trace.flux, dtype=np.float64)
        if x_values.ndim != 1 or y_values.ndim != 1 or x_values.size != y_values.size:
            raise ValueError("Wavelength and flux arrays must be matching 1D arrays.")
        if x_limits is not None:
            visible = (x_values >= x_limits[0]) & (x_values <= x_limits[1])
            x_values, y_values = x_values[visible], y_values[visible]
        if x_values.size:
            axes.plot(
                x_values,
                y_values,
                color=trace.color,
                linewidth=line_size,
                linestyle=line_type,
                label=trace.label,
            )
            plotted_count += 1
    if plotted_count > 1:
        axes.legend(facecolor="#1f2937", edgecolor="#6b7280", labelcolor="#e5e7eb")
    if x_limits is not None:
        axes.set_xlim(*x_limits)

    if show_element_lines:
        visible_low, visible_high = axes.get_xlim()
        displayed_lines = [
            line for line in ELEMENT_LINES if visible_low <= line[0] <= visible_high
        ]
        for index, (wavelength, label, element) in enumerate(displayed_lines):
            color = ELEMENT_LINE_COLORS[element]
            axes.axvline(
                wavelength,
                color=color,
                linestyle="--",
                alpha=0.65,
                linewidth=0.8,
            )
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
    if color_axes is not None:
        visible_low, visible_high = axes.get_xlim()
        color_wavelengths = np.linspace(visible_low, visible_high, 1024)
        color_values = np.zeros((1024, 3), dtype=np.float64)
        if plot_traces:
            latest_trace = plot_traces[-1]
            trace_wavelengths = np.asarray(
                latest_trace.wavelengths,
                dtype=np.float64,
            )
            trace_intensities = np.asarray(latest_trace.flux, dtype=np.float64)
            valid = np.isfinite(trace_wavelengths) & np.isfinite(trace_intensities)
            if np.any(valid):
                trace_wavelengths = trace_wavelengths[valid]
                trace_intensities = trace_intensities[valid]
                order = np.argsort(trace_wavelengths)
                trace_wavelengths = trace_wavelengths[order]
                trace_intensities = trace_intensities[order]
                unique_wavelengths, unique_indices = np.unique(
                    trace_wavelengths,
                    return_index=True,
                )
                trace_wavelengths = unique_wavelengths
                trace_intensities = trace_intensities[unique_indices]
                sampled_intensities = np.interp(
                    color_wavelengths,
                    trace_wavelengths,
                    trace_intensities,
                    left=0.0,
                    right=0.0,
                )
                color_values = _visible_spectrum_rgb(
                    color_wavelengths,
                    sampled_intensities,
                )
        color_axes.imshow(
            color_values[np.newaxis, :, :],
            extent=(visible_low, visible_high, 0, 1),
            aspect="auto",
            interpolation="nearest",
        )
        color_axes.set_xlim(visible_low, visible_high)
        color_axes.set_yticks([])
        color_axes.set_xlabel("Longueur d’onde (Å)", color="#e5e7eb")
        color_axes.tick_params(axis="x", colors="#d1d5db")
        color_axes.set_facecolor("#111827")
        for spine in color_axes.spines.values():
            spine.set_color("#6b7280")
    figure.subplots_adjust(left=0.11, right=0.97, bottom=0.16, top=0.91)
    axes_left, _axes_bottom, axes_width, _axes_height = axes.get_position().bounds
    output = BytesIO()
    FigureCanvasAgg(figure).print_png(output)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}", (axes_left, axes_left + axes_width)


class QuickSpecFletApp:
    """Build and coordinate the QuickSpec Flet interface.

    The instance owns the image and spectrum processors, UI controls, current
    image selection, and persisted file-picker directory.
    """

    def __init__(self) -> None:
        """Initialize processing state and restore the last image directory."""
        self.image_processor: Image = Image()
        self.spectra_processor: Spectra = Spectra()
        self.selection_points: list[ImagePoint] = []
        self.selecting: bool = False
        self.page: ft.Page | None = None
        self.image_shape: tuple[int, int] | None = None
        self.image_stack: ft.Stack | None = None
        self.image_viewer: ft.InteractiveViewer | None = None
        self.image_status: ft.Text | None = None
        self.spectrum_status: ft.Text | None = None
        self.spectrum_zoom_status: ft.Text | None = None
        self.spectrum_image: ft.Image | None = None
        self.spectrum_gesture: ft.GestureDetector | None = None
        self.spectrum_color_checkbox: ft.Checkbox | None = None
        self.spectrum_lines_checkbox: ft.Checkbox | None = None
        self.parameter_status: ft.Text | None = None
        self.star_analyser_radio: ft.RadioGroup | None = None
        self.focal_field: ft.TextField | None = None
        self.pixel_size_field: ft.TextField | None = None
        self.lambda_min_field: ft.TextField | None = None
        self.lambda_max_field: ft.TextField | None = None
        self.line_size_field: ft.TextField | None = None
        self.line_type_field: ft.TextField | None = None
        self.log_view: ft.ListView | None = None
        self.debug_checkbox: ft.Checkbox | None = None
        self.log_handler: FletLogHandler | None = None
        self.picker: ft.FilePicker | None = None
        self.tabs: ft.Tabs | None = None
        self.location_file = Path.home() / ".quickspec-sa100" / "last-image-location.txt"
        self.parameters_file = Path.home() / ".quickspec-sa100" / "parameters.json"
        self.parameters = AppParameters()
        self.parameter_load_error = self._load_saved_parameters()
        self._configure_spectra_processor()
        self.spectrum_traces: list[SpectrumTrace] = []
        self.spectrum_xlim: WavelengthLimits | None = None
        self._spectrum_drag_start_x: float | None = None
        self._spectrum_plot_size: StageSize | None = None
        self._spectrum_axes_x_limits: tuple[float, float] = (0.0942, 0.9629)
        self._spectrum_sequence: int = 0
        self.current_image_label: str = "Image"
        self.last_directory: str = self._read_last_directory()
        self.stage_size: StageSize = (800.0, 480.0)

    def build(self, page: ft.Page) -> ft.Tabs:
        """Create the application tabs and attach them to a page.

        Args:
            page: Flet page that will host the application controls and services.

        Returns:
            The configured four-tab control.
        """
        self.page = page
        page.title = "QuickSpec SA100"
        page.theme_mode = ft.ThemeMode.DARK
        page.padding = 8
        page.on_resize = self._on_resize
        self.picker = ft.FilePicker()
        page.services.append(self.picker)

        self.image_status = ft.Text(
            "Ouvre une image, puis sélectionne l’ordre zéro, le spectre et le ciel.",
            size=13,
        )
        self.spectrum_status = ft.Text(
            "Sélectionne trois points sur l’image pour extraire le spectre.",
            size=13,
        )
        self._set_stage_size()
        self.image_stack = ft.Stack(
            width=self.stage_size[0],
            height=self.stage_size[1],
            alignment=ft.Alignment.TOP_LEFT,
            controls=[
                ft.Container(
                    content=ft.Text("Ouvre une image", color=ft.Colors.WHITE_70),
                    alignment=ft.Alignment.CENTER,
                    width=self.stage_size[0],
                    height=self.stage_size[1],
                )
            ],
        )
        image_gesture = ft.GestureDetector(
            content=self.image_stack,
            on_tap=self._on_image_tap,
            mouse_cursor=ft.MouseCursor.CLICK,
        )
        self.image_viewer = ft.InteractiveViewer(
            content=image_gesture,
            expand=True,
            min_scale=1,
            max_scale=12,
            pan_enabled=True,
            scale_enabled=True,
            constrained=True,
        )
        self.spectrum_image = ft.Image(
            src=spectrum_png_data_uri(),
            fit=ft.BoxFit.CONTAIN,
            expand=True,
        )
        self.spectrum_gesture = ft.GestureDetector(
            content=self.spectrum_image,
            on_pan_start=self._on_spectrum_pan_start,
            on_pan_end=self._on_spectrum_pan_end,
            on_size_change=self._on_spectrum_plot_size_change,
            mouse_cursor=ft.MouseCursor.CLICK,
        )
        self.spectrum_color_checkbox = ft.Checkbox(
            label="Couleurs",
            value=False,
            on_change=self._on_spectrum_display_option_changed,
        )
        self.spectrum_lines_checkbox = ft.Checkbox(
            label="Raies H/He/Na/Mg/Ca",
            value=False,
            on_change=self._on_spectrum_display_option_changed,
        )
        self.spectrum_zoom_status = ft.Text("", size=12)

        image_tab = ft.Column(
            expand=True,
            spacing=8,
            controls=[
                ft.Row(
                    wrap=True,
                    controls=[
                        ft.Button(
                            "Ouvrir une image",
                            icon=ft.Icons.FOLDER_OPEN,
                            on_click=self._open_picker,
                        ),
                        ft.Button(
                            "Sélection SA100",
                            icon=ft.Icons.TOUCH_APP,
                            on_click=self._begin_selection,
                        ),
                        ft.Text("Pince pour zoomer · glisse pour déplacer", size=12),
                    ],
                ),
                self.image_status,
                ft.Container(
                    expand=True,
                    bgcolor="#080b10",
                    border_radius=8,
                    clip_behavior=ft.ClipBehavior.HARD_EDGE,
                    content=self.image_viewer,
                ),
            ],
        )
        spectrum_tab = ft.Column(
            expand=True,
            spacing=8,
            controls=[
                ft.Row(
                    controls=[
                        ft.Text("Spectre SA100", size=20, weight=ft.FontWeight.BOLD),
                        ft.Row(
                            wrap=True,
                            controls=[
                                ft.Button(
                                    "Reset zoom",
                                    icon=ft.Icons.ZOOM_OUT_MAP,
                                    on_click=self._reset_spectrum_zoom,
                                ),
                                ft.Button(
                                    "Clear",
                                    icon=ft.Icons.DELETE_OUTLINE,
                                    on_click=self._clear_spectrum,
                                ),
                                self.spectrum_color_checkbox,
                                self.spectrum_lines_checkbox,
                            ],
                        ),
                    ],
                ),
                self.spectrum_status,
                ft.Text(
                    "Glisse horizontalement sur le graphe pour zoomer.",
                    size=12,
                    color=ft.Colors.WHITE_70,
                ),
                self.spectrum_zoom_status,
                ft.Container(
                    expand=True,
                    content=self.spectrum_gesture,
                ),
            ],
        )
        self.log_view = ft.ListView(
            expand=True,
            spacing=2,
            auto_scroll=True,
        )
        self.debug_checkbox = ft.Checkbox(
            label="Debug",
            value=False,
            on_change=self._on_debug_changed,
        )
        logs_tab = ft.Column(
            expand=True,
            spacing=8,
            controls=[
                ft.Row(
                    controls=[
                        ft.Text("Sortie du Logger", size=20, weight=ft.FontWeight.BOLD),
                        self.debug_checkbox,
                        ft.Button(
                            "Clear",
                            icon=ft.Icons.DELETE_OUTLINE,
                            on_click=self._clear_logs,
                        ),
                    ],
                ),
                ft.Container(
                    expand=True,
                    bgcolor="#080b10",
                    border_radius=8,
                    padding=8,
                    content=self.log_view,
                ),
            ],
        )
        self._build_parameter_controls()
        parameters_tab = ft.Column(
            expand=True,
            scroll=ft.ScrollMode.AUTO,
            spacing=12,
            controls=[
                ft.Text("Parameters", size=20, weight=ft.FontWeight.BOLD),
                ft.Text("StarAanalyser — réseau", size=14),
                self.star_analyser_radio,
                self.focal_field,
                self.pixel_size_field,
                self.lambda_min_field,
                self.lambda_max_field,
                self.line_size_field,
                self.line_type_field,
                ft.Text(
                    "Styles Matplotlib : -, --, -., :, solid, dashed, dashdot, dotted",
                    size=12,
                    color=ft.Colors.WHITE_70,
                ),
                ft.Text(
                    "Focal sera utilisée pour la calibration de l’étape 4.",
                    size=12,
                    color=ft.Colors.WHITE_70,
                ),
                ft.Row(
                    wrap=True,
                    controls=[
                        ft.Button(
                            "Apply and save",
                            icon=ft.Icons.SAVE_OUTLINED,
                            on_click=self._apply_and_save_parameters,
                        ),
                        ft.Button(
                            "Undo",
                            icon=ft.Icons.UNDO,
                            on_click=self._undo_parameters,
                        ),
                    ],
                ),
                self.parameter_status,
            ],
        )
        tabs = ft.Tabs(
            length=4,
            selected_index=0,
            expand=True,
            content=ft.Column(
                expand=True,
                controls=[
                    ft.TabBar(
                        tabs=[
                            ft.Tab(label=ft.Text("Image"), icon=ft.Icons.IMAGE),
                            ft.Tab(label=ft.Text("Spectre"), icon=ft.Icons.SHOW_CHART),
                            ft.Tab(label=ft.Text("Logs"), icon=ft.Icons.TERMINAL),
                            ft.Tab(
                                label=ft.Text("Parameters"),
                                icon=ft.Icons.SETTINGS,
                            ),
                        ]
                    ),
                    ft.TabBarView(
                        expand=True,
                        controls=[image_tab, spectrum_tab, logs_tab, parameters_tab],
                    ),
                ],
            ),
        )
        self.tabs = tabs
        self._attach_log_handler()
        page.add(ft.SafeArea(expand=True, content=tabs))
        return tabs

    def _build_parameter_controls(self) -> None:
        """Create parameter widgets initialized from the last applied settings."""
        self.star_analyser_radio = ft.RadioGroup(
            value=str(self.parameters.star_analyser),
            content=ft.Row(
                controls=[
                    ft.Radio(value="100", label="100 traits/mm"),
                    ft.Radio(value="200", label="200 traits/mm"),
                ]
            ),
        )
        self.focal_field = ft.TextField(
            label="Focal (mm)",
            value=str(self.parameters.focal_mm),
            keyboard_type=ft.KeyboardType.NUMBER,
        )
        self.pixel_size_field = ft.TextField(
            label="Pixel_size (µm)",
            value=str(self.parameters.pixel_size_um),
            keyboard_type=ft.KeyboardType.TEXT,
        )
        self.lambda_min_field = ft.TextField(
            label="lambda_min (Å)",
            value=str(self.parameters.lambda_min),
            keyboard_type=ft.KeyboardType.NUMBER,
        )
        self.lambda_max_field = ft.TextField(
            label="lambda_max (Å)",
            value=str(self.parameters.lambda_max),
            keyboard_type=ft.KeyboardType.NUMBER,
        )
        self.line_size_field = ft.TextField(
            label="line_size",
            value=str(self.parameters.line_size),
            keyboard_type=ft.KeyboardType.NUMBER,
        )
        self.line_type_field = ft.TextField(
            label="line_type (ls=...)",
            value=self.parameters.line_type,
            keyboard_type=ft.KeyboardType.TEXT,
        )
        initial_status = self.parameter_load_error or "Valeurs prêtes à modifier."
        self.parameter_status = ft.Text(initial_status, size=13)

    def _load_saved_parameters(self) -> str | None:
        """Load and validate persisted parameters, reporting malformed settings.

        Returns:
            A user-visible error message, or ``None`` when loading succeeded.
        """
        try:
            raw_parameters = json.loads(
                self.parameters_file.read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            logger.exception("Unable to read saved application parameters")
            return f"Impossible de lire les paramètres enregistrés : {exc}"

        try:
            self.parameters = AppParameters.from_mapping(raw_parameters)
        except ValueError as exc:
            logger.exception("Saved application parameters are invalid")
            return f"Paramètres enregistrés invalides : {exc}"
        return None

    def _configure_spectra_processor(self) -> None:
        """Apply the current optical and wavelength settings to the processor."""
        self.spectra_processor.pixel_size_um = self.parameters.pixel_size_um
        self.spectra_processor.grating_period_nm = (
            1_000_000.0 / self.parameters.star_analyser
        )
        self.spectra_processor.lambda_min = float(self.parameters.lambda_min)
        self.spectra_processor.lambda_max = float(self.parameters.lambda_max)

    def _read_parameters_from_controls(self) -> AppParameters:
        """Parse and validate the parameter values currently shown in the form.

        Returns:
            A validated parameter set ready to apply and save.

        Raises:
            ValueError: If a field is missing, malformed, or outside its range.
        """
        if (
            self.star_analyser_radio is None
            or self.focal_field is None
            or self.pixel_size_field is None
            or self.lambda_min_field is None
            or self.lambda_max_field is None
            or self.line_size_field is None
            or self.line_type_field is None
        ):
            raise ValueError("Le formulaire des paramètres n’est pas disponible.")
        try:
            return AppParameters(
                star_analyser=int(self.star_analyser_radio.value or ""),
                focal_mm=int(self.focal_field.value or ""),
                pixel_size_um=float(self.pixel_size_field.value or ""),
                lambda_min=int(self.lambda_min_field.value or ""),
                lambda_max=int(self.lambda_max_field.value or ""),
                line_size=int(self.line_size_field.value or ""),
                line_type=self.line_type_field.value or "",
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Vérifie les valeurs saisies : {exc}") from exc

    def _write_saved_parameters(self, parameters: AppParameters) -> None:
        """Atomically persist one validated parameter set as JSON.

        Args:
            parameters: Validated settings to write.

        Raises:
            OSError: If the settings file cannot be written.
        """
        self.parameters_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = self.parameters_file.with_name(
            f"{self.parameters_file.name}.tmp"
        )
        temporary_file.write_text(
            json.dumps(asdict(parameters), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary_file.replace(self.parameters_file)

    def _set_parameter_fields(self, parameters: AppParameters) -> None:
        """Populate the form with an applied set of parameters.

        Args:
            parameters: Values to display in the parameter widgets.
        """
        if (
            self.star_analyser_radio is None
            or self.focal_field is None
            or self.pixel_size_field is None
            or self.lambda_min_field is None
            or self.lambda_max_field is None
            or self.line_size_field is None
            or self.line_type_field is None
        ):
            return
        self.star_analyser_radio.value = str(parameters.star_analyser)
        self.focal_field.value = str(parameters.focal_mm)
        self.pixel_size_field.value = str(parameters.pixel_size_um)
        self.lambda_min_field.value = str(parameters.lambda_min)
        self.lambda_max_field.value = str(parameters.lambda_max)
        self.line_size_field.value = str(parameters.line_size)
        self.line_type_field.value = parameters.line_type

    def _set_parameter_status(self, message: str) -> None:
        """Display a parameter operation result in the Parameters tab.

        Args:
            message: User-visible result or validation error.
        """
        if self.parameter_status is not None:
            self.parameter_status.value = message
            if self.page is not None:
                self.page.update()

    def _apply_and_save_parameters(self, _event: ft.ControlEvent) -> None:
        """Validate, persist, and apply the parameter form values.

        Args:
            _event: Flet click event from the Apply and save button.
        """
        try:
            parameters = self._read_parameters_from_controls()
        except ValueError as exc:
            logger.warning("Rejected invalid application parameters: %s", exc)
            self._set_parameter_status(str(exc))
            return
        try:
            self._write_saved_parameters(parameters)
        except OSError as exc:
            logger.exception("Unable to save application parameters")
            self._set_parameter_status(f"Échec de l’enregistrement : {exc}")
            return

        self.parameters = parameters
        self._configure_spectra_processor()
        self._set_parameter_status("Paramètres appliqués et enregistrés.")
        logger.info("Application parameters saved")
        if (
            self.spectra_processor.result is not None
            and self.image_processor.data is not None
            and len(self.selection_points) == 3
        ):
            self._extract_spectrum(
                navigate_to_spectrum=False,
                replace_latest=True,
            )
        else:
            self._render_spectrum()

    def _undo_parameters(self, _event: ft.ControlEvent) -> None:
        """Restore the form to the most recently applied parameter values.

        Args:
            _event: Flet click event from the Undo button.
        """
        self._set_parameter_fields(self.parameters)
        self._set_parameter_status("Modifications annulées.")

    def _attach_log_handler(self) -> None:
        """Route application log records to the current Logs tab."""
        global _active_log_handler

        root_logger = logging.getLogger()
        if _active_log_handler is not None:
            root_logger.removeHandler(_active_log_handler)
            _active_log_handler.close()
        self.log_handler = FletLogHandler(self._append_log_record)
        root_logger.addHandler(self.log_handler)
        _active_log_handler = self.log_handler
        self._set_debug_logging(False)

    def _on_debug_changed(self, _event: ft.ControlEvent) -> None:
        """Enable or disable detailed application logging from the checkbox.

        Args:
            _event: Flet change event from the Debug checkbox.
        """
        self._set_debug_logging(
            bool(self.debug_checkbox and self.debug_checkbox.value)
        )

    def _set_debug_logging(self, enabled: bool) -> None:
        """Set application loggers and the UI handler to the selected verbosity.

        Args:
            enabled: Whether DEBUG and more detailed TRACE records are enabled.
        """
        level = TRACE_LEVEL if enabled else logging.INFO
        if self.log_handler is not None:
            self.log_handler.setLevel(level)
        for application_logger in APPLICATION_LOGGERS:
            application_logger.setLevel(level)

    def _append_log_record(self, record: logging.LogRecord) -> None:
        """Append one formatted log record using its severity color.

        Args:
            record: Log record to display in the Logs tab.
        """
        if self.log_view is None or self.log_handler is None:
            return
        color = LOG_LEVEL_COLORS.get(record.levelno, "#ffffff")
        self.log_view.controls.append(
            ft.Text(
                self.log_handler.format(record),
                color=color,
                font_family="monospace",
                selectable=True,
                size=15,
            )
        )
        if self.page is not None:
            self.page.update()

    def _clear_logs(self, _event: ft.ControlEvent | None = None) -> None:
        """Remove all displayed log records.

        Args:
            _event: Optional click event from the Logs tab's clear button.
        """
        if self.log_view is None:
            return
        self.log_view.controls.clear()
        if self.page is not None:
            self.page.update()

    def _set_stage_size(self) -> None:
        """Update the image viewport size from the current page dimensions."""
        page_width = self.page.width if self.page else None
        page_height = self.page.height if self.page else None
        self.stage_size = (
            max(float(page_width or 800) - 32, 280),
            max(float(page_height or 700) - 220, 220),
        )

    def _on_resize(self, _event: ft.PageResizeEvent) -> None:
        """Resize the image viewport after the application window changes size.

        Args:
            _event: Flet resize event; dimensions are read from the page itself.
        """
        self._set_stage_size()
        if self.image_stack is None:
            return
        self.image_stack.width, self.image_stack.height = self.stage_size
        self._refresh_image_controls()

    async def _open_picker(self, _event: ft.ControlEvent) -> None:
        """Open the supported-image picker and load the selected file.

        Args:
            _event: Flet click event from the open-image button.
        """
        try:
            selected = await self.picker.pick_files(
                dialog_title=                "Choisir une image",
                initial_directory=self.last_directory,
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=[
                    "png",
                    "jpg",
                    "jpeg",
                    "tif",
                    "tiff",
                    "fit",
                    "fits",
                    "fts",
                    "gz",
                ],
                allow_multiple=False,
            )
            if not selected:
                return
            image_path = selected[0].path
            if not image_path:
                raise ValueError("Le sélecteur n’a pas fourni de chemin de fichier.")
            path = Path(image_path)
            self.last_directory = str(path.parent)
            self._save_last_directory()
            self._load_image(path)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.exception("Unable to open the selected image")
            self._show_image_status(f"Erreur d’ouverture : {exc}")

    def _load_image(self, path: str | Path) -> None:
        """Load an image and reset its selection without clearing plotted spectra.

        Args:
            path: Path to a supported raster or FITS image.
        """
        data = self.image_processor.load([path])
        self.image_shape = data.shape
        self.current_image_label = Path(path).name
        self.selection_points.clear()
        self.selecting = False
        self.spectra_processor.result = None
        self.spectrum_status.value = self._spectrum_summary()
        self._show_image_status(
            f"{path.name} — {data.shape[1]} × {data.shape[0]} px"
        )
        self._refresh_image_controls(data)
        self._update_spectrum_image()
        if self.page is not None:
            self.page.update()

    def _refresh_image_controls(self, data: ArrayLike | None = None) -> None:
        """Refresh the image preview and overlay selected-point markers.

        Args:
            data: Optional 2D image to display; defaults to the loaded image.
        """
        if data is None:
            data = self.image_processor.data
        if data is None or self.image_shape is None:
            return
        stage_width, stage_height = self.stage_size
        controls = [
            ft.Image(
                src=image_png_data_uri(data),
                width=stage_width,
                height=stage_height,
                fit=ft.BoxFit.CONTAIN,
                gapless_playback=True,
            )
        ]
        left, top, display_width, display_height = image_bounds(
            self.image_shape, self.stage_size
        )
        for index, (x, y) in enumerate(self.selection_points):
            controls.append(
                ft.Icon(
                    ft.Icons.CLOSE,
                    color=MARKER_COLORS[index],
                    size=22,
                    left=left + x / self.image_shape[1] * display_width - 11,
                    top=top + y / self.image_shape[0] * display_height - 11,
                )
            )
        self.image_stack.width = stage_width
        self.image_stack.height = stage_height
        self.image_stack.controls = controls
        self.image_stack.update()

    def _begin_selection(self, _event: ft.ControlEvent | None = None) -> None:
        """Start a new three-point SA100 selection.

        Args:
            _event: Optional click event from the selection button.
        """
        if self.image_processor.data is None:
            self._show_image_status("Charge d’abord une image.")
            return
        self.selection_points.clear()
        self.selecting = True
        self._refresh_image_controls()
        self._show_image_status("Clique l’ordre zéro (1/3).")

    def _on_image_tap(self, event: ft.TapEvent) -> None:
        """Record one image point and extract the spectrum after the third tap.

        Args:
            event: Flet tap event containing the viewport-local position.
        """
        if not self.selecting or self.image_shape is None:
            return
        point = image_point_from_local(
            event.local_position,
            self.image_shape,
            self.stage_size,
        )
        if point is None:
            return
        self.selection_points.append(point)
        self._refresh_image_controls()
        if len(self.selection_points) < 3:
            self._show_image_status(
                f"Clique le {MARKER_NAMES[len(self.selection_points)]} "
                f"({len(self.selection_points) + 1}/3)."
            )
            return
        self._extract_spectrum()

    def _extract_spectrum(
        self,
        *,
        navigate_to_spectrum: bool = True,
        replace_latest: bool = False,
    ) -> None:
        """Extract one spectrum and add or replace its plotted trace.

        Args:
            navigate_to_spectrum: Whether to open the Spectrum tab after success.
            replace_latest: Replace the latest trace when reapplying parameters.
        """
        self.selecting = False
        try:
            result = self.spectra_processor.process(
                self.image_processor.data,
                *self.selection_points,
            )
        except (ValueError, RuntimeError) as exc:
            logger.exception("SA100 spectrum extraction failed")
            if self.spectrum_status is not None:
                self.spectrum_status.value = f"Extraction impossible : {exc}"
            self._show_image_status(f"Extraction impossible : {exc}")
            if self.page is not None:
                self.page.update()
            return

        self._add_spectrum_trace(
            result.wavelengths,
            result.flux,
            replace_latest=replace_latest,
        )
        self._render_spectrum()
        if self.spectrum_status is not None:
            self.spectrum_status.value = self._spectrum_summary(
                result.dispersion_angstrom_per_pixel
            )
        self._show_image_status("Spectre extrait. Sélectionne à nouveau pour recommencer.")
        if navigate_to_spectrum and self.page and self.tabs:
            self.tabs.selected_index = 1
            self.page.update()
        elif self.page is not None:
            self.page.update()

    def _add_spectrum_trace(
        self,
        wavelengths: ArrayLike,
        flux: ArrayLike,
        *,
        replace_latest: bool = False,
    ) -> None:
        """Add a color-coded spectrum, replacing the oldest if the plot is full.

        Args:
            wavelengths: Wavelength samples in Angstroms.
            flux: Matching sky-subtracted flux samples.
            replace_latest: Replace the current latest trace without changing its
                color or label.
        """
        x_values = np.asarray(wavelengths, dtype=np.float64).copy()
        y_values = np.asarray(flux, dtype=np.float32).copy()
        if x_values.ndim != 1 or y_values.ndim != 1 or x_values.size != y_values.size:
            raise ValueError("Wavelength and flux arrays must be matching 1D arrays.")

        if replace_latest and self.spectrum_traces:
            previous = self.spectrum_traces[-1]
            self.spectrum_traces[-1] = SpectrumTrace(
                x_values,
                y_values,
                previous.color,
                previous.label,
            )
            return

        if len(self.spectrum_traces) >= MAX_SPECTRUM_TRACES:
            self.spectrum_traces.pop(0)
        used_colors = {trace.color for trace in self.spectrum_traces}
        available_colors = [
            color for color in SPECTRUM_COLORS if color not in used_colors
        ]
        color = random.choice(available_colors)
        self._spectrum_sequence += 1
        label = f"{self._spectrum_sequence} — {self.current_image_label}"
        self.spectrum_traces.append(
            SpectrumTrace(x_values, y_values, color, label)
        )

    def _spectrum_summary(self, dispersion: float | None = None) -> str:
        """Return the status message for the current set of plotted spectra.

        Args:
            dispersion: Optional dispersion of the most recently extracted trace.

        Returns:
            A short status message describing the plotted spectra.
        """
        if not self.spectrum_traces:
            return "Charge une image et sélectionne trois points pour ajouter un spectre."
        details = f"{len(self.spectrum_traces)}/{MAX_SPECTRUM_TRACES} spectres affichés"
        if dispersion is not None:
            details += f" — dispersion {dispersion:.2f} Å/px"
        return details

    def _spectrum_data_limits(self) -> WavelengthLimits:
        """Compute the full visible wavelength range including plot margins.

        Returns:
            The plot's complete wavelength limits.
        """
        wavelengths = np.concatenate(
            [trace.wavelengths for trace in self.spectrum_traces]
        )
        low = float(np.min(wavelengths))
        high = float(np.max(wavelengths))
        if low == high:
            margin = max(abs(low) * 0.05, 1.0)
        else:
            margin = (high - low) * 0.05
        return low - margin, high + margin

    def _render_spectrum(self) -> None:
        """Render all plotted spectra with the current line style and zoom."""
        if self.spectrum_image is None:
            return
        self.spectrum_image.src, self._spectrum_axes_x_limits = (
            _render_spectrum_plot(
                line_size=self.parameters.line_size,
                line_type=self.parameters.line_type,
                traces=tuple(self.spectrum_traces),
                x_limits=self.spectrum_xlim,
                show_color_band=bool(
                    self.spectrum_color_checkbox
                    and self.spectrum_color_checkbox.value
                ),
                show_element_lines=bool(
                    self.spectrum_lines_checkbox
                    and self.spectrum_lines_checkbox.value
                ),
            )
        )
        self.spectrum_image.update()

    def _on_spectrum_display_option_changed(
        self,
        _event: ft.ControlEvent,
    ) -> None:
        """Regenerate the spectrum plot after a display option changes.

        Args:
            _event: Flet change event from either spectrum display checkbox.
        """
        self._render_spectrum()

    def _on_spectrum_plot_size_change(
        self, event: ft.LayoutSizeChangeEvent
    ) -> None:
        """Remember the rendered graph width for pointer-to-wavelength mapping.

        Args:
            event: Flet layout event reporting the graph's rendered dimensions.
        """
        if event.width > 0 and event.height > 0:
            self._spectrum_plot_size = (float(event.width), float(event.height))

    def _on_spectrum_pan_start(self, event: ft.DragStartEvent) -> None:
        """Record the start of a horizontal wavelength-range selection.

        Args:
            event: Flet drag-start event from the spectrum graph.
        """
        self._spectrum_drag_start_x = (
            float(event.local_position.x)
            if event.local_position is not None
            else None
        )

    def _on_spectrum_pan_end(self, event: ft.DragEndEvent) -> None:
        """Zoom to the wavelength interval selected by the user.

        Args:
            event: Flet drag-end event from the spectrum graph.
        """
        start_x = self._spectrum_drag_start_x
        end_x = (
            float(event.local_position.x)
            if event.local_position is not None
            else None
        )
        self._spectrum_drag_start_x = None
        if (
            start_x is None
            or end_x is None
            or self._spectrum_plot_size is None
            or not self.spectrum_traces
        ):
            return
        plot_width, plot_height = self._spectrum_plot_size
        image_scale = min(
            plot_width / SPECTRUM_IMAGE_WIDTH,
            plot_height / SPECTRUM_IMAGE_HEIGHT,
        )
        displayed_width = SPECTRUM_IMAGE_WIDTH * image_scale
        image_left = (plot_width - displayed_width) / 2
        start_fraction = (start_x - image_left) / displayed_width
        end_fraction = (end_x - image_left) / displayed_width
        axes_left, axes_right = self._spectrum_axes_x_limits
        if not (
            axes_left <= start_fraction <= axes_right
            and axes_left <= end_fraction <= axes_right
            and abs(end_fraction - start_fraction) * SPECTRUM_IMAGE_WIDTH >= 4
        ):
            return

        visible_low, visible_high = (
            self.spectrum_xlim or self._spectrum_data_limits()
        )
        plot_width = axes_right - axes_left
        start_value = visible_low + (start_fraction - axes_left) / plot_width * (
            visible_high - visible_low
        )
        end_value = visible_low + (end_fraction - axes_left) / plot_width * (
            visible_high - visible_low
        )
        new_limits = (min(start_value, end_value), max(start_value, end_value))
        if new_limits[0] >= new_limits[1]:
            return
        self.spectrum_xlim = new_limits
        self._render_spectrum()
        if self.spectrum_zoom_status is not None:
            self.spectrum_zoom_status.value = (
                f"Zoom : {new_limits[0]:.1f}–{new_limits[1]:.1f} Å"
            )
        if self.page is not None:
            self.page.update()

    def _reset_spectrum_zoom(self, _event: ft.ControlEvent) -> None:
        """Restore the full wavelength range and regenerate the spectrum plot.

        Args:
            _event: Flet click event from the Reset zoom button.
        """
        self.spectrum_xlim = None
        self._render_spectrum()
        if self.spectrum_zoom_status is not None:
            self.spectrum_zoom_status.value = ""
        if self.page is not None:
            self.page.update()

    def _clear_spectrum(self, _event: ft.ControlEvent | None = None) -> None:
        """Clear every plotted spectrum and restore the empty graph.

        Args:
            _event: Optional click event from the clear button.
        """
        self.spectra_processor.result = None
        self.spectrum_traces.clear()
        self.spectrum_xlim = None
        self.spectrum_status.value = self._spectrum_summary()
        if self.spectrum_zoom_status is not None:
            self.spectrum_zoom_status.value = ""
        self._update_spectrum_image()
        if self.page is not None:
            self.page.update()

    def _update_spectrum_image(self) -> None:
        """Regenerate the spectrum plot from the current set of traces."""
        self._render_spectrum()

    def _show_image_status(self, message: str) -> None:
        """Set the image-tab status message and update its control.

        Args:
            message: Status text to show to the user.
        """
        if self.image_status is not None:
            self.image_status.value = message
            self.image_status.update()

    def _read_last_directory(self) -> str:
        """Restore the saved image directory or fall back to the home folder.

        Returns:
            An existing directory path suitable for the file picker.
        """
        try:
            location = self.location_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            location = ""
        except OSError:
            logger.warning("Unable to read the saved image directory")
            location = ""
        if location and Path(location).is_dir():
            return location
        return str(Path.home())

    def _save_last_directory(self) -> None:
        """Persist the current file-picker directory."""
        try:
            self.location_file.parent.mkdir(parents=True, exist_ok=True)
            self.location_file.write_text(self.last_directory, encoding="utf-8")
        except OSError:
            logger.warning("Unable to save the image directory", exc_info=True)


def main(page: ft.Page) -> None:
    """Build the QuickSpec application on the Flet page.

    Args:
        page: Flet page supplied by the application runner.
    """
    QuickSpecFletApp().build(page)


if __name__ == "__main__":
    ft.run(main)

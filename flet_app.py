"""Interface Flet du moteur de traitement QuickSpec SA100."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import random
from typing import Callable, TypeAlias

import flet as ft
import numpy as np
from numpy.typing import ArrayLike

from calibration import load_masters
from image_processing import Image
from plotting import (  # noqa: F401  (certains noms sont ré-exportés pour les tests)
    DEFAULT_PLOT_SIZE,
    ELEMENT_LINE_COLORS,
    ELEMENT_LINES,
    MAX_SPECTRUM_TRACES,
    SPECTRUM_COLORS,
    VALID_LINE_TYPES,
    SpectrumTrace,
    AUTO_CUT_PERCENTILES,
    cut_limits,
    cut_percentile,
    cut_position,
    cut_sample,
    _render_spectrum_plot,
    _visible_spectrum_rgb,
    image_bounds,
    image_png_data_uri,
    spectrum_png_data_uri,
)
from settings import (
    CALIBRATION_FIELDS,
    PARAM_SPECS,
    AppParameters,
    load_parameters,
    save_parameters,
)
from response import apply_response, compute_response, load_response, save_response
from spectra_processing import Spectra, refine_centroid
from spectrum_io import SPECTRUM_EXTENSIONS, read_spectrum

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


MARKER_COLORS = ("#ff3030", "#22c55e", "#3b82f6")
MARKER_NAMES = ("ordre zéro", "trace du spectre", "fond de ciel")
IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "tif", "tiff", "fit", "fits", "fts", "gz"]
LOG_LEVEL_COLORS = {
    logging.INFO: "#22c55e",
    logging.WARNING: "#facc15",
    logging.ERROR: "#ef4444",
    logging.CRITICAL: "#ef4444",
}
APPLICATION_LOGGERS = (
    logger,
    logging.getLogger("calibration"),
    logging.getLogger("image_io"),
    logging.getLogger("image_processing"),
    logging.getLogger("spectra_processing"),
    logging.getLogger("plotting"),
    logging.getLogger("settings"),
)
BAND_COLORS = ("#ff2020", "#3b82f6")  # bande du spectre (rouge), bande de ciel (bleu)
BAND_LINE_PX = 1.0
SPECTRUM_RENDER_SCALE = 2.0  # PNG du graphe rendu 2× pour rester net au zoom
SECTION_NOTES = {
    "Réseau et optique": (
        "Réseau devant l’objectif : λ = d·sin(atan(x/f)), x = distance à l’ordre zéro. "
        "La taille de pixel est celle du capteur natif ; le binning 2×2 des JPEG/TIFF couleur "
        "est pris en compte automatiquement."
    ),
    "Réponse instrumentale": (
        "Le spectre extrait est divisé par cette réponse si le fichier existe dans le répertoire des "
        "masters. Elle se calcule dans l’onglet Spectre à partir d’une étoile de référence."
    ),
    "Extraction du spectre": (
        "Épaisseur totale des bandes de binning, centrées sur la trace du spectre "
        "et sur le point de fond de ciel."
    ),
    "Corrections offset / dark / flat": (
        "Appliquées à chaque image au chargement (extensions cherchées : fits, fit, fts, png, "
        "jpg, jpeg ; un nom vide désactive la correction ; si un master manque, un warning "
        "apparaît dans Logs). Le dark doit déjà être débarrassé de l’offset, et le flat de "
        "l’offset et du dark."
    ),
    "Affichage du spectre": "Styles Matplotlib : -, --, -., :, solid, dashed, dashdot, dotted",
}
SIZE_TOLERANCE_PX = 2.0  # variation de taille ignorée (évite de re-afficher pour rien)
ImagePoint: TypeAlias = tuple[float, float]
StageSize: TypeAlias = tuple[float, float]
_active_log_handler: logging.Handler | None = None


class FletLogHandler(logging.Handler):
    """Transmet les enregistrements de log à un rappel appartenant à la vue Flet."""

    def __init__(self, on_record: Callable[[logging.LogRecord], None]) -> None:
        super().__init__(level=logging.INFO)
        self.on_record = on_record
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%H:%M:%S"
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        self.on_record(record)


def image_point_from_local(
    local_position: ft.Offset | None,
    image_shape: tuple[int, int],
    stage_size: StageSize,
) -> ImagePoint | None:
    """Convertit une position locale de la zone d'affichage en pixels image.

    Returns:
        ``(x, y)`` en pixels image, ou ``None`` hors de l'image affichée.
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


def _size_changed(old: StageSize | None, new: StageSize) -> bool:
    """Vrai si la taille a réellement changé (au-delà de la tolérance)."""
    return (
        old is None
        or abs(old[0] - new[0]) >= SIZE_TOLERANCE_PX
        or abs(old[1] - new[1]) >= SIZE_TOLERANCE_PX
    )


class QuickSpecFletApp:
    """Construit et coordonne l'interface QuickSpec (onglets Image, Spectre, Parameters, Logs)."""

    def __init__(self) -> None:
        """Initialise les processeurs, restaure les paramètres et le dernier répertoire."""
        self.image_processor: Image = Image()
        self.spectra_processor: Spectra = Spectra()
        self.selection_points: list[ImagePoint] = []
        self.selecting = False
        self.page: ft.Page | None = None
        self.image_shape: tuple[int, int] | None = None
        self.image_stack: ft.Stack | None = None
        self.image_viewer: ft.InteractiveViewer | None = None
        self.image_status: ft.Text | None = None
        self.spectrum_status: ft.Text | None = None
        self.spectrum_image: ft.Image | None = None
        self.spectrum_stage: ft.Container | None = None
        self.spectrum_viewer: ft.InteractiveViewer | None = None
        self.spectrum_color_checkbox: ft.Checkbox | None = None
        self.spectrum_lines_checkbox: ft.Checkbox | None = None
        self.spectrum_normalize_checkbox: ft.Checkbox | None = None
        self.parameter_status: ft.Text | None = None
        # Contrôles du formulaire, créés depuis PARAM_SPECS : param_fields[nom] et
        # un attribut par paramètre (star_analyser_radio, focal_field, ...).
        self.param_fields: dict[str, ft.Control] = {}
        self.log_view: ft.ListView | None = None
        self.debug_checkbox: ft.Checkbox | None = None
        self.log_handler: FletLogHandler | None = None
        self.picker: ft.FilePicker | None = None
        self.tabs: ft.Tabs | None = None
        self.tab_pages: list[ft.Control] = []

        self.parameters_file = Path.home() / ".quickspec-sa100" / "parameters.json"
        self.parameters = AppParameters()
        self.parameter_load_error = self._load_saved_parameters()
        self._configure_spectra_processor()

        self.spectrum_traces: list[SpectrumTrace] = []
        self._spectrum_plot_size: StageSize | None = None  # zone d'affichage mesurée
        self._spectrum_sequence = 0
        self.current_image_label = "Image"
        # Dernier dossier utilisé pendant ce lancement seulement (dossier personnel au départ).
        self.last_directory: str = str(Path.home())
        self.stage_size: StageSize = (800.0, 480.0)  # zone image, remplacée par la mesure
        self._stage_measured = False
        self._previews: list[tuple[object, str]] = []  # cache (tableau source, data URI)
        self._cut_source: object | None = None  # image d'origine dont on a tiré l'échantillon
        self._cut_sample: np.ndarray | None = None
        self._display_limits: tuple[float, float] | None = None
        self.cut_positions: tuple[float, float] = self._auto_cut_positions()
        self.cut_slider: ft.RangeSlider | None = None
        self.cut_text: ft.Text | None = None
        self.show_rectified = False  # afficher l'image transformée + bandes de binning
        self.rectified_checkbox: ft.Checkbox | None = None

    # ------------------------------------------------------------------ construction

    def build(self, page: ft.Page) -> ft.Tabs:
        """Crée les quatre onglets (Image, Spectre, Parameters, Logs) et les attache à la page."""
        logger.info("Construction de l'interface (page %s × %s)", page.width, page.height)
        self.page = page
        page.title = "QuickSpec SA100"
        page.theme_mode = ft.ThemeMode.DARK
        page.padding = 8
        self.picker = ft.FilePicker()
        page.services.append(self.picker)

        self.tab_pages: list[ft.Control] = [
            self._build_image_tab(),
            self._build_spectrum_tab(),
            self._build_parameters_tab(),
            self._build_logs_tab(),
        ]
        for index, view in enumerate(self.tab_pages):
            view.expand = True
            view.visible = index == 0
        self.tabs = ft.Tabs(
            length=4,
            selected_index=0,
            expand=True,
            on_change=self._on_tab_change,
            content=ft.Column(
                expand=True,
                controls=[
                    ft.TabBar(
                        tabs=[
                            ft.Tab(label=ft.Text("Image"), icon=ft.Icons.IMAGE),
                            ft.Tab(label=ft.Text("Spectre"), icon=ft.Icons.SHOW_CHART),
                            ft.Tab(label=ft.Text("Parameters"), icon=ft.Icons.SETTINGS),
                            ft.Tab(label=ft.Text("Logs"), icon=ft.Icons.TERMINAL),
                        ]
                    ),
                    # Pas de TabBarView : il change d'onglet au glissement du doigt, ce qui
                    # gêne le zoom à deux doigts sur l'image. Seul un appui sur l'icône
                    # d'un onglet change maintenant de page (voir _apply_tab_visibility).
                    *self.tab_pages,
                ],
            ),
        )
        self._attach_log_handler()
        page.add(ft.SafeArea(expand=True, content=self.tabs))
        logger.info("Interface prête — paramètres actifs : %s", self.parameters)
        self._log_calibration()
        return self.tabs

    def _build_image_tab(self) -> ft.Control:
        """Onglet Image : boutons, statut et visionneuse zoomable à taille mesurée."""
        self.image_status = ft.Text(
            "Ouvre une ou plusieurs images, puis sélectionne l’ordre zéro, le spectre et le ciel.", size=13
        )
        # Le placeholder épouse la zone (left/top/right/bottom).
        self.image_stack = ft.Stack(
            width=self.stage_size[0],
            height=self.stage_size[1],
            alignment=ft.Alignment.TOP_LEFT,
            controls=[self._placeholder()],
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
        self.rectified_checkbox = ft.Checkbox(
            label="Image transformée", value=False, on_change=self._on_rectified_toggle
        )
        low, high = self.cut_positions
        self.cut_slider = ft.RangeSlider(
            min=0, max=100, start_value=low, end_value=high, width=220,
            tooltip="Niveaux noir / blanc (percentiles de l’image)",
            on_change=self._on_cuts_moving,
            on_change_end=self._on_cuts_changed,
        )
        self.cut_text = ft.Text(self._cut_label(), size=12, width=88)
        return ft.Column(
            expand=True,
            spacing=8,
            controls=[
                ft.Row(
                    wrap=True,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Button("Ouvrir des images", icon=ft.Icons.FOLDER_OPEN, on_click=self._open_picker),
                        ft.Button("Sélection des points", icon=ft.Icons.TOUCH_APP, on_click=self._begin_selection),
                        ft.Row(  # curseurs de contraste : restent groupés quand la ligne se replie
                            tight=True,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            controls=[
                                self.cut_slider,
                                self.cut_text,
                                ft.IconButton(
                                    ft.Icons.AUTO_FIX_HIGH, tooltip="Contraste automatique",
                                    on_click=self._on_cuts_auto,
                                ),
                                self.rectified_checkbox,
                            ],
                        ),
                        #self.rectified_checkbox,
                        #ft.Text("Pince pour zoomer · glisse pour déplacer", size=12),
                    ],
                ),
                self.image_status,
                ft.Container(
                    expand=True,
                    bgcolor="#080b10",
                    border_radius=8,
                    clip_behavior=ft.ClipBehavior.HARD_EDGE,
                    content=self.image_viewer,
                    on_size_change=self._on_image_stage_size_change,
                    size_change_interval=100,
                ),
            ],
        )

    @staticmethod
    def _placeholder() -> ft.Control:
        return ft.Container(
            content=ft.Text("Ouvre une image", color=ft.Colors.WHITE_70),
            alignment=ft.Alignment.CENTER,
            left=0, top=0, right=0, bottom=0,
        )

    def _build_spectrum_tab(self) -> ft.Control:
        """Onglet Spectre : le graphe est re-rendu à la taille mesurée de son conteneur."""
        self.spectrum_status = ft.Text(
            "Sélectionne trois points sur l’image pour extraire le spectre.", size=13
        )
        # Le graphe est rendu en SPECTRUM_RENDER_SCALE × la zone affichée, puis zoomé à la pince
        # (InteractiveViewer) sans nouveau rendu matplotlib.
        self.spectrum_image = ft.Image(
            src=spectrum_png_data_uri(), fit=ft.BoxFit.CONTAIN,
            width=DEFAULT_PLOT_SIZE[0], height=DEFAULT_PLOT_SIZE[1],
        )
        self.spectrum_viewer = ft.InteractiveViewer(
            content=self.spectrum_image,
            expand=True,
            min_scale=1,
            max_scale=8,
            pan_enabled=True,
            scale_enabled=True,
            constrained=True,
        )
        self.spectrum_stage = ft.Container(
            expand=True,
            bgcolor="#080b10",
            content=self.spectrum_viewer,
            on_size_change=self._on_spectrum_plot_size_change,
            size_change_interval=150,
        )
        self.spectrum_color_checkbox = ft.Checkbox(
            label="Couleurs", value=False, on_change=self._on_spectrum_display_option_changed
        )
        self.spectrum_lines_checkbox = ft.Checkbox(
            label="Raies", value=False, on_change=self._on_spectrum_display_option_changed
        )
        self.spectrum_normalize_checkbox = ft.Checkbox(
            label="Normalisé", value=False, on_change=self._on_spectrum_display_option_changed
        )
        return ft.Column(
            expand=True,
            spacing=8,
            controls=[
                ft.Row(
                    controls=[
                        #ft.Text("Spectre SA100", size=20, weight=ft.FontWeight.BOLD),
                        ft.Row(
                            wrap=True,
                            controls=[
                                ft.Button("Ouvrir un spectre", icon=ft.Icons.FILE_OPEN, on_click=self._open_spectrum_file),
                                ft.Button("Calculer la réponse", icon=ft.Icons.TUNE, on_click=self._compute_response_from_file),
                                ft.Button("Reset zoom", icon=ft.Icons.ZOOM_OUT_MAP, on_click=self._reset_spectrum_zoom),
                                ft.Button("Clear", icon=ft.Icons.DELETE_OUTLINE, on_click=self._clear_spectrum),
                                self.spectrum_color_checkbox,
                                self.spectrum_lines_checkbox,
                                self.spectrum_normalize_checkbox,
                            ],
                        ),
                    ],
                ),
                self.spectrum_status,
                self.spectrum_stage,
            ],
        )

    def _build_logs_tab(self) -> ft.Control:
        self.log_view = ft.ListView(expand=True, spacing=2, auto_scroll=True)
        self.debug_checkbox = ft.Checkbox(label="Debug", value=False, on_change=self._on_debug_changed)
        return ft.Column(
            expand=True,
            spacing=8,
            controls=[
                ft.Row(
                    controls=[
                        ft.Text("Sortie du Logger", size=20, weight=ft.FontWeight.BOLD),
                        self.debug_checkbox,
                        ft.Button("Clear", icon=ft.Icons.DELETE_OUTLINE, on_click=self._clear_logs),
                    ]
                ),
                ft.Container(expand=True, bgcolor="#080b10", border_radius=8, padding=8, content=self.log_view),
            ],
        )

    def _build_parameters_tab(self) -> ft.Control:
        """Onglet Parameters : un contrôle par entrée de PARAM_SPECS, groupés par section."""
        form_controls: list[ft.Control] = []
        for spec in PARAM_SPECS:
            value = getattr(self.parameters, spec.name)
            if spec.name == "star_analyser":
                control: ft.Control = ft.RadioGroup(
                    value=str(value),
                    content=ft.Row(
                        controls=[
                            ft.Radio(value="100", label="100 traits/mm"),
                            ft.Radio(value="200", label="200 traits/mm"),
                        ]
                    ),
                )
            else:
                control = ft.TextField(
                    label=spec.label,
                    value=str(value),
                    keyboard_type=ft.KeyboardType.NUMBER if spec.kind is int else ft.KeyboardType.TEXT,
                )
            self.param_fields[spec.name] = control
            setattr(self, spec.widget, control)
            if spec.section:
                form_controls.append(ft.Text(spec.section, size=15, weight=ft.FontWeight.BOLD))
                if spec.section in SECTION_NOTES:
                    form_controls.append(
                        ft.Text(SECTION_NOTES[spec.section], size=12, color=ft.Colors.WHITE_70)
                    )
            form_controls.append(control)
        self.parameter_status = ft.Text(self.parameter_load_error or "Valeurs prêtes à modifier.", size=13)
        return ft.Column(
            expand=True,
            scroll=ft.ScrollMode.AUTO,
            spacing=12,
            controls=[
                ft.Text("Parameters", size=20, weight=ft.FontWeight.BOLD),
                *form_controls,
                ft.Row(
                    wrap=True,
                    controls=[
                        ft.Button("Apply and save", icon=ft.Icons.SAVE_OUTLINED, on_click=self._apply_and_save_parameters),
                        ft.Button("Undo", icon=ft.Icons.UNDO, on_click=self._undo_parameters),
                    ],
                ),
                self.parameter_status,
            ],
        )

    # ------------------------------------------------------------------ paramètres

    def _load_saved_parameters(self) -> str | None:
        """Charge les paramètres enregistrés ; renvoie un message d'erreur ou ``None``."""
        self.parameters, error = load_parameters(self.parameters_file)
        return error

    def _configure_spectra_processor(self) -> None:
        """Applique les paramètres optiques au processeur de spectres."""
        p, sp = self.parameters, self.spectra_processor
        sp.pixel_size_um = p.pixel_size_um
        sp.focal_mm = float(p.focal_mm)
        sp.binning = self.image_processor.binning
        sp.grating_period_nm = 1_000_000.0 / p.star_analyser
        sp.lambda_min = float(p.lambda_min)
        sp.lambda_max = float(p.lambda_max)
        sp.spectrum_height = p.spectrum_height
        sp.sky_height = p.sky_height
        self._log_calibration()

    def _log_calibration(self) -> None:
        """Trace les paramètres de calibration actifs."""
        p, sp = self.parameters, self.spectra_processor
        logger.info(
            "Calibration : réseau %d traits/mm (période %.0f nm), focale %d mm, pixel natif "
            "%.2f µm × binning %d = %.2f µm, fenêtre λ %d–%d Å",
            p.star_analyser, sp.grating_period_nm, p.focal_mm, p.pixel_size_um, sp.binning,
            p.pixel_size_um * sp.binning, p.lambda_min, p.lambda_max,
        )
        logger.info(
            "Bandes de binning : spectre %d px, ciel %d px", p.spectrum_height, p.sky_height
        )
        logger.info("Affinage de l'ordre zéro : rayon %d px", p.centroid_radius)

    def _read_parameters_from_controls(self) -> AppParameters:
        """Lit et valide le formulaire.

        Raises:
            ValueError: Si une saisie est absente, mal formée ou hors limites.
        """
        if not self.param_fields:
            raise ValueError("Le formulaire des paramètres n’est pas disponible.")
        raw = {name: control.value or "" for name, control in self.param_fields.items()}
        try:
            return AppParameters.from_form(raw)
        except ValueError as exc:
            raise ValueError(f"Vérifie les valeurs saisies : {exc}") from exc

    def _set_parameter_fields(self, parameters: AppParameters) -> None:
        """Affiche un jeu de paramètres dans le formulaire."""
        for name, control in self.param_fields.items():
            control.value = str(getattr(parameters, name))

    def _set_parameter_status(self, message: str) -> None:
        if self.parameter_status is not None:
            self.parameter_status.value = message
            if self.page is not None:
                self.page.update()

    def _apply_and_save_parameters(self, _event: ft.ControlEvent) -> None:
        """Valide, enregistre et applique le formulaire, puis recalcule le spectre courant."""
        try:
            parameters = self._read_parameters_from_controls()
        except ValueError as exc:
            logger.warning("Paramètres refusés : %s", exc)
            self._set_parameter_status(str(exc))
            return
        try:
            save_parameters(self.parameters_file, parameters)
        except OSError as exc:
            logger.exception("Enregistrement des paramètres impossible")
            self._set_parameter_status(f"Échec de l’enregistrement : {exc}")
            return

        changes = {
            name: (getattr(self.parameters, name), getattr(parameters, name))
            for name in self.param_fields
            if getattr(self.parameters, name) != getattr(parameters, name)
        }
        logger.info("Paramètres appliqués ; modifiés : %s", changes or "aucun")
        self.parameters = parameters
        self._configure_spectra_processor()
        self._set_parameter_status("Paramètres appliqués et enregistrés.")
        if self.image_processor.paths and any(name in changes for name in CALIBRATION_FIELDS):
            logger.info("Masters modifiés : rechargement des images avec les nouvelles corrections")
            try:
                self._load_images(self.image_processor.paths, keep_selection=True)
            except (OSError, ValueError, RuntimeError) as exc:
                logger.exception("Rechargement des images impossible")
                self._set_parameter_status(f"Paramètres enregistrés, mais rechargement impossible : {exc}")
                return
        if self.image_processor.data is not None and len(self.selection_points) == 3:
            logger.info("Recalcul du spectre courant avec les nouveaux paramètres")
            self._extract_spectrum(navigate_to_spectrum=False, replace_latest=True)
        else:
            self._render_spectrum()

    def _undo_parameters(self, _event: ft.ControlEvent) -> None:
        """Restaure le formulaire avec les derniers paramètres appliqués."""
        logger.info("Modifications du formulaire annulées")
        self._set_parameter_fields(self.parameters)
        self._set_parameter_status("Modifications annulées.")

    # ------------------------------------------------------------------ logs

    def _attach_log_handler(self) -> None:
        """Branche le journal de l'application sur l'onglet Logs."""
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
        self._set_debug_logging(bool(self.debug_checkbox and self.debug_checkbox.value))

    def _set_debug_logging(self, enabled: bool) -> None:
        """Règle la verbosité (INFO, ou DEBUG si activé) des loggers et de l'interface."""
        level = logging.DEBUG if enabled else logging.INFO
        if self.log_handler is not None:
            self.log_handler.setLevel(level)
        for application_logger in APPLICATION_LOGGERS:
            application_logger.setLevel(level)

    def _append_log_record(self, record: logging.LogRecord) -> None:
        if self.log_view is None or self.log_handler is None:
            return
        self.log_view.controls.append(
            ft.Text(
                self.log_handler.format(record),
                color=LOG_LEVEL_COLORS.get(record.levelno, "#ffffff"),
                font_family="monospace",
                selectable=True,
                size=15,
            )
        )
        if self.page is not None:
            self.page.update()

    def _clear_logs(self, _event: ft.ControlEvent | None = None) -> None:
        if self.log_view is None:
            return
        self.log_view.controls.clear()
        if self.page is not None:
            self.page.update()

    # ------------------------------------------------------------------ tailles dynamiques

    def _on_tab_change(self, _event: ft.ControlEvent) -> None:
        """Rafraîchit le contenu de l'onglet qui devient visible (taille parfois nulle avant)."""
        index = self.tabs.selected_index if self.tabs else None
        logger.debug("Onglet %s affiché", index)
        self._apply_tab_visibility()
        if index == 0:
            self._refresh_image_controls()
        elif index == 1:
            self._render_spectrum()

    def _apply_tab_visibility(self) -> None:
        """N'affiche que la page de l'onglet sélectionné (changement uniquement par appui)."""
        if self.tabs is None:
            return
        selected = self.tabs.selected_index or 0
        for index, view in enumerate(self.tab_pages):
            view.visible = index == selected
        if self.page is not None:
            self.page.update()

    def _on_image_stage_size_change(self, event: ft.LayoutSizeChangeEvent) -> None:
        """Adapte la zone image à la taille réellement mesurée de son conteneur."""
        new_size = (float(event.width), float(event.height))
        if new_size[0] <= 0 or new_size[1] <= 0:
            logger.debug("Zone image de taille nulle ignorée (onglet masqué ?)")
            return
        if self._stage_measured and not _size_changed(self.stage_size, new_size):
            return
        logger.debug(
            "Zone image : %.0f × %.0f → %.0f × %.0f px",
            *self.stage_size, *new_size,
        )
        self.stage_size = new_size
        self._stage_measured = True
        self._apply_stage_size()
        self._refresh_image_controls()

    def _apply_stage_size(self) -> None:
        if self.image_stack is not None:
            self.image_stack.width, self.image_stack.height = self.stage_size

    def _on_spectrum_plot_size_change(self, event: ft.LayoutSizeChangeEvent) -> None:
        """Re-rend le graphe à la taille mesurée de son conteneur."""
        if event.width <= 0 or event.height <= 0:
            logger.debug("Zone du graphe de taille nulle ignorée (onglet masqué ?)")
            return
        new_size = (float(event.width), float(event.height))
        previous = self._spectrum_plot_size
        self._spectrum_plot_size = new_size
        if not _size_changed(previous, new_size):
            return
        logger.debug("Zone du graphe : %s → %.0f × %.0f px", previous, *new_size)
        self.spectrum_image.width, self.spectrum_image.height = new_size
        self._render_spectrum()

    # ------------------------------------------------------------------ image

    async def _open_picker(self, _event: ft.ControlEvent) -> None:
        """Ouvre le sélecteur (une ou plusieurs images, dernier répertoire) et charge la sélection."""
        logger.info("Sélecteur d'images ouvert dans %s", self.last_directory)
        try:
            selected = await self.picker.pick_files(
                dialog_title="Choisir une ou plusieurs images",
                initial_directory=self.last_directory,
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=IMAGE_EXTENSIONS + [e.upper() for e in IMAGE_EXTENSIONS],
                allow_multiple=True,
            )
            if not selected:
                logger.info("Sélection d'images annulée")
                return
            if not all(file.path for file in selected):
                raise ValueError("Le sélecteur n’a pas fourni de chemin de fichier.")
            paths = [Path(file.path) for file in selected]
            self.last_directory = str(paths[0].parent)
            self._show_image_status(f"Chargement de {len(paths)} image(s)…")
            # Lecture, corrections et recalage dans un thread : l'interface reste réactive.
            data = await asyncio.to_thread(self._compute_stack, paths)
            self._show_loaded(paths, data)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.exception("Ouverture des images impossible")
            self._show_image_status(f"Erreur d’ouverture : {exc}")

    def _compute_stack(self, paths: list[Path]) -> np.ndarray:
        """Charge les masters, corrige chaque image, les recale sur la première et les additionne."""
        p = self.parameters
        directory = Path(p.calibration_dir) if p.calibration_dir else paths[0].parent
        masters = load_masters(
            directory, {"offset": p.offset_name, "dark": p.dark_name, "flat": p.flat_name}
        )
        return self.image_processor.load_stack(paths, **masters)

    def _load_images(self, paths: list[str | Path], *, keep_selection: bool = False) -> None:
        """Charge une ou plusieurs images (version synchrone de l'ouverture)."""
        paths = [Path(path) for path in paths]
        self._show_loaded(paths, self._compute_stack(paths), keep_selection=keep_selection)

    def _show_loaded(
        self, paths: list[Path], data: np.ndarray, *, keep_selection: bool = False
    ) -> None:
        """Affiche l'image obtenue ; réinitialise la sélection sauf ``keep_selection``.

        Les spectres déjà tracés sont conservés.
        """
        count = len(paths)
        self.image_shape = data.shape
        self.current_image_label = paths[0].name + (f" (+{count - 1})" if count > 1 else "")
        if not keep_selection:
            self.selection_points.clear()
        self.selecting = False
        self.spectra_processor.result = None
        self._configure_spectra_processor()  # le binning dépend de l'image chargée
        self._previews.clear()
        self._cut_source = None  # nouvel échantillon pour les curseurs
        if not keep_selection:
            self._reset_cuts()
        self._display_limits = None
        self._set_show_rectified(False)
        finite = data[np.isfinite(data)]
        logger.info(
            "Image %d × %d px : min=%.1f médiane=%.1f max=%.1f",
            data.shape[1], data.shape[0], finite.min(), np.median(finite), finite.max(),
        )
        applied = self.image_processor.applied_corrections
        details = [f"{data.shape[1]} × {data.shape[0]} px", f"corrections : {', '.join(applied) or 'aucune'}"]
        if count > 1:
            details.insert(0, f"{count} images recalées et additionnées")
        self.spectrum_status.value = self._spectrum_summary()
        self._show_image_status(f"{self.current_image_label} — " + " — ".join(details))
        self._refresh_image_controls(data)
        self._render_spectrum()
        if self.page is not None:
            self.page.update()

    def _preview_uri(self, data: ArrayLike) -> str:
        """Aperçu PNG de ``data``, calculé une seule fois (le resize réutilise le cache)."""
        for source, uri in self._previews:
            if source is data:
                return uri
        uri = image_png_data_uri(data, self._stretch_limits())
        self._previews = (self._previews + [(data, uri)])[-2:]  # image d'origine + transformée
        return uri

    @staticmethod
    def _auto_cut_positions() -> tuple[float, float]:
        return tuple(cut_position(p) for p in AUTO_CUT_PERCENTILES)

    def _cut_percentiles(self) -> tuple[float, float]:
        return cut_percentile(self.cut_positions[0]), cut_percentile(self.cut_positions[1])

    def _cut_label(self) -> str:
        low, high = self._cut_percentiles()
        return f"{low:.1f} – {high:.2f} %"

    def _stretch_limits(self) -> tuple[float, float] | None:
        """Niveaux noir/blanc de l'image d'origine, communs à l'image transformée.

        Ils viennent des curseurs (percentiles de l'image d'origine). Sans bornes communes,
        les coins noirs ajoutés par la rotation faussent le contraste et l'image transformée
        paraît plus claire que l'originale.
        """
        original = self.image_processor.data
        if original is None:
            return None
        if self._cut_source is not original:
            self._cut_source = original
            self._cut_sample = cut_sample(original)
            self._display_limits = None
        if self._display_limits is None:
            if not self._cut_sample.size:
                return None
            low_p, high_p = self._cut_percentiles()
            self._display_limits = cut_limits(self._cut_sample, low_p, high_p)
            logger.info(
                "Contraste : percentiles %.2f %% – %.2f %% → noir=%.1f blanc=%.1f (commun aux deux images)",
                low_p, high_p, *self._display_limits,
            )
        return self._display_limits

    def _reset_cuts(self) -> None:
        """Remet les curseurs au contraste automatique."""
        self.cut_positions = self._auto_cut_positions()
        self._display_limits = None
        if self.cut_slider is not None:
            self.cut_slider.start_value, self.cut_slider.end_value = self.cut_positions
        if self.cut_text is not None:
            self.cut_text.value = self._cut_label()

    def _on_cuts_moving(self, event: ft.ControlEvent) -> None:
        """Pendant le glissé : met à jour seulement l'affichage des percentiles (pas de re-rendu)."""
        self.cut_positions = (float(self.cut_slider.start_value), float(self.cut_slider.end_value))
        self.cut_text.value = self._cut_label()
        self.cut_text.update()

    def _on_cuts_changed(self, event: ft.ControlEvent) -> None:
        """Au relâchement : recalcule les niveaux noir/blanc et redessine l'image."""
        self.cut_positions = (float(self.cut_slider.start_value), float(self.cut_slider.end_value))
        self._apply_cuts()

    def _on_cuts_auto(self, _event: ft.ControlEvent) -> None:
        logger.info("Contraste automatique")
        self._reset_cuts()
        self._apply_cuts()

    def _apply_cuts(self) -> None:
        self.cut_text.value = self._cut_label()
        self._display_limits = None
        self._previews.clear()  # les aperçus PNG dépendent des niveaux
        self._refresh_image_controls()
        if self.page is not None:
            self.page.update()

    def _display_data(self) -> ArrayLike | None:
        """Image affichée : l'image transformée après extraction, sinon l'originale."""
        result = self.spectra_processor.result
        if self.show_rectified and result is not None:
            return result.rectified_image
        return self.image_processor.data

    def _set_show_rectified(self, show: bool) -> None:
        self.show_rectified = show
        if self.rectified_checkbox is not None:
            self.rectified_checkbox.value = show

    def _on_rectified_toggle(self, _event: ft.ControlEvent) -> None:
        """Bascule entre l'image d'origine et l'image transformée."""
        wanted = bool(self.rectified_checkbox and self.rectified_checkbox.value)
        if wanted and self.spectra_processor.result is None:
            logger.info("Pas d'image transformée : extrais d'abord un spectre")
            wanted = False
        self._set_show_rectified(wanted)
        logger.info("Affichage : %s", "image transformée" if wanted else "image d'origine")
        self._refresh_image_controls()
        if self.page is not None:
            self.page.update()

    def _band_lines(
        self, left: float, top: float, display_width: float, display_height: float
    ) -> list[ft.Control]:
        """Lignes horizontales délimitant la bande du spectre (rouge) et celle du ciel (bleu)."""
        result = self.spectra_processor.result
        height, width = self.image_shape
        x_left = left + result.x_start / width * display_width
        lines: list[ft.Control] = []
        for name, rows, color in (
            ("spectre", result.spectrum_rows, BAND_COLORS[0]),
            ("ciel", result.sky_rows, BAND_COLORS[1]),
        ):
            for row in rows:
                lines.append(
                    ft.Container(
                        left=x_left,
                        top=top + row / height * display_height - BAND_LINE_PX / 2,
                        width=left + display_width - x_left,
                        height=BAND_LINE_PX,
                        bgcolor=color,
                    )
                )
            logger.debug("Lignes %s tracées aux rangées %d et %d", name, *rows)
        return lines

    def _refresh_image_controls(self, data: ArrayLike | None = None) -> None:
        """Redessine l'aperçu à la taille de la zone et les marqueurs des points choisis."""
        if data is None:
            data = self._display_data()
        if data is None or self.image_shape is None:
            return
        width, height = self.stage_size
        controls: list[ft.Control] = [
            ft.Image(
                src=self._preview_uri(data),
                width=width,
                height=height,
                fit=ft.BoxFit.CONTAIN,
                gapless_playback=True,
            )
        ]
        left, top, display_width, display_height = image_bounds(self.image_shape, self.stage_size)
        if self.show_rectified and self.spectra_processor.result is not None:
            controls.extend(self._band_lines(left, top, display_width, display_height))
        for index, (x, y) in enumerate([] if self.show_rectified else self.selection_points):
            controls.append(
                ft.Icon(
                    ft.Icons.CLOSE,
                    color=MARKER_COLORS[index],
                    size=12,
                    left=left + x / self.image_shape[1] * display_width - 6,
                    top=top + y / self.image_shape[0] * display_height - 6,
                )
            )
        self._apply_stage_size()
        self.image_stack.controls = controls
        self.image_stack.update()

    def _begin_selection(self, _event: ft.ControlEvent | None = None) -> None:
        """Démarre une nouvelle sélection de trois points."""
        if self.image_processor.data is None:
            self._show_image_status("Charge d’abord une image.")
            return
        logger.info("Nouvelle sélection : ordre zéro, trace du spectre, fond de ciel")
        self.selection_points.clear()
        self.selecting = True
        self._set_show_rectified(False)
        self._refresh_image_controls()
        self._show_image_status("Clique l’ordre zéro (1/3).")

    def _on_image_tap(self, event: ft.TapEvent) -> None:
        """Enregistre un point ; extrait le spectre au troisième."""
        if not self.selecting or self.image_shape is None:
            return
        point = image_point_from_local(event.local_position, self.image_shape, self.stage_size)
        if point is None:
            logger.debug("Clic hors de l'image ignoré : %s", event.local_position)
            return
        if not self.selection_points:
            point = refine_centroid(
                self.image_processor.data, *point, self.parameters.centroid_radius
            )
        self.selection_points.append(point)
        logger.info(
            "Point %d/3 (%s) : x=%.1f y=%.1f px",
            len(self.selection_points), MARKER_NAMES[len(self.selection_points) - 1], *point,
        )
        self._refresh_image_controls()
        if len(self.selection_points) < 3:
            self._show_image_status(
                f"Clique le {MARKER_NAMES[len(self.selection_points)]} "
                f"({len(self.selection_points) + 1}/3)."
            )
            return
        self._extract_spectrum()

    def _extract_spectrum(
        self, *, navigate_to_spectrum: bool = False, replace_latest: bool = False
    ) -> None:
        """Extrait un spectre, ajoute (ou remplace) sa trace et affiche l'image transformée."""
        self.selecting = False
        try:
            result = self.spectra_processor.process(
                self.image_processor.data, *self.selection_points
            )
        except (ValueError, RuntimeError) as exc:
            logger.exception("Extraction du spectre impossible")
            self.spectrum_status.value = f"Extraction impossible : {exc}"
            self._show_image_status(f"Extraction impossible : {exc}")
            if self.page is not None:
                self.page.update()
            return

        wavelengths, flux, corrected = self._apply_instrument_response(result.wavelengths, result.flux)
        self._add_spectrum_trace(wavelengths, flux, replace_latest=replace_latest)
        self._render_spectrum()
        self.spectrum_status.value = self._spectrum_summary(result.dispersion_angstrom_per_pixel)
        if corrected:
            self.spectrum_status.value += " — divisé par la réponse instrumentale"
        self._set_show_rectified(True)
        self._refresh_image_controls()
        self._show_image_status(
            "Spectre extrait (image transformée, bandes spectre en rouge et ciel en bleu). "
            "Sélectionne à nouveau pour recommencer."
        )
        if navigate_to_spectrum and self.page and self.tabs:
            logger.info("Passage à l'onglet Spectre")
            self.tabs.selected_index = 1
            self._apply_tab_visibility()
        if self.page is not None:
            self.page.update()

    # ------------------------------------------------------------------ spectre

    def _masters_directory(self) -> Path:
        """Répertoire des masters : celui des paramètres, sinon le dossier de l'image ouverte."""
        if self.parameters.calibration_dir:
            return Path(self.parameters.calibration_dir)
        if self.image_processor.paths:
            return Path(self.image_processor.paths[0]).parent
        return Path(self.last_directory)

    def _apply_instrument_response(
        self, wavelengths: ArrayLike, flux: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Divise le spectre par la réponse instrumentale si elle existe ; renvoie aussi un indicateur."""
        wavelengths = np.asarray(wavelengths, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)
        name = self.parameters.response_name
        if not name:
            logger.info("Réponse instrumentale désactivée (nom vide)")
            return wavelengths, flux, False
        response = load_response(self._masters_directory(), name)
        if response is None:
            return wavelengths, flux, False
        try:
            wavelengths, flux = apply_response(wavelengths, flux, response)
        except ValueError as exc:
            logger.warning("Réponse instrumentale non appliquée : %s", exc)
            return wavelengths, flux, False
        return wavelengths, flux, True

    async def _pick_spectrum_files(self, title: str, *, multiple: bool) -> list[Path]:
        """Sélecteur de fichiers de spectres (DAT, TXT, CSV, FITS) ; liste vide si annulé."""
        selected = await self.picker.pick_files(
            dialog_title=title,
            initial_directory=self.last_directory,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=SPECTRUM_EXTENSIONS + [e.upper() for e in SPECTRUM_EXTENSIONS],
            allow_multiple=multiple,
        )
        if not selected:
            logger.info("Sélection de spectre annulée")
            return []
        if not all(file.path for file in selected):
            raise ValueError("Le sélecteur n’a pas fourni de chemin de fichier.")
        paths = [Path(file.path) for file in selected]
        self.last_directory = str(paths[0].parent)
        return paths

    def _show_spectrum_message(self, message: str) -> None:
        self.spectrum_status.value = message
        if self.page is not None:
            self.page.update()

    async def _open_spectrum_file(self, _event: ft.ControlEvent) -> None:
        """Ajoute au graphe un ou plusieurs spectres 1D lus depuis des fichiers."""
        try:
            for path in await self._pick_spectrum_files("Choisir un ou plusieurs spectres", multiple=True):
                self.add_spectrum_from_file(path)
        except (OSError, ValueError) as exc:
            logger.warning("Ouverture du spectre impossible : %s", exc)
            self._show_spectrum_message(f"Ouverture impossible : {exc}")

    def add_spectrum_from_file(self, path: Path) -> None:
        """Lit un spectre 1D et l'ajoute comme trace (sans correction : il est déjà calibré)."""
        wavelengths, flux = read_spectrum(path)
        self._add_spectrum_trace(wavelengths, flux, label=path.name)
        self._render_spectrum()
        self._show_spectrum_message(self._spectrum_summary())

    async def _compute_response_from_file(self, _event: ft.ControlEvent) -> None:
        """Choisit le spectre de référence et calcule la réponse à partir du dernier spectre extrait."""
        if self.spectra_processor.result is None:
            self._show_spectrum_message(
                "Extrais d’abord le spectre de l’étoile de référence (trois points sur l’image)."
            )
            return
        try:
            paths = await self._pick_spectrum_files("Spectre de référence (MILES, CALSPEC…)", multiple=False)
            if paths:
                self.compute_response(paths[0])
        except (OSError, ValueError) as exc:
            logger.warning("Calcul de la réponse impossible : %s", exc)
            self._show_spectrum_message(f"Calcul de la réponse impossible : {exc}")

    def compute_response(self, reference_path: Path) -> Path:
        """Calcule et enregistre la réponse instrumentale, puis recalcule le spectre courant.

        Raises:
            ValueError: Si aucun spectre n'est extrait, si le nom de la réponse est vide ou si
                l'ajustement échoue.
        """
        p, result = self.parameters, self.spectra_processor.result
        if result is None:
            raise ValueError("aucun spectre extrait")
        if not p.response_name:
            raise ValueError("le nom de la réponse instrumentale est vide (onglet Parameters)")
        logger.info("Calcul de la réponse instrumentale avec la référence %s", reference_path.name)
        reference_wavelengths, reference_flux = read_spectrum(reference_path)
        fit = compute_response(
            result.wavelengths, result.flux, reference_wavelengths, reference_flux,
            degree=p.response_degree,
            smooth_px=p.response_smooth_px,
            line_half_width=float(p.response_line_mask),
            lambda_min=float(p.lambda_min),
            lambda_max=float(p.lambda_max),
        )
        directory = self._masters_directory()
        saved = save_response(directory, p.response_name, fit)
        if len(self.selection_points) == 3 and self.image_processor.data is not None:
            self._extract_spectrum(replace_latest=True)  # affiche le spectre déjà corrigé
        self._show_spectrum_message(
            f"Réponse enregistrée : {saved} (λ {fit.wavelengths[0]:.0f}–{fit.wavelengths[-1]:.0f} Å, "
            f"écart autour du polynôme {fit.rms_percent:.1f} %)."
        )
        return saved

    def _add_spectrum_trace(
        self,
        wavelengths: ArrayLike,
        flux: ArrayLike,
        *,
        replace_latest: bool = False,
        label: str | None = None,
    ) -> None:
        """Ajoute une trace colorée (la plus ancienne est remplacée si le graphe est plein)."""
        x_values = np.asarray(wavelengths, dtype=np.float64).copy()
        y_values = np.asarray(flux, dtype=np.float32).copy()
        if x_values.ndim != 1 or y_values.ndim != 1 or x_values.size != y_values.size:
            raise ValueError("Wavelength and flux arrays must be matching 1D arrays.")

        if replace_latest and self.spectrum_traces:
            previous = self.spectrum_traces[-1]
            self.spectrum_traces[-1] = SpectrumTrace(x_values, y_values, previous.color, previous.label)
            logger.info("Trace « %s » recalculée (%d points)", previous.label, x_values.size)
            return

        if len(self.spectrum_traces) >= MAX_SPECTRUM_TRACES:
            removed = self.spectrum_traces.pop(0)
            logger.info("Graphe plein : trace « %s » supprimée", removed.label)
        used = {trace.color for trace in self.spectrum_traces}
        color = random.choice([c for c in SPECTRUM_COLORS if c not in used])
        self._spectrum_sequence += 1
        label = label or f"{self._spectrum_sequence} — {self.current_image_label}"
        self.spectrum_traces.append(SpectrumTrace(x_values, y_values, color, label))
        logger.info(
            "Trace « %s » ajoutée (%d points, λ %.0f–%.0f Å, couleur %s)",
            label, x_values.size, x_values.min(), x_values.max(), color,
        )

    def _spectrum_summary(self, dispersion: float | None = None) -> str:
        if not self.spectrum_traces:
            return "Charge une image et sélectionne trois points pour ajouter un spectre."
        details = f"{len(self.spectrum_traces)}/{MAX_SPECTRUM_TRACES} spectres affichés"
        if dispersion is not None:
            details += f" — dispersion {dispersion:.2f} Å/px"
        return details

    def _display_traces(self) -> tuple[SpectrumTrace, ...]:
        """Traces à tracer ; avec « Normalisé », chacune est ramenée à un maximum de 1."""
        if not (self.spectrum_normalize_checkbox and self.spectrum_normalize_checkbox.value):
            return tuple(self.spectrum_traces)
        normalized = []
        for trace in self.spectrum_traces:
            peak = float(np.nanmax(trace.flux)) if trace.flux.size else 0.0
            flux = trace.flux / peak if peak > 0 else trace.flux
            normalized.append(SpectrumTrace(trace.wavelengths, flux.astype(np.float32), trace.color, trace.label))
        return tuple(normalized)

    def _render_spectrum(self) -> None:
        """Re-génère le graphe à la taille mesurée de la zone d'affichage."""
        if self.spectrum_image is None:
            return
        size = self._spectrum_plot_size or DEFAULT_PLOT_SIZE
        self.spectrum_image.src, _ = _render_spectrum_plot(
            line_size=self.parameters.line_size,
            line_type=self.parameters.line_type,
            traces=self._display_traces(),
            show_color_band=bool(self.spectrum_color_checkbox and self.spectrum_color_checkbox.value),
            show_element_lines=bool(self.spectrum_lines_checkbox and self.spectrum_lines_checkbox.value),
            size_px=size,
            scale=SPECTRUM_RENDER_SCALE,
        )
        self.spectrum_image.update()

    def _on_spectrum_display_option_changed(self, _event: ft.ControlEvent) -> None:
        logger.info(
            "Affichage : couleurs=%s raies=%s normalisé=%s",
            self.spectrum_color_checkbox.value, self.spectrum_lines_checkbox.value,
            self.spectrum_normalize_checkbox.value,
        )
        self._render_spectrum()

    async def _reset_spectrum_zoom(self, _event: ft.ControlEvent) -> None:
        """Remet le graphe à l'échelle 1, non déplacé."""
        logger.info("Zoom du spectre réinitialisé")
        if self.spectrum_viewer is None:
            return
        try:
            await self.spectrum_viewer.reset()
        except RuntimeError:
            logger.debug("Visionneuse du spectre pas encore affichée : rien à réinitialiser")

    def _clear_spectrum(self, _event: ft.ControlEvent | None = None) -> None:
        logger.info("Effacement des %d spectre(s) affiché(s)", len(self.spectrum_traces))
        self.spectra_processor.result = None
        self.spectrum_traces.clear()
        self.spectrum_status.value = self._spectrum_summary()
        self._render_spectrum()
        if self.page is not None:
            self.page.update()

    # ------------------------------------------------------------------ divers

    def _show_image_status(self, message: str) -> None:
        if self.image_status is not None:
            self.image_status.value = message
            self.image_status.update()


def main(page: ft.Page) -> None:
    """Point d'entrée Flet."""
    QuickSpecFletApp().build(page)


if __name__ == "__main__":
    ft.run(main)

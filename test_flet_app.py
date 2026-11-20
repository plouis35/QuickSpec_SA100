"""Tests for Flet image display, coordinate mapping and SA100 workflow."""

import base64
from io import BytesIO
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import flet as ft
import numpy as np
from PIL import Image as PillowImage

from plotting import display_limits
from flet_app import (
    AppParameters,
    ELEMENT_LINE_COLORS,
    MAX_SPECTRUM_TRACES,
    QuickSpecFletApp,
    SPECTRUM_COLORS,
    SpectrumTrace,
    _render_spectrum_plot,
    _visible_spectrum_rgb,
    image_point_from_local,
    image_png_data_uri,
    spectrum_png_data_uri,
)
from spectra_processing import Spectra


class FakePage:
    """Provide the minimal page interface required by the UI tests."""

    def __init__(self) -> None:
        """Initialize a fake Linux page and empty control collections."""
        self.width = 800
        self.height = 700
        self.platform = ft.PagePlatform.LINUX
        self.overlay = []
        self.services = []
        self.controls = []

    def add(self, *controls: ft.Control) -> None:
        """Append controls to the page's visible control collection."""
        self.controls.extend(controls)

    def update(self) -> None:
        """Stand in for the page update operation."""
        pass


class FletImageTests(unittest.TestCase):
    """Verify Flet image previews and spectrum interaction behavior."""

    def test_tap_coordinates_map_through_contain_fit(self) -> None:
        """Map taps through a contain-fitted image and reject letterbox taps."""
        image_shape = (100, 200)
        stage_size = (400, 400)

        point = image_point_from_local(ft.Offset(200, 100), image_shape, stage_size)
        outside = image_point_from_local(ft.Offset(200, 50), image_shape, stage_size)

        self.assertEqual(point, (100, 0))
        self.assertIsNone(outside)

    def test_preview_data_uri_is_grayscale_png_and_does_not_change_source(
        self,
    ) -> None:
        """Encode a grayscale preview without mutating its source pixels."""
        image = np.array([[0, 100], [200, 300]], dtype=np.float32)

        uri = image_png_data_uri(image)
        decoded = base64.b64decode(uri.removeprefix("data:image/png;base64,"))
        with PillowImage.open(BytesIO(decoded)) as preview:
            self.assertEqual(preview.mode, "L")
            self.assertEqual(preview.size, (2, 2))
        np.testing.assert_array_equal(image, [[0, 100], [200, 300]])

    def test_spectrum_png_renders_with_matplotlib(self) -> None:
        """Render a labeled spectrum as a usable PNG data URI."""
        uri = spectrum_png_data_uri(
            [4000, 5000],
            [1, 3],
            line_size=2,
            line_type="--",
        )
        decoded = base64.b64decode(uri.removeprefix("data:image/png;base64,"))

        with PillowImage.open(BytesIO(decoded)) as chart:
            self.assertEqual(chart.format, "PNG")
            self.assertGreater(chart.width, 500)
            self.assertGreater(chart.height, 250)

    def test_spectrum_plot_renders_multiple_traces_with_zoom_limits(self) -> None:
        """Render multiple colored traces and honor zoomed wavelength limits."""
        traces = (
            SpectrumTrace(
                np.array([4000, 5000, 6000], dtype=np.float64),
                np.array([1, 2, 3], dtype=np.float32),
                "#ff0000",
                "Image 1",
            ),
            SpectrumTrace(
                np.array([4000, 5000, 6000], dtype=np.float64),
                np.array([3, 2, 1], dtype=np.float32),
                "#008000",
                "Image 2",
            ),
        )
        uri = spectrum_png_data_uri(
            traces=traces,
            x_limits=(4500, 5500),
        )
        decoded = base64.b64decode(uri.removeprefix("data:image/png;base64,"))

        with PillowImage.open(BytesIO(decoded)) as chart:
            self.assertEqual(chart.format, "PNG")
            self.assertGreater(chart.width, 500)

    def test_visible_spectrum_colors_follow_wavelength_and_intensity(self) -> None:
        """Map visible wavelength hues to colors and dim them by intensity."""
        colors = _visible_spectrum_rgb(
            [3800, 5100, 7000, 8000],
            [10, 5, 2, 10],
        )

        np.testing.assert_allclose(colors[0], [0.55, 0, 1])
        np.testing.assert_allclose(colors[1], [0, 0.5, 0])
        np.testing.assert_allclose(colors[2], [0.2, 0, 0])
        np.testing.assert_array_equal(colors[3], [0, 0, 0])

    def test_element_line_markers_are_rendered_in_visible_range(self) -> None:
        """Render light-gray element markers only in the visible range."""
        trace = SpectrumTrace(
            np.array([4000, 4500, 5000, 6000, 7000], dtype=np.float64),
            np.array([1, 2, 3, 4, 5], dtype=np.float32),
            "#38bdf8",
            "Image 1",
        )
        uri, _axes_limits = _render_spectrum_plot(
            traces=(trace,),
            x_limits=(4000, 7000),
            show_element_lines=True,
        )
        decoded = base64.b64decode(uri.removeprefix("data:image/png;base64,"))

        self.assertEqual(set(ELEMENT_LINE_COLORS.values()), {"#d1d5db"})
        with PillowImage.open(BytesIO(decoded)) as chart:
            self.assertEqual(chart.format, "PNG")

    def test_color_band_uses_latest_trace_and_renders_below_spectrum(self) -> None:
        """Color the enabled band from the latest trace's intensity profile."""
        traces = (
            SpectrumTrace(
                np.array([4000, 5000, 6000], dtype=np.float64),
                np.array([100, 100, 100], dtype=np.float32),
                "#ff0000",
                "Image 1",
            ),
            SpectrumTrace(
                np.array([4000, 5000, 6000], dtype=np.float64),
                np.array([0, 5, 10], dtype=np.float32),
                "#008000",
                "Image 2",
            ),
        )
        with patch(
            "plotting._visible_spectrum_rgb",
            wraps=_visible_spectrum_rgb,
        ) as visible_colors:
            uri = spectrum_png_data_uri(
                traces=traces,
                x_limits=(4000, 6000),
                show_color_band=True,
            )

        self.assertEqual(visible_colors.call_count, 1)
        sampled_intensities = visible_colors.call_args.args[1]
        self.assertAlmostEqual(float(np.min(sampled_intensities)), 0, places=5)
        self.assertAlmostEqual(float(np.max(sampled_intensities)), 10, places=5)
        decoded = base64.b64decode(uri.removeprefix("data:image/png;base64,"))
        with PillowImage.open(BytesIO(decoded)) as chart:
            self.assertEqual(chart.format, "PNG")
            self.assertGreater(chart.height, 250)

    def test_spectrum_display_checkboxes_control_plot_options(self) -> None:
        """Regenerate the spectrum when either display checkbox changes."""
        app = QuickSpecFletApp()
        app.build(FakePage())
        app.spectrum_color_checkbox.value = True
        app.spectrum_lines_checkbox.value = True

        with (
            patch.object(app.spectrum_image, "update"),
            patch(
                "flet_app._render_spectrum_plot",
                return_value=("data:image/png;base64,", (0.1, 0.9)),
            ) as render_spectrum,
        ):
            app._on_spectrum_display_option_changed(
                ft.ControlEvent(
                    name="on_change",
                    control=app.spectrum_color_checkbox,
                )
            )

        self.assertTrue(render_spectrum.call_args.kwargs["show_color_band"])
        self.assertTrue(render_spectrum.call_args.kwargs["show_element_lines"])

    def test_app_builds_four_tabs_and_registers_picker(self) -> None:
        """Build all application tabs and register the file picker service."""
        with TemporaryDirectory() as directory:
            with patch("pathlib.Path.home", return_value=Path(directory)):
                app = QuickSpecFletApp()
                page = FakePage()

                tabs = app.build(page)

                self.assertIsInstance(tabs, ft.Tabs)
                self.assertEqual(tabs.length, 4)
                self.assertEqual(page.overlay, [])
                self.assertEqual(page.services, [app.picker])
                self.assertEqual(len(page.controls), 1)
                self.assertIsInstance(app.spectrum_image, ft.Image)
                self.assertTrue(app.spectrum_image.expand)
                self.assertFalse(app.spectrum_color_checkbox.value)
                self.assertFalse(app.spectrum_lines_checkbox.value)
                self.assertIsInstance(app.log_view, ft.ListView)
                self.assertIsInstance(app.debug_checkbox, ft.Checkbox)
                self.assertFalse(app.debug_checkbox.value)
                self.assertEqual(app.star_analyser_radio.value, "200")
                self.assertEqual(app.focal_field.value, "50")
                self.assertEqual(app.lambda_min_field.value, "3500")

    def test_logs_are_colored_and_can_be_cleared(self) -> None:
        """Display logger output in color and clear the Logs tab."""
        app = QuickSpecFletApp()
        app.build(FakePage())
        app._clear_logs()  # écarte les logs de démarrage

        with patch.object(app.page, "update"):
            logging.getLogger("image_processing").info("test log entry")

            self.assertEqual(len(app.log_view.controls), 1)
            self.assertIn("test log entry", app.log_view.controls[0].value)
            self.assertEqual(app.log_view.controls[0].color, "#22c55e")
            self.assertEqual(app.log_view.controls[0].size, 15)

            app._clear_logs()

            self.assertEqual(app.log_view.controls, [])

    def test_log_severity_colors_and_debug_checkbox(self) -> None:
        """Color log levels and toggle DEBUG and TRACE visibility."""
        app = QuickSpecFletApp()
        app.build(FakePage())
        app._clear_logs()  # écarte les logs de démarrage
        image_logger = logging.getLogger("image_processing")
        event = ft.ControlEvent(name="on_change", control=app.debug_checkbox)

        with patch.object(app.page, "update"):
            image_logger.debug("hidden debug message")
            self.assertEqual(app.log_view.controls, [])

            app.debug_checkbox.value = True
            app._on_debug_changed(event)
            image_logger.debug("visible debug message")
            image_logger.log(5, "visible trace message")
            image_logger.warning("warning message")
            image_logger.error("error message")
            image_logger.log(25, "other level message")

        self.assertEqual(len(app.log_view.controls), 5)
        self.assertEqual(
            [control.color for control in app.log_view.controls],
            ["#ffffff", "#ffffff", "#facc15", "#ef4444", "#ffffff"],
        )
        self.assertIn("TRACE", app.log_view.controls[1].value)

        app.debug_checkbox.value = False
        app._on_debug_changed(event)
        with patch.object(app.page, "update"):
            image_logger.debug("hidden again")
        self.assertEqual(len(app.log_view.controls), 5)

    def test_parameters_are_saved_applied_and_loaded(self) -> None:
        """Persist validated parameter values and restore them on next launch."""
        with TemporaryDirectory() as directory:
            with patch(
                "pathlib.Path.home",
                return_value=Path(directory),
            ):
                app = QuickSpecFletApp()
                page = FakePage()
                app.build(page)
                app.star_analyser_radio.value = "100"
                app.focal_field.value = "135"
                app.pixel_size_field.value = "3.76"
                app.lambda_min_field.value = "4000"
                app.lambda_max_field.value = "7000"
                app.line_size_field.value = "3"
                app.line_type_field.value = "--"

                with (
                    patch.object(page, "update"),
                    patch.object(app.spectrum_image, "update"),
                ):
                    app._apply_and_save_parameters(
                        ft.ControlEvent(
                            name="on_click",
                            control=None,
                        )
                    )
                self.assertEqual(app.parameters.star_analyser, 100)
                self.assertEqual(app.spectra_processor.grating_period_nm, 10000)
                self.assertEqual(app.spectra_processor.pixel_size_um, 3.76)
                self.assertEqual(app.spectra_processor.lambda_min, 4000)
                self.assertTrue(app.parameters_file.exists())

                restored = QuickSpecFletApp()
                self.assertEqual(
                    restored.parameters,
                    AppParameters(
                        star_analyser=100,
                        focal_mm=135,
                        pixel_size_um=3.76,
                        lambda_min=4000,
                        lambda_max=7000,
                        line_size=3,
                        line_type="--",
                    ),
                )
                restored.build(FakePage())
                self.assertEqual(restored.line_type_field.value, "--")

    def test_invalid_parameters_are_rejected_and_undo_restores_applied_values(
        self,
    ) -> None:
        """Reject invalid wavelength ranges and restore the last applied form."""
        with TemporaryDirectory() as directory:
            with patch(
                "pathlib.Path.home",
                return_value=Path(directory),
            ):
                app = QuickSpecFletApp()
                page = FakePage()
                app.build(page)
                app.lambda_max_field.value = "3000"

                with patch.object(page, "update"):
                    app._apply_and_save_parameters(
                        ft.ControlEvent(name="on_click", control=None)
                    )

                self.assertEqual(app.parameters.lambda_max, 7500)
                self.assertIn("lambda_max", app.parameter_status.value)
                self.assertFalse(app.parameters_file.exists())

                app._undo_parameters(ft.ControlEvent(name="on_click", control=None))
                self.assertEqual(app.lambda_max_field.value, "7500")

    def test_app_parameters_validate_positive_values_and_line_style(self) -> None:
        """Reject invalid persisted parameter sets before they can be applied."""
        with self.assertRaisesRegex(ValueError, "Focal"):
            AppParameters(focal_mm=0)
        with self.assertRaisesRegex(ValueError, "lambda_max"):
            AppParameters(lambda_min=5000, lambda_max=5000)
        with self.assertRaisesRegex(ValueError, "line_type"):
            AppParameters(line_type="invalid")

    def test_apply_reprocesses_current_spectrum_and_uses_plot_style(self) -> None:
        """Recalibrate an existing extraction and render its configured line style."""
        with TemporaryDirectory() as directory:
            with patch(
                "pathlib.Path.home",
                return_value=Path(directory),
            ):
                app = QuickSpecFletApp()
                page = FakePage()
                app.build(page)
                image = np.full((100, 1500), 2.0, dtype=np.float32)
                image[38:42, 50:] += 10.0
                app.image_processor.data = image
                app.image_shape = image.shape
                app.selection_points = [(50, 40), (1000, 40), (1000, 20)]
                existing_result = app.spectra_processor.process(
                    image,
                    *app.selection_points,
                )
                app._add_spectrum_trace(
                    existing_result.wavelengths,
                    existing_result.flux,
                )
                existing_color = app.spectrum_traces[-1].color
                app.lambda_min_field.value = "1000"
                app.lambda_max_field.value = "10000"
                app.line_size_field.value = "3"
                app.line_type_field.value = ":"

                with (
                    patch.object(page, "update"),
                    patch.object(app.spectrum_image, "update"),
                    patch.object(app.image_status, "update"),
                    patch.object(app.image_stack, "update"),
                    patch(
                        "flet_app._render_spectrum_plot",
                        wraps=_render_spectrum_plot,
                    ) as render_spectrum,
                ):
                    app._apply_and_save_parameters(
                        ft.ControlEvent(name="on_click", control=None)
                    )

                self.assertIsNotNone(app.spectra_processor.result)
                self.assertGreater(
                    app.spectra_processor.result.wavelengths.min(),
                    1000,
                )
                self.assertLess(
                    app.spectra_processor.result.wavelengths.max(),
                    10000,
                )
                self.assertEqual(len(app.spectrum_traces), 1)
                self.assertEqual(app.spectrum_traces[-1].color, existing_color)
                render_spectrum.assert_called_once()
                self.assertEqual(
                    render_spectrum.call_args.kwargs["line_size"],
                    3,
                )
                self.assertEqual(
                    render_spectrum.call_args.kwargs["line_type"],
                    ":",
                )

    def test_spectra_keep_distinct_colors_and_replace_oldest_at_limit(self) -> None:
        """Keep eight uniquely colored traces and replace the oldest on overflow."""
        app = QuickSpecFletApp()
        wavelengths = np.array([4000, 5000], dtype=np.float64)
        flux = np.array([1, 2], dtype=np.float32)
        app.current_image_label = "sample.fit"

        with patch("flet_app.random.choice", side_effect=lambda colors: colors[-1]):
            for index in range(MAX_SPECTRUM_TRACES):
                app.current_image_label = f"sample-{index}.fit"
                app._add_spectrum_trace(wavelengths, flux)

            first_label = app.spectrum_traces[0].label
            first_color = app.spectrum_traces[0].color
            app.current_image_label = "ninth.fit"
            app._add_spectrum_trace(wavelengths, flux)

        self.assertEqual(len(app.spectrum_traces), MAX_SPECTRUM_TRACES)
        self.assertEqual(
            len({trace.color for trace in app.spectrum_traces}),
            MAX_SPECTRUM_TRACES,
        )
        self.assertNotIn(first_label, [trace.label for trace in app.spectrum_traces])
        self.assertEqual(app.spectrum_traces[-1].color, first_color)
        self.assertEqual(
            {trace.color for trace in app.spectrum_traces},
            set(SPECTRUM_COLORS),
        )

    def test_spectrum_drag_zoom_regenerates_plot_and_reset_restores_full_range(
        self,
    ) -> None:
        """Zoom to a dragged wavelength interval and restore the full plot."""
        app = QuickSpecFletApp()
        app.build(FakePage())
        app._add_spectrum_trace(
            np.arange(4000, 7001, dtype=np.float64),
            np.linspace(1, 3, 3001, dtype=np.float32),
        )
        width, height = 770.0, 440.0
        with patch.object(app.spectrum_image, "update"):
            app._on_spectrum_plot_size_change(
                ft.LayoutSizeChangeEvent(
                    name="on_size_change",
                    control=app.spectrum_gesture,
                    width=width,
                    height=height,
                )
            )
            app._render_spectrum()
        axes_left, axes_right = app._spectrum_axes_x_limits
        plot_width = axes_right - axes_left
        start = width * (axes_left + plot_width * 0.2)
        end = width * (axes_left + plot_width * 0.7)
        with (
            patch.object(app.page, "update"),
            patch.object(app.spectrum_image, "update"),
            patch.object(app.spectrum_zoom_status, "update"),
            patch(
                "flet_app._render_spectrum_plot",
                wraps=_render_spectrum_plot,
            ) as render_spectrum,
        ):
            app._on_spectrum_pan_start(
                ft.DragStartEvent(
                    name="on_pan_start",
                    control=app.spectrum_gesture,
                    kind=ft.PointerDeviceType.MOUSE,
                    local_position=ft.Offset(start, 100),
                    global_position=ft.Offset(start, 100),
                )
            )
            render_spectrum.assert_not_called()
            app.page.update.assert_not_called()

            app._on_spectrum_pan_end(
                ft.DragEndEvent(
                    name="on_pan_end",
                    control=app.spectrum_gesture,
                    local_position=ft.Offset(end, 100),
                    global_position=ft.Offset(end, 100),
                    velocity=ft.Offset(0, 0),
                )
            )
            self.assertIsNotNone(app.spectrum_xlim)
            self.assertLess(
                app.spectrum_xlim[1] - app.spectrum_xlim[0],
                app._spectrum_data_limits()[1] - app._spectrum_data_limits()[0],
            )
            render_spectrum.assert_called_once()
            self.assertEqual(
                render_spectrum.call_args.kwargs["x_limits"],
                app.spectrum_xlim,
            )

            app._reset_spectrum_zoom(
                ft.ControlEvent(name="on_click", control=None)
            )
        self.assertIsNone(app.spectrum_xlim)
        self.assertIsNone(app.spectrum_xlim)
        self.assertEqual(render_spectrum.call_count, 2)

    def test_loading_another_image_preserves_traces_and_clear_removes_them(
        self,
    ) -> None:
        """Preserve plotted spectra across image loads until Clear is pressed."""
        app = QuickSpecFletApp()
        app.build(FakePage())
        app._add_spectrum_trace(
            np.array([4000, 5000], dtype=np.float64),
            np.array([1, 2], dtype=np.float32),
        )
        image_path = Path("/tmp/quickspec-preserve-traces.png")
        image_data = np.ones((20, 20), dtype=np.float32)
        app.image_processor.load_stack = lambda _paths, **_masters: image_data
        app._refresh_image_controls = lambda _data=None: None

        with (
            patch("flet_app.load_masters", return_value={"offset": None, "dark": None, "flat": None}),
            patch.object(app.image_status, "update"),
            patch.object(app.spectrum_status, "update"),
            patch.object(app.spectrum_image, "update"),
        ):
            app._load_images([image_path])

        self.assertEqual(len(app.spectrum_traces), 1)
        with patch.object(app.spectrum_image, "update"):
            app._clear_spectrum()
        self.assertEqual(app.spectrum_traces, [])
        self.assertIsNone(app.spectrum_xlim)

    def test_three_image_taps_extract_spectrum_and_open_spectrum_tab(
        self,
    ) -> None:
        """Extract a spectrum after the three selection taps."""
        app = QuickSpecFletApp()
        page = FakePage()
        app.build(page)
        image = np.full((100, 240), 2.0, dtype=np.float32)
        image[38:42, 50:] += 10.0
        app.image_processor.data = image
        app.image_shape = image.shape
        app.spectra_processor = Spectra(
            lambda_min=0,
            lambda_max=10000,
        )
        app.selecting = True
        scale = min(app.stage_size[0] / image.shape[1], app.stage_size[1] / image.shape[0])
        left = (app.stage_size[0] - image.shape[1] * scale) / 2
        top = (app.stage_size[1] - image.shape[0] * scale) / 2
        pixels = ((50, 40), (180, 40), (180, 20))

        with (
            patch.object(app.image_stack, "update"),
            patch.object(app.image_status, "update"),
            patch.object(app.spectrum_image, "update"),
        ):
            for x, y in pixels:
                app._on_image_tap(
                    ft.TapEvent(
                        name="on_tap",
                        control=None,
                        local_position=ft.Offset(left + x * scale, top + y * scale),
                    )
                )

        self.assertIsNotNone(app.spectra_processor.result)
        self.assertGreater(app.spectra_processor.result.flux.size, 0)
        self.assertEqual(app.tabs.selected_index, 0)  # on reste sur l'image
        self.assertTrue(app.show_rectified)
        self.assertTrue(app.rectified_checkbox.value)
        lines = [c for c in app.image_stack.controls if isinstance(c, ft.Container)]
        self.assertEqual(len(lines), 4)  # 2 lignes spectre + 2 lignes ciel
        self.assertEqual({c.bgcolor for c in lines}, {"#ff2020", "#3b82f6"})
        self.assertEqual(len(app.image_stack.controls), 5)  # image + 4 lignes, plus de croix

    def test_last_picker_directory_is_persisted(self) -> None:
        """Persist the most recently used image directory."""
        app = QuickSpecFletApp()
        with TemporaryDirectory() as directory:
            app.location_file = Path(directory) / "settings" / "last-image-location.txt"
            app.last_directory = directory

            app._save_last_directory()

            self.assertEqual(app.location_file.read_text(), directory)


class DynamicSizeTests(unittest.TestCase):
    """Vérifie que image et graphe suivent la taille mesurée de leur conteneur."""

    def test_spectrum_plot_png_has_requested_size(self) -> None:
        """Le PNG du graphe a la taille demandée (à un pixel près)."""
        uri, _ = _render_spectrum_plot(size_px=(1100, 600))
        decoded = base64.b64decode(uri.removeprefix("data:image/png;base64,"))
        with PillowImage.open(BytesIO(decoded)) as chart:
            self.assertEqual(chart.size, (1100, 600))

    def test_spectrum_size_change_rerenders_once_per_real_change(self) -> None:
        """Re-rend le graphe à la nouvelle taille, mais pas pour une variation < 2 px."""
        app = QuickSpecFletApp()
        app.build(FakePage())

        def event(width: float, height: float) -> ft.LayoutSizeChangeEvent:
            return ft.LayoutSizeChangeEvent(
                name="on_size_change", control=app.spectrum_gesture, width=width, height=height
            )

        with (
            patch.object(app.spectrum_image, "update"),
            patch("flet_app._render_spectrum_plot", wraps=_render_spectrum_plot) as render,
        ):
            app._on_spectrum_plot_size_change(event(900, 500))
            app._on_spectrum_plot_size_change(event(900.5, 500.5))
            app._on_spectrum_plot_size_change(event(0, 0))  # onglet masqué
            app._on_spectrum_plot_size_change(event(1200, 700))

        self.assertEqual(render.call_count, 2)
        self.assertEqual(render.call_args_list[0].kwargs["size_px"], (900.0, 500.0))
        self.assertEqual(render.call_args_list[1].kwargs["size_px"], (1200.0, 700.0))

    def test_image_stage_follows_measured_size(self) -> None:
        """La zone image prend la taille mesurée du conteneur, et ignore une taille nulle."""
        app = QuickSpecFletApp()
        app.build(FakePage())
        with patch.object(app.image_stack, "update"):
            app._on_image_stage_size_change(
                ft.LayoutSizeChangeEvent(name="on_size_change", control=None, width=1000, height=640)
            )
            app._on_image_stage_size_change(
                ft.LayoutSizeChangeEvent(name="on_size_change", control=None, width=0, height=0)
            )
        self.assertEqual(app.stage_size, (1000.0, 640.0))
        self.assertEqual((app.image_stack.width, app.image_stack.height), (1000.0, 640.0))

    def test_parameters_missing_keys_use_defaults(self) -> None:
        """Un ancien fichier JSON sans nouvelle clé reste lisible."""
        parameters = AppParameters.from_mapping({"focal_mm": 85})
        self.assertEqual(parameters.focal_mm, 85)
        self.assertEqual(parameters.star_analyser, 200)


class ContrastTests(unittest.TestCase):
    """L'image transformée doit avoir le même contraste que l'image d'origine."""

    def test_rectified_preview_uses_original_stretch(self) -> None:
        """Les coins noirs de la rotation ne doivent pas éclaircir l'aperçu."""
        original = np.full((50, 50), 100.0, dtype=np.float32)
        original[20:30, 20:30] = 400.0
        rectified = original.copy()
        rectified[:15] = 0.0  # coins vides créés par la rotation

        def background_gray(data, limits=None) -> int:
            decoded = base64.b64decode(image_png_data_uri(data, limits).split(",")[1])
            with PillowImage.open(BytesIO(decoded)) as image:
                return image.getpixel((40, 40))

        shared = display_limits(original)
        self.assertEqual(
            background_gray(original), background_gray(rectified, shared)
        )
        # sans bornes communes, le fond apparaît nettement plus clair
        self.assertGreater(background_gray(rectified), background_gray(original) + 20)


class MultiImageLoadingTests(unittest.TestCase):
    """Chargement de plusieurs images avec masters dans l'application."""

    def setUp(self) -> None:
        from test_stacking import save_gray16, star_field

        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        shape = (160, 240)
        save_gray16(self.directory / "master_offset.png", np.full(shape, 100))
        save_gray16(self.directory / "master_dark.png", np.full(shape, 20))
        save_gray16(self.directory / "master_flat.png", np.full(shape, 4000))
        self.paths = [
            save_gray16(self.directory / f"light{i}.png", star_field(dy=dy, dx=dx, seed=i) + 120)
            for i, (dy, dx) in enumerate([(0, 0), (3.0, -2.0), (-4.0, 5.0)])
        ]
        self.app = QuickSpecFletApp()
        self.page = FakePage()
        self.app.build(self.page)
        self.app.parameters_file = self.directory / "parameters.json"
        self.updates = (
            patch.object(self.page, "update"),
            patch.object(self.app.image_status, "update"),
            patch.object(self.app.spectrum_status, "update"),
            patch.object(self.app.spectrum_image, "update"),
            patch.object(self.app.image_stack, "update"),
        )
        for update in self.updates:
            update.start()
            self.addCleanup(update.stop)

    def test_masters_are_applied_and_frames_registered_and_summed(self) -> None:
        self.app._load_images(self.paths)

        processor = self.app.image_processor
        self.assertEqual(processor.applied_corrections, ["offset", "dark", "flat"])
        self.assertAlmostEqual(processor.registration_shifts[1][0], -3.0, delta=0.15)
        self.assertAlmostEqual(processor.registration_shifts[1][1], 2.0, delta=0.15)
        self.assertAlmostEqual(processor.registration_shifts[2][0], 4.0, delta=0.15)
        self.assertAlmostEqual(processor.registration_shifts[2][1], -5.0, delta=0.15)
        self.assertEqual(self.app.current_image_label, "light0.png (+2)")
        status = self.app.image_status.value
        self.assertIn("3 images recalées et additionnées", status)
        self.assertIn("corrections : offset, dark, flat", status)

    def test_missing_master_logs_a_warning_and_load_continues(self) -> None:
        (self.directory / "master_dark.png").unlink()
        with self.assertLogs("calibration", "WARNING") as logs:
            self.app._load_images(self.paths[:1])
        self.assertTrue(any("Master dark introuvable" in line for line in logs.output))
        self.assertEqual(self.app.image_processor.applied_corrections, ["offset", "flat"])
        self.assertIn("corrections : offset, flat", self.app.image_status.value)

    def test_masters_directory_parameter_overrides_the_image_folder(self) -> None:
        masters = self.directory / "masters"
        masters.mkdir()
        for name in ("master_offset.png", "master_dark.png", "master_flat.png"):
            (self.directory / name).rename(masters / name)
        with self.assertLogs("calibration", "WARNING"):
            self.app._load_images(self.paths[:1])  # masters introuvables à côté de l'image
        self.assertEqual(self.app.image_processor.applied_corrections, [])

        self.app.parameters = AppParameters(calibration_dir=str(masters))
        self.app._load_images(self.paths[:1])
        self.assertEqual(self.app.image_processor.applied_corrections, ["offset", "dark", "flat"])

    def test_changing_a_master_name_reloads_images_and_keeps_the_selection(self) -> None:
        self.app._load_images(self.paths[:1])
        self.app.selection_points[:] = [(50.0, 40.0), (180.0, 40.0), (180.0, 20.0)]
        self.app.dark_name_field.value = ""  # désactive le dark

        with patch.object(self.app, "_extract_spectrum") as extract:
            self.app._apply_and_save_parameters(ft.ControlEvent(name="on_click", control=None))

        self.assertEqual(self.app.image_processor.applied_corrections, ["offset", "flat"])
        self.assertEqual(len(self.app.selection_points), 3)
        extract.assert_called_once()

    def test_band_thickness_parameters_reach_the_spectra_processor(self) -> None:
        self.app.spectrum_height_field.value = "7"
        self.app.sky_height_field.value = "15"
        self.app._apply_and_save_parameters(ft.ControlEvent(name="on_click", control=None))
        self.assertEqual(self.app.spectra_processor.spectrum_height, 7)
        self.assertEqual(self.app.spectra_processor.sky_height, 15)


class ContrastCutsTests(unittest.TestCase):
    """Curseurs de contraste (noir/blanc en percentiles)."""

    def test_cut_scale_endpoints_and_inverse(self) -> None:
        from plotting import cut_percentile, cut_position

        self.assertAlmostEqual(cut_percentile(0), 0.0)
        self.assertAlmostEqual(cut_percentile(100), 100.0)
        for percentile in (0.5, 20.0, 50.0, 99.0, 99.5, 99.9):
            self.assertAlmostEqual(cut_percentile(cut_position(percentile)), percentile, places=6)
        # l'échelle est dilatée près de 100 % : 99–100 % occupe ~14 % de la course
        self.assertGreater(100 - cut_position(99.0), 10)

    def test_cut_limits_are_robust_to_hot_pixels(self) -> None:
        from plotting import cut_limits, cut_sample

        image = np.random.default_rng(0).normal(1000, 20, (200, 300)).astype(np.float32)
        image[10, 10] = 1e6  # pixel chaud
        low, high = cut_limits(cut_sample(image), 0.5, 99.5)
        self.assertLess(high, 1200)
        self.assertLess(low, high)

    def test_sample_is_bounded_and_reproducible(self) -> None:
        from plotting import CUT_SAMPLE_SIZE, cut_sample

        image = np.arange(2000 * 1000, dtype=np.float32).reshape(1000, 2000)
        first, second = cut_sample(image), cut_sample(image)
        self.assertEqual(first.size, CUT_SAMPLE_SIZE)
        np.testing.assert_array_equal(first, second)

    def test_toolbar_order_buttons_slider_checkbox(self) -> None:
        app = QuickSpecFletApp()
        app.build(FakePage())
        toolbar = app.tabs.content.controls[1].controls[0].controls[0].controls
        self.assertEqual(toolbar[0].content, "Ouvrir une image")
        self.assertEqual(toolbar[1].content, "Sélection SA100")
        self.assertIn(app.cut_slider, toolbar[2].controls)
        self.assertIs(toolbar[3], app.rectified_checkbox)

    def test_slider_release_recomputes_levels_and_auto_restores_them(self) -> None:
        from plotting import cut_position

        app = QuickSpecFletApp()
        app.build(FakePage())
        image = np.linspace(0, 1000, 100 * 100, dtype=np.float32).reshape(100, 100)
        app.image_processor.data = image
        app.image_shape = image.shape
        with (
            patch.object(app.page, "update"),
            patch.object(app.image_stack, "update"),
            patch.object(app.cut_text, "update"),
        ):
            auto = app._stretch_limits()
            app.cut_slider.start_value, app.cut_slider.end_value = 20.0, 60.0
            app._on_cuts_changed(ft.ControlEvent(name="on_change_end", control=app.cut_slider))
            changed = app._stretch_limits()
            self.assertGreater(changed[0], auto[0])
            self.assertLess(changed[1], auto[1])

            app._on_cuts_auto(ft.ControlEvent(name="on_click", control=None))
            restored = app._stretch_limits()
        self.assertEqual(restored, auto)
        self.assertAlmostEqual(app.cut_slider.end_value, cut_position(99.5))


if __name__ == "__main__":
    unittest.main()

import base64
from io import BytesIO

import flet as ft
import matplotlib
matplotlib.use("Agg")

import numpy as np
import scipy
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.signal import find_peaks


def main(page: ft.Page):
    page.title = "QuickSpec — test Flet"
    page.theme_mode = ft.ThemeMode.LIGHT

    x = np.linspace(0, 10, 500)
    y = np.sin(x) + 0.2 * np.sin(4 * x)
    peaks, _ = find_peaks(y, prominence=0.3)
    figure = Figure(figsize=(6, 3), dpi=120)
    axes = figure.add_subplot(111)
    axes.plot(x, y, color="#1565c0")
    axes.scatter(x[peaks], y[peaks], color="#d32f2f", s=18)
    axes.set(title="Test NumPy + SciPy + Matplotlib", xlabel="x", ylabel="signal")
    axes.grid(True, alpha=0.25)
    output = BytesIO()
    FigureCanvasAgg(figure).print_png(output)
    chart = ft.Image(
        src=f"data:image/png;base64,{base64.b64encode(output.getvalue()).decode('ascii')}",
        fit=ft.BoxFit.CONTAIN,
        expand=True,
    )
    status = ft.Text("Bibliothèques scientifiques chargées.")

    def increment_click(e: ft.Event[ft.FloatingActionButton]):
        status.value = "Flet fonctionne avec les bibliothèques scientifiques."
        status.update()

    page.floating_action_button = ft.FloatingActionButton(
        icon=ft.Icons.ADD, key="increment", on_click=increment_click
    )
    page.add(
        ft.SafeArea(
            expand=True,
            content=ft.Container(
                content=ft.Column(
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Text("QuickSpec SA100", size=24, weight=ft.FontWeight.BOLD),
                        ft.Text("Widgets Flet + bibliothèques scientifiques Android"),
                        chart,
                        status,
                        ft.Text("Appuie sur + pour tester un événement."),
                    ],
                ),
                alignment=ft.Alignment.CENTER,
            ),
        )
    )


if __name__ == "__main__":
    ft.run(main)

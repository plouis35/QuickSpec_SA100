import flet.testing as ftt


async def test_increment(flet_app: ftt.FletTestApp):
    """Verify the scientific chart is visible and the Flet button responds.

    The `flet_app` fixture is provided automatically by the flet pytest plugin.
    Run with `flet test` (or `uv run pytest`) from the project directory.
    """
    tester = flet_app.tester

    await tester.pump_and_settle()

    assert (
        await tester.find_by_text("Widgets Flet + bibliothèques scientifiques Android")
    ).count == 1
    assert (await tester.find_by_text("Bibliothèques scientifiques chargées.")).count == 1

    await tester.tap(await tester.find_by_key("increment"))
    await tester.pump_and_settle()

    assert (
        await tester.find_by_text(
            "Flet fonctionne avec les bibliothèques scientifiques."
        )
    ).count == 1

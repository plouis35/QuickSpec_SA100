"""Recherche et lecture des images maîtresses (master offset, dark, flat)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

from image_io import ImageArray, read_image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Ordre de priorité quand plusieurs fichiers portent le même nom de base.
MASTER_EXTENSIONS = (".fits", ".fit", ".fts", ".png", ".jpg", ".jpeg")


def find_master(directory: Path, name: str) -> Path | None:
    """Trouve le fichier d'un master à partir de son nom (avec ou sans extension).

    Args:
        directory: Répertoire de recherche (ignoré si ``name`` est un chemin absolu).
        name: Nom de base (``master_dark``) ou nom complet (``master_dark.fits``).

    Returns:
        Le fichier trouvé (FITS en priorité, puis PNG, puis JPEG), ou ``None``.
    """
    candidate = directory / name
    if candidate.suffix.lower() in MASTER_EXTENSIONS and candidate.is_file():
        return candidate
    found = [
        directory / f"{name}{variant}"
        for extension in MASTER_EXTENSIONS
        for variant in (extension, extension.upper())
        if (directory / f"{name}{variant}").is_file()
    ]
    if len(found) > 1:
        logger.warning(
            "Plusieurs fichiers pour « %s » : %s ; utilisé : %s",
            name, ", ".join(path.name for path in found), found[0].name,
        )
    return found[0] if found else None


def load_masters(directory: Path, names: Mapping[str, str]) -> dict[str, ImageArray | None]:
    """Charge les masters demandés ; un master absent ou illisible vaut ``None`` (avec un warning).

    Args:
        directory: Répertoire des masters.
        names: Nom de chaque master, par exemple ``{"offset": "master_offset", ...}``.
            Un nom vide désactive la correction correspondante, sans warning.

    Returns:
        Un tableau (ou ``None``) par clé de ``names``.
    """
    logger.info("Recherche des masters dans %s", directory)
    masters: dict[str, ImageArray | None] = {}
    directory_ok = directory.is_dir()
    if not directory_ok:
        logger.warning("Répertoire des masters introuvable : %s", directory)
    for kind, name in names.items():
        masters[kind] = None
        if not name:
            logger.info("Master %s désactivé (nom vide)", kind)
            continue
        if not directory_ok:
            continue
        path = find_master(directory, name)
        if path is None:
            logger.warning(
                "Master %s introuvable : %s (extensions cherchées : %s)",
                kind, directory / name, ", ".join(MASTER_EXTENSIONS[::2] + MASTER_EXTENSIONS[3:4]),
            )
            continue
        try:
            masters[kind] = read_image(path)
        except (OSError, ValueError) as exc:
            logger.warning("Master %s illisible (%s) : %s", kind, path.name, exc)
            continue
        logger.info("Master %s : %s (%d × %d px)", kind, path.name, *reversed(masters[kind].shape))
    return masters

"""Paramètres utilisateur de QuickSpec SA100 et leur persistance (sans dépendance à Flet)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path

from plotting import VALID_LINE_TYPES

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class ParamSpec:
    """Description d'un paramètre : sert à générer le formulaire et à valider les saisies."""

    name: str  # champ de AppParameters
    label: str  # libellé du formulaire
    kind: type  # int, float ou str
    widget: str  # nom de l'attribut du contrôle dans l'application
    error_name: str  # nom utilisé dans les messages d'erreur
    section: str = ""  # titre affiché au-dessus du champ quand la section change
    strip: bool = True  # retire les espaces autour d'une saisie texte


# Pour ajouter un paramètre : un champ dans AppParameters + une ligne ici (+ sa règle dans __post_init__).
PARAM_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("star_analyser", "Star Analyser", int, "star_analyser_radio", "Star Analyser",
              section="Réseau et optique"),
    ParamSpec("focal_mm", "Focal (mm)", int, "focal_field", "Focal"),
    ParamSpec("pixel_size_um", "Pixel_size capteur natif (µm)", float, "pixel_size_field", "Pixel_size"),
    ParamSpec("lambda_min", "lambda_min (Å)", int, "lambda_min_field", "lambda_min",
              section="Fenêtre en longueur d’onde"),
    ParamSpec("lambda_max", "lambda_max (Å)", int, "lambda_max_field", "lambda_max"),
    ParamSpec("centroid_radius", "Affinage de l’ordre zéro : rayon (px, 0 = désactivé)", int,
              "centroid_radius_field", "centroid_radius", section="Extraction du spectre"),
    ParamSpec("spectrum_height", "Épaisseur de la bande du spectre (px)", int,
              "spectrum_height_field", "spectrum_height"),
    ParamSpec("sky_height", "Épaisseur de la bande du fond de ciel (px)", int,
              "sky_height_field", "sky_height"),
    ParamSpec("calibration_dir", "Répertoire des masters (vide = dossier de l’image)", str,
              "calibration_dir_field", "Répertoire des masters",
              section="Corrections offset / dark / flat"),
    ParamSpec("offset_name", "Master offset (nom, sans extension)", str,
              "offset_name_field", "Master offset"),
    ParamSpec("dark_name", "Master dark (nom, sans extension)", str,
              "dark_name_field", "Master dark"),
    ParamSpec("flat_name", "Master flat (nom, sans extension)", str,
              "flat_name_field", "Master flat"),
    ParamSpec("line_size", "line_size", int, "line_size_field", "line_size",
              section="Affichage du spectre"),
    ParamSpec("line_type", "line_type (ls=...)", str, "line_type_field", "line_type", strip=False),
)

# Paramètres dont la modification impose de recharger les images (corrections appliquées au chargement).
CALIBRATION_FIELDS = ("calibration_dir", "offset_name", "dark_name", "flat_name")


def _has_type(value: object, kind: type) -> bool:
    """Vrai si ``value`` est du type attendu (un booléen n'est jamais un nombre)."""
    if isinstance(value, bool):
        return False
    if kind is float:
        return isinstance(value, (int, float))
    return isinstance(value, kind)


@dataclass(frozen=True)
class AppParameters:
    """Paramètres optiques, de longueur d'onde et d'affichage, validés à la création."""

    star_analyser: int = 200
    focal_mm: int = 50
    pixel_size_um: float = 6.54  # Canon 6D
    lambda_min: int = 3500
    lambda_max: int = 7500
    line_size: int = 1
    line_type: str = "-"
    centroid_radius: int = 8
    spectrum_height: int = 10
    sky_height: int = 20
    calibration_dir: str = ""
    offset_name: str = "master_offset"
    dark_name: str = "master_dark"
    flat_name: str = "master_flat"

    def __post_init__(self) -> None:
        """Rejette les types et valeurs invalides."""
        for spec in PARAM_SPECS:
            if not _has_type(getattr(self, spec.name), spec.kind):
                kind = {int: "an integer", float: "a number", str: "text"}[spec.kind]
                raise ValueError(f"{spec.error_name} must be {kind}.")
        if self.star_analyser not in (100, 200):
            raise ValueError("Star Analyser must be 100 or 200 traits/mm.")
        if self.focal_mm <= 0:
            raise ValueError("Focal must be an integer greater than zero.")
        if not math.isfinite(self.pixel_size_um) or self.pixel_size_um <= 0:
            raise ValueError("Pixel_size must be a finite number greater than zero.")
        if self.lambda_min <= 0 or self.lambda_max <= 0:
            raise ValueError("lambda_min and lambda_max must be greater than zero.")
        if self.lambda_max <= self.lambda_min:
            raise ValueError("lambda_max must be greater than lambda_min.")
        if self.line_size <= 0:
            raise ValueError("line_size must be an integer greater than zero.")
        if self.centroid_radius < 0:
            raise ValueError("centroid_radius must be an integer greater than or equal to zero.")
        if self.spectrum_height <= 0 or self.sky_height <= 0:
            raise ValueError("spectrum_height and sky_height must be integers greater than zero.")
        if self.line_type not in VALID_LINE_TYPES:
            raise ValueError("line_type must be a supported Matplotlib line style.")

    @classmethod
    def from_mapping(cls, values: object) -> AppParameters:
        """Crée les paramètres depuis un JSON décodé ; les clés absentes prennent leur défaut.

        Raises:
            ValueError: Si ``values`` n'est pas un objet JSON ou contient une valeur invalide.
        """
        if not isinstance(values, dict):
            raise ValueError("The saved parameters must be a JSON object.")
        defaults = cls()
        kwargs = {}
        for spec in PARAM_SPECS:
            if spec.name not in values:
                logger.info("Paramètre « %s » absent du fichier : valeur par défaut", spec.name)
                kwargs[spec.name] = getattr(defaults, spec.name)
            elif _has_type(values[spec.name], spec.kind):
                kwargs[spec.name] = values[spec.name]
            else:
                raise ValueError(f"Saved {spec.name} has an invalid JSON type.")
        return cls(**kwargs)

    @classmethod
    def from_form(cls, raw: dict[str, str]) -> AppParameters:
        """Crée les paramètres depuis les textes saisis dans le formulaire.

        Raises:
            ValueError: Si un champ est vide, mal formé ou hors limites.
        """
        kwargs = {}
        for spec in PARAM_SPECS:
            text = raw.get(spec.name) or ""
            if spec.strip:
                text = text.strip()
            try:
                kwargs[spec.name] = spec.kind(text)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{spec.error_name} : valeur « {text} » invalide") from exc
        return cls(**kwargs)


def load_parameters(path: Path) -> tuple[AppParameters, str | None]:
    """Charge les paramètres enregistrés.

    Returns:
        Les paramètres (par défaut si le fichier est absent ou invalide) et un
        message d'erreur destiné à l'utilisateur, ou ``None``.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.info("Pas de fichier de paramètres (%s) : valeurs par défaut", path)
        return AppParameters(), None
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("Lecture des paramètres impossible : %s", path)
        return AppParameters(), f"Impossible de lire les paramètres enregistrés : {exc}"
    try:
        parameters = AppParameters.from_mapping(raw)
    except ValueError as exc:
        logger.exception("Paramètres enregistrés invalides : %s", path)
        return AppParameters(), f"Paramètres enregistrés invalides : {exc}"
    logger.info("Paramètres chargés depuis %s : %s", path, parameters)
    return parameters, None


def save_parameters(path: Path, parameters: AppParameters) -> None:
    """Écrit les paramètres en JSON de façon atomique.

    Raises:
        OSError: Si le fichier ne peut pas être écrit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(asdict(parameters), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    logger.info("Paramètres enregistrés dans %s", path)


def read_last_directory(location_file: Path) -> str:
    """Dernier répertoire d'image ouvert s'il existe encore, sinon le dossier personnel."""
    try:
        location = location_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        location = ""
    except OSError:
        logger.warning("Lecture du dernier répertoire impossible : %s", location_file)
        location = ""
    if location and Path(location).is_dir():
        logger.info("Dernier répertoire restauré : %s", location)
        return location
    logger.info("Pas de dernier répertoire valide : dossier personnel")
    return str(Path.home())


def write_last_directory(location_file: Path, directory: str) -> None:
    """Mémorise le dernier répertoire d'image ouvert."""
    try:
        location_file.parent.mkdir(parents=True, exist_ok=True)
        location_file.write_text(directory, encoding="utf-8")
        logger.debug("Dernier répertoire mémorisé : %s", directory)
    except OSError:
        logger.warning("Enregistrement du dernier répertoire impossible", exc_info=True)

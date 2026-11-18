# QuickSpec SA100 — Flet / Matplotlib

Cette application Android utilise Flet pour l’interface et Matplotlib pour le
graphique du spectre.

## Code

- `flet_app.py` — ouverture d’images, zoom/panoramique, sélection des trois
  points SA100 et affichage du spectre.
- `image_io.py` — lecture PNG, JPG, TIFF, FITS 2D simple et FITS gzip, sans
  dépendance Astropy.
- `image_processing.py` — intégration des images lues en tableaux NumPy,
  combinaison, corrections offset/dark/flat et registration.
- `spectra_processing.py` — redressement, extraction avec soustraction du ciel
  et calibration linéaire SA100.
- `pyproject.toml` — dépendances et configuration Flet Android arm64.

Le lecteur FITS accepte un HDU primaire contenant une image 2D non compressée
avec `BITPIX` 8, 16, 32, 64, -32 ou -64 ; `BSCALE`, `BZERO` et `BLANK` sont pris
en compte. Les fichiers `.fit`, `.fits` et `.fts`, éventuellement terminés par
`.gz`, sont acceptés. Les cubes, extensions HDU supplémentaires, tables et
images FITS compressées en tuiles ne sont pas pris en charge.

Les TIFF monochromes conservent leurs valeurs et leur résolution, y compris en
16 bits. Les JPG et TIFF couleur sont convertis en luminance puis binés par
somme de blocs 2×2 ; une dernière ligne ou colonne impaire est ignorée. Les JPG
sont avec perte et ne sont donc pas recommandés pour des mesures précises. Les
PNG gardent le comportement existant. Les paramètres optiques et de sélection
utilisent les valeurs par défaut de `Spectra`.

## Lancer sur Linux

Depuis ce dossier, installe les dépendances dans un environnement Python :

```bash
python3 -m venv .venv-flet
source .venv-flet/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-flet.txt
flet run --module flet_app
```

Le sélecteur de fichiers Flet sous Linux requiert **Zenity**. Dans l’onglet
**Image**, ouvre un PNG, JPG, TIFF ou FITS, choisis **Sélection SA100**, puis
clique l’ordre zéro, la trace et le fond de ciel. Après l’extraction, l’onglet
**Spectre** affiche le spectre calibré en longueur d’onde (Å) et en flux.
Chaque nouvelle extraction ajoute une courbe d’une couleur distincte sans
effacer les précédentes, y compris après ouverture d’une nouvelle image. Huit
courbes au maximum sont affichées ; la neuvième remplace la plus ancienne.
Glisse horizontalement sur le graphe pour zoomer sur une plage de longueurs
d’onde ; le graphe est régénéré au relâchement. **Reset zoom** restaure
l’échelle complète et **Clear** efface toutes les courbes. Une bande de
couleurs visibles peut être activée sous le graphe avec **Couleurs** ; sa
luminosité suit l’intensité du spectre le plus récent. La case **Raies
H/He/Na/Mg/Ca** affiche les principales raies optiques de ces éléments dans
la plage visible, avec leur identification et leur longueur d’onde. Les deux
options sont décochées au démarrage.
L’onglet **Logs** affiche les messages du Logger en couleur (INFO en vert,
WARNING en jaune, erreurs en rouge) et son bouton **Clear** efface l’affichage
courant. La case **Debug** active aussi les niveaux DEBUG et TRACE.
L’onglet **Parameters** permet de régler le réseau Star Analyser, la focale,
la taille des pixels, la plage de longueurs d’onde et le style du tracé.
**Apply and save** valide et conserve les valeurs dans
`~/.quickspec-sa100/parameters.json` ; **Undo** restaure les dernières valeurs
appliquées. La focale (50 mm par défaut) sera utilisée lors de l’étape 4.

## Tests

```bash
python -m unittest -v test_flet_app test_image_io test_processing
```

## Générer l’APK

Depuis ce dossier :

```bash
flet build apk --arch arm64-v8a
```

L’APK est créé dans `build/apk/`. Il embarque Python 3.14, NumPy, SciPy,
Matplotlib et Pillow. Matplotlib est extrait dans l’espace privé de
l’application afin que ses polices soient accessibles sur Android.

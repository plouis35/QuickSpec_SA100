# QuickSpec SA100 

Cette application MacOS/Linux/Android ....



## Lancer sur Linux

Depuis ce dossier, installe les dépendances dans un environnement Python :

```bash
python3 -m venv .venv-flet
source .venv-flet/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-flet.txt
flet run --module flet_app
```



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

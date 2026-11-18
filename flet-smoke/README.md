# QuickSpec — test Flet

Application de test réduite pour vérifier les widgets Flet et l’empaquetage
des bibliothèques `numpy`, `scipy`, `matplotlib` et `Pillow` dans un APK.
Elle génère un petit signal, repère ses pics avec SciPy, puis affiche le
graphique Matplotlib. Elle ne contient pas encore le traitement QuickSpec.

## Tester sur le poste Linux

Depuis ce dossier, avec l’environnement qui contient Flet :

```bash
flet run src/main.py
```

## Générer l’APK (à faire après validation)

Depuis ce dossier :

```bash
flet build apk --arch arm64-v8a
```

Le projet utilise l’identifiant Android `fr.plouis.quickspec-flet-smoke`.
Flet télécharge Flutter et installe les outils du SDK Android requis si
nécessaire. Le projet limite la mémoire utilisée par Gradle pour les machines
à mémoire modeste.

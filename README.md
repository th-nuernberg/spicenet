![](media/spice-net-logo-green-ai.png)

# SPICEnet in Python

[(Original Repo)](https://github.com/102-97-98-105/spicenet-embedded)

Im Ordner python_package finden sich die von mir geschriebenen Dateien. Ich habe ursprünglich das
Repo https://github.com/th-nuernberg/spicenet geforked.

## Einordnung der Notebooks

- [tests](/code/python_package/tests.ipynb): Das Notebook ist vor allem während der Entwicklung entstanden und als
  Negativbeispiel aktuell konfiguriert.
- [solar panels](/code/python_package/solar_panel_example.ipynb): Das ist ein funktionierendes Beispiel. (Mir ist
  bekannt, dass das Dataset nicht gut ist, aber es war ein gutes Beispiel für den Vortrag.)
- [weather example](/code/python_package/weather_example.ipynb): Das Beispiel ist leider nicht schön geworden was Matrix
  und Ergebnisse angeht, ich wollte es aber drinnen lassen um zu einem späteren Zeitpunkt es zu verbessern.
- [presentation](/code/python_package/presentation.ipynb): Nur Plots für die Präsentation.
- [tutorial](/code/python_package/spice_net_tutorial.ipynb): Hier habe ich angefangen eine Anleitung zur Benutzung des
  Packages zu entwickeln.

## Zukunftspläne

- Ausbau des Tutorials. Ich möchte vor allem noch Erkenntnisse einbringen wie die Hyperparameter am besten eingestellt
  werden sollten und wie die Plots zu interpretieren sind.
- Implementieren der besseren Decoding-Methode.
- Besseres befassen damit wie die Learning-Rates aussehen sollten.
- Deploy des packages nach pip.
- Implementierung in C++ für den Arduino. (Single core, floating point optimiert)

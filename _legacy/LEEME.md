# Codigo archivado

Nada de esta carpeta se ejecuta. Se conserva solo como referencia historica.

- `app (5).py` — el monolito original de 3.777 lineas, anterior a la division
  en modulos. La aplicacion viva es `app.py` + los paquetes `config/`, `core/`,
  `data/`, `ml/`, `ui/` y `utils/`. Tener las dos copias a la vista fue la
  causa de que se editara el archivo equivocado.
- `extract_tabs_and_charts.py`, `append_business.py` — scripts de un solo uso
  que hicieron aquella division. Fue una de estas ejecuciones la que dejo
  fuera cuatro funciones y rompio la app.
- `cold.py` — copia suelta de `cold_start_proxy`, que vive en `ml/engine.py`.

# RNA_Estradas
Procesado de datos e creación dunha Rede Neuronal Artificial para a detección automática de asfalto con imaxes do Sentinel 2.

<h2>Preparación das imaxes:</h2>
<h3>Partimos dunha imaxe de 12 bandas do satélite Sentinel 2 que procesamos con axuda de QGIS 3.12</h3>

A imaxe conta con bandas de tres tamaños: 10x10, 20x20 e 60x60.
Dividimos os píxeles das imaxes de menor resolución espacial para que dotas teñan o mesmo número de píxeles e estes ocupen a mesma posición.

Creamos unha nova capa na que incorporamos píxeles clasificados manualmente como "asfalto" (1) ou "non asfalto" (0) e enchemos os píxeles non clasificados con valor -1.

Exportamos cada unha destas capas como un ficheiro de valores separados por comas (.csv).


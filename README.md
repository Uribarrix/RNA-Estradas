# RNA_Estradas
Procesado de datos e creación dunha Rede Neuronal Artificial para a detección automática de asfalto con imaxes do Sentinel 2.

<h2>Preparación das imaxes:</h2>
<h3>Partimos dunha imaxe de 12 bandas do satélite Sentinel 2 que procesamos con axuda de QGIS 3.12</h3>

A imaxe conta con bandas de tres tamaños: 10x10, 20x20 e 60x60.
Dividimos os píxeles das imaxes de menor resolución espacial para que dotas teñan o mesmo número de píxeles e estes ocupen a mesma posición.

Creamos unha nova capa na que incorporamos píxeles clasificados manualmente como "asfalto" (1) ou "non asfalto" (0) e enchemos os píxeles non clasificados con valor -1.

Exportamos cada unha destas capas como un ficheiro de valores separados por comas (.csv).

Recorte da zona de Xixón, 12 bandas xa en formato .csv:
<link>https://mega.nz/file/cdkjFDTJ#J1w-ZtQ_zBP0CygUhG_G2r0jwzT_4I2qa_MiAZ4rHMY</link>

Seguindo os pasos comentados no código de RNA_Estradas.py, preparamos os datos para seren empregados na rene nueronal.

Configuramos unha rede sinxela de tres capas ocultas e unha de saída.

Levamos a cabo probas de grid search para obter os mellores valores para o batch size, número de épocas e o optimizador.

Realizamos o proceso de validación cruzada para valorar o axuste dos datos con modelo.

Agora facemos unha primeira predición sobre a zona que queremos clasificar. Comprobamos visualmente, montando o .csv da predición coma unha capa máis do proxecto de QGIS, que houbo zonas mal clasificadas e poñemos máis puntos sobre zonas de falso positivo e falso negativo.

Repetimos até que tivemos un resultado satisfactorio.
<img src="https://github.com/Uribarrix/RNA-Estradas/blob/master/mapaBN.jpg"></img>


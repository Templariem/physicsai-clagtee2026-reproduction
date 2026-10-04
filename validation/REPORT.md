# Comprobación Mac / CUDA — 4 de octubre de 2026

La reproducción del paper se ejecutó en el Mac original y en un entorno Python nuevo del homelab. El paquete completo está disponible públicamente en la release clagtee-2026-full-v1.0. Este informe comprueba la implementación; no añade validación física independiente ni cambia las tablas del manuscrito.

## Equipos y alcance

- MacBook M4 Pro, 48 GB de memoria unificada, macOS 26.6.2, Python 3.14.6, PyTorch 2.13.0/MPS; entorno científico existente del paper.
- RTX 5060 Ti, 16 GB VRAM, Linux, Python 3.14.4, PyTorch 2.13.0+cu130; entorno virtual creado desde cero. Driver NVIDIA 595.91.07, runtime CUDA 13.0 y arquitectura `sm_120` reconocida. `pip check`: sin dependencias incompatibles.
- FP32, cuatro hilos CPU, sin AMP ni TF32. Se mantuvieron particiones, calibración, semillas, épocas y parámetros. No se seleccionaron ajustes según el holdout.
- Los pesos oficiales previamente autorizados se copiaron entre las cachés privadas del mismo autor para esta prueba. No se copiaron credenciales ni se incorporaron pesos preentrenados al repositorio. El comando de descarga para terceros consulta los repositorios oficiales y exige sus propias autorizaciones.
- El homelab tenía otro proceso del usuario con aproximadamente 1,3 GiB reservados en la GPU; no se detuvo. El benchmark se ejecutó después de finalizar nuestros entrenamientos.

## Inferencia con los adaptadores del paper

Se regeneraron en CUDA las características de todos los fotogramas y las 14 variantes fotométricas de desarrollo. Los adaptadores, PCA y escaladores guardados se mantuvieron fijos. La comparación sobre el holdout es:

| Modelo | Máxima diferencia de predicción (V) | Diferencia de MAE (V) | Diferencia de R² |
|---|---:|---:|---:|
| Physical | 5.159e-07 | -1.088e-08 | 5.595e-09 |
| DINOv3 | 7.424e-07 | -1.875e-08 | -1.749e-09 |
| I-JEPA | 5.386e-07 | -3.854e-08 | 9.750e-09 |

Los errores de inferencia son menores que 0,000001 V y no alteran las cifras publicadas al redondearlas. `cuda_fresh_features/` contiene las 36 predicciones, resultados por intervalo de calibración, cinco folds y diagnóstico de visión sola. `feature_comparison.json` documenta también las diferencias en embeddings; no se exige identidad bit a bit.

## Entrenamiento desde cero

En MPS, los adaptadores físicos/residuales y los diez modelos del diagnóstico de visión sola reproducen las métricas originales a la precisión numérica de los archivos. En CUDA se repitió el protocolo completo usando las características recién generadas:

| Modelo | R² paper/Mac | R² CUDA reentrenado | MAE paper (V) | MAE CUDA (V) |
|---|---:|---:|---:|---:|
| Physical | 0.988853 | 0.988544 | 0.061967 | 0.064505 |
| DINOv3 | 0.988934 | 0.988742 | 0.057821 | 0.059809 |
| I-JEPA | 0.988488 | 0.988474 | 0.062291 | 0.063521 |

También se entrenaron los modelos de visión sola de la Tabla V; este diagnóstico es más sensible al cambio de backend:

| Modelo, visión sola | R² paper/Mac | R² CUDA | MAE paper (V) | MAE CUDA (V) |
|---|---:|---:|---:|---:|
| DINOv3 | -0.237453 | -0.033245 | 0.596260 | 0.590770 |
| I-JEPA | 0.352197 | 0.320958 | 0.472349 | 0.463138 |

Las diferencias de entrenamiento no deben presentarse como reproducción exacta entre backends. Mantener semillas no iguala necesariamente las secuencias aleatorias y operaciones de MPS y CUDA. Los resultados siguen mostrando una contribución dominante de los inputs físicos y un diagnóstico de visión sola considerablemente más débil. Esta comprobación usa una ejecución por plataforma del protocolo con sus semillas fijas; no estima una distribución de variabilidad entre múltiples repeticiones.

## Tiempos de inferencia

Cinco pasadas por encoder, cinco warm-ups y 36 cuadros por pasada, batch uno y sincronización del dispositivo. Carga, lectura de imágenes, anotación, geometría, FEM y visualización quedan fuera de los tiempos. Se conservan las medidas originales del paper; esta tabla muestra la nueva comprobación.

| Modelo | Mac: ms ± SD | Mac: FPS | CUDA: ms ± SD | CUDA: FPS |
|---|---:|---:|---:|---:|
| DINOv3-S/16 | 12.517 ± 0.322 | 79.89 | 8.688 ± 0.029 | 115.11 |
| I-JEPA-H/14 | 77.208 ± 0.261 | 12.95 | 39.592 ± 0.101 | 25.26 |

Los tiempos reflejan estos equipos y este alcance de medición; no corresponden al rendimiento de una aplicación completa de cámara. Los dos backbones se ejecutan de uno en uno dentro de los 16 GB disponibles.

## FEM, figuras e integridad

- Experimento I: se repitieron los 12 casos de verificación (tres mallas × cuatro regímenes). Los errores coinciden con los originales a precisión numérica; los tiempos se guardan por separado.
- Experimento II: se ejecutaron los cuatro casos y se regeneraron sus campos y composición.
- Experimento III: la solución regenerada coincide exactamente con el input de IA en el Mac; en Linux la máxima diferencia en `U_abs` es 3,13×10⁻⁹ V. La malla coincide exactamente.
- La calibración usa únicamente cinco LED; se ejecutó nuevamente la sensibilidad dejando fuera uno por vez. Los 107 cuadros sin consulta válida mantienen etiquetas NaN.
- Se comprobó la separación por videos de todos los folds y del holdout. El único archivo original omitido del paquete es `frame_086.jpg`, que no entra al loader por carecer de una esfera utilizable; quedan exactamente los 271 fotogramas usados.
- Se corrigió la procedencia de las filas latentes/PCA de la Fig. 5 con autorización del autor: el generador antiguo usaba DINOv2 bajo el rótulo DINOv3. Las nuevas filas utilizan los checkpoints oficiales evaluados. Todos los valores de las 36 matrices de voltaje son idénticos a los anteriores; solo tres píxeles de antialiasing de marcadores difieren en cada fila híbrida. RGB y fila empírica permanecen idénticos.
- A petición del autor se recuperó después la paleta verde/naranja/morado en las dos filas de DINOv3 mediante una transformación RGB fija, documentada en `reference/figures/dino_display_palette.json`. Respecto de la figura corregida anterior, solo cambian los píxeles de las filas 3 y 5; las otras seis filas son idénticas. Los embeddings, PCA y predicciones no cambian. `dino_palette_update.json` registra la comprobación.
- `MANIFEST.sha256` fija datos y referencias; `python -m physicsai.verify` verifica hashes, tamaños del conjunto y particiones. Los generadores escriben en `runs/`, sin sustituir los resultados de referencia.

## Archivos de evidencia

Los subdirectorios `mac_*` y `cuda_*` contienen métricas, predicciones, folds, parámetros del benchmark y entornos. `cuda_requirements_freeze.txt`, `cuda_pip_check.txt` y `cuda_driver.csv` documentan el entorno nuevo. Los entrenamientos completos y sus checkpoints adicionales permanecen también en la carpeta privada de validación del homelab. `commands.md` registra los comandos ejecutados. La release pública incluye el código, los datos y los registros necesarios para reproducir el paper; los pesos preentrenados se obtienen por separado de sus fuentes oficiales.

Vision por Computador — CNN, Transfer Learning, Deteccion y Segmentacion

Dos ejercicios del Curso de Especializacion en Inteligencia Artificial y Big Data (Liceo La Paz) que comparan, de forma sistematica, distintos enfoques de vision por computador: clasificacion de imagenes entrenando una CNN desde cero frente a transfer learning, y deteccion mas segmentacion con modelos preentrenados.

## Clasificacion: CNN desde cero vs transfer learning (MobileNetV2)

Dataset: CIFAR-10 (10 clases, 50.000 imagenes de entrenamiento, 10.000 de test).

Metodologia: un baseline por arquitectura, seguido de cuatro experimentos sistematicos por modelo (learning rate, data augmentation, regularizacion, fine-tuning y scheduler de learning rate), evaluacion final sobre el conjunto de test y analisis de errores con matriz de confusion.

Resultados:

- CNN desde cero — baseline: val_accuracy 0.740. Tras optimizacion (ReduceLROnPlateau): test accuracy 0.759.
- MobileNetV2 (transfer learning) — baseline: val_accuracy 0.813. Tras fine-tuning de las ultimas 30 capas: test accuracy 0.859, un 9.98 pp por encima de la CNN entrenada desde cero.

Analisis de errores: 14.0% de error global sobre el conjunto de validacion (1.403 de 10.000 imagenes). Los pares de clases mas confundidos son semanticamente proximos — perro/gato, rana/pajaro, caballo/ciervo — lo que indica que el modelo aprende representaciones correctas a nivel general pero no discrimina bien los rasgos finos que distinguen clases visualmente similares a baja resolucion. A partir de este analisis se proponen dos mejoras concretas: descongelar mas capas del backbone (60 en vez de 30) con un learning rate mas bajo, y aplicar label smoothing para reducir la sobreconfianza del modelo en sus errores.

## Deteccion y segmentacion: Faster R-CNN y DeepLabV3

Dataset: Penn-Fudan Pedestrian (170 imagenes con mascaras de instancia por peaton).

Faster R-CNN (preentrenado en COCO, backbone ResNet-50 + FPN): mAP@0.5 = 0.9917, mAP@0.5:0.95 = 0.8308, mAR@100 = 0.8627. El umbral de score por defecto (0.5) genera 37 falsos positivos sobre 88 predicciones totales en las 20 imagenes de validacion evaluadas, sobre todo en escenas con varias personas superpuestas.

DeepLabV3 (preentrenado en Pascal VOC, backbone ResNet-50): Dice medio = 0.8929 (desviacion estandar 0.0663) sobre las mismas 20 imagenes. Los casos con peor Dice corresponden a grupos de personas juntas, donde el modelo funde varias siluetas individuales en una unica region sobredimensionada — una limitacion esperable de la segmentacion semantica frente a la segmentacion de instancias.

El calculo de IoU se verifico manualmente paso a paso sobre dos casos (un verdadero positivo y un falso positivo) antes de aplicarlo de forma sistematica, y cada fallo relevante del modelo se documento con su hipotesis de causa y una mejora propuesta.

## Stack

Python, PyTorch, TensorFlow/Keras, torchvision, torchmetrics, scikit-learn, Google Colab (GPU Tesla T4).

## Notebooks

- `pipeline_clasificacion.ipynb` — comparacion sistematica CNN vs MobileNetV2 sobre CIFAR-10.
- `deteccion_segmentacion.ipynb` — Faster R-CNN y DeepLabV3 sobre Penn-Fudan.

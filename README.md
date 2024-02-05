# Proyecto de Clasificación de Imágenes de Animales

Este proyecto utiliza redes neuronales convolucionales (CNN) para clasificar imágenes de animales en tres categorías: perros, gatos y aves. El proyecto está estructurado de manera modular para facilitar su comprensión, desarrollo y mantenimiento.

## Descripción

El proyecto se basa en TensorFlow y Keras para la creación y entrenamiento del modelo de clasificación. Utiliza un conjunto de datos predefinido que se descarga automáticamente al ejecutar el script. Este proyecto es ideal para aquellos que buscan iniciarse en el aprendizaje profundo y la clasificación de imágenes.

## Características

- Descarga automática y preparación de datos.
- Uso de `ImageDataGenerator` para la preparación y aumento de datos.
- Definición de una CNN simple para la clasificación de imágenes.
- Evaluación del modelo y visualización de métricas.

## Estructura del Proyecto

El proyecto se divide en varios módulos:

- `config.py`: Configuraciones generales y rutas.
- `data_management.py`: Gestión de descarga, extracción y preparación de datos.
- `model.py`: Definición del modelo CNN.
- `data_preprocessing.py`: Preparación de datos para entrenamiento y validación.
- `training_evaluation.py`: Entrenamiento y evaluación del modelo.
- `visualization.py`: Visualización de imágenes y métricas.
- `main.py`: Script principal que orquesta la ejecución del proyecto.

## Requisitos

- Python 3.6+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Instalación

Para ejecutar este proyecto, primero clone el repositorio y luego instale las dependencias necesarias.

```bash
git clone https://github.com/monsieurgoodmood/animal_classification
cd animalclassification
pip install -r requirements.txt


## Uso --> Para ejecutar el proyecto, navegue al directorio del proyecto y ejecute:
python main.py
